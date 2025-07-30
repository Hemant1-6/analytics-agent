from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.concurrency import run_in_threadpool
import pandas as pd
import json
from typing import Dict, Any

from app.models.analytics import QueryRequest, AnalyticsResponse, DatasetInfo
from app.services import llm_service, data_service
from app.core.cache import DATASET_CACHE
from app.utils.widget_formatter import format_data_into_widget, determine_widget_type

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.post("/upload", status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")
    
    try:
        df = pd.read_csv(file.file)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        dataset_id = file.filename
        DATASET_CACHE[dataset_id] = df
        return {"success": True, "dataset_id": dataset_id, "message": "Dataset uploaded and cached successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {e}")

@router.get("/dataset/{dataset_id}", response_model=DatasetInfo)
async def get_dataset_info(dataset_id: str):
    df = DATASET_CACHE.get(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found. Please upload it first.")
    
    return {
        "file_name": dataset_id,
        "columns": df.columns.tolist(),
        "shape": list(df.shape),
        "sample_data": df.head().to_dict(orient="records"),
    }

@router.post("/analyze", response_model=AnalyticsResponse)
async def analyze_data(request: QueryRequest):
    df = DATASET_CACHE.get(request.dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found. Please upload it first.")

    try:
        df_info_str = await run_in_threadpool(data_service.get_dataframe_info, df)
        
        # Step 1: LLM generates only the pandas code
        code_response = await run_in_threadpool(
            llm_service.generate_code, request.query, df_info_str
        )
        pandas_code = code_response.get("pandas_code")
        if not pandas_code:
            raise ValueError("LLM failed to generate pandas code.")

        # Step 2: Execute the code to get the real data result
        raw_result = await run_in_threadpool(data_service.execute_pandas_code, df, pandas_code)
        
        # Step 3: Python logic determines the best widget type from the actual result
        widget_type = determine_widget_type(raw_result, request.query)
        
        # Step 4: A separate, simple LLM call generates a clean title
        title_response = await run_in_threadpool(llm_service.generate_title, request.query)
        widget_title = title_response.get("widget_title", "Data Analysis")

        # Step 5: Serialize the result and format the final widget JSON
        formatted_data = data_service.serialize_result(raw_result)
        final_widget_json = format_data_into_widget(widget_type, formatted_data, widget_title)

        # Step 6: Generate summary and suggestions
        summary_sample_data = json.dumps(formatted_data[:20])
        summary_suggestions_response = await run_in_threadpool(
            llm_service.generate_summary_and_suggestions, request.query, summary_sample_data, df_info_str
        )
        
        description = summary_suggestions_response.get("summary")
        suggestions = summary_suggestions_response.get("suggestions")
        
        return AnalyticsResponse(
            success=True,
            description=description,
            chart_type=widget_type,
            suggested_chart_config=final_widget_json,
            proactive_suggestions=suggestions,
            executed_code=pandas_code
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
