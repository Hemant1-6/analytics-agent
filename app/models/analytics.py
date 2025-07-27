from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class QueryRequest(BaseModel):
    dataset_id: str
    query: str

class AnalyticsResponse(BaseModel):
    success: bool
    description: Optional[str] = None
    chart_type: Optional[str] = None
    suggested_chart_config: Optional[Dict[str, Any]] = None
    proactive_suggestions: Optional[List[str]] = None
    executed_code: Optional[str] = None
    error: Optional[str] = None

class DatasetInfo(BaseModel):
    file_name: str
    columns: List[str]
    shape: List[int]
    sample_data: List[Dict[str, Any]]

# Pydantic models for LangChain's JSON output parser
class CodeChartResponse(BaseModel):
    pandas_code: str = Field(description="The pandas code to execute.")
    widget_typeofchart: str = Field(description="The type of widget to display.")
    widget_title: str = Field(description="A concise, descriptive title for the chart or table.")

class SummarySuggestionsResponse(BaseModel):
    summary: str = Field(description="A summary of the data insights.")
    suggestions: List[str] = Field(description="A list of follow-up questions.")