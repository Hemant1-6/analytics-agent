import logging
from typing import Dict, Any
from app.core.config import settings
from app.models.analytics import CodeChartResponse, SummarySuggestionsResponse

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

# Initialize the Gemini model via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.GEMINI_API_KEY)

def generate_code_and_chart_type(query: str, df_info: str) -> Dict[str, Any]:
    """
    Analyzes the query to generate pandas code and select the best widget type using LangChain and Gemini.
    """
    parser = JsonOutputParser(pydantic_object=CodeChartResponse)

    prompt = PromptTemplate(
        template="""
You are an expert data analyst AI. Your task is to interpret a user's query and provide the tools to answer it by following a strict process.

**DataFrame Info:**
{df_info}

**User Query:**
"{query}"

**Instructions:**

**Step 1: Deconstruct the Query**
- Identify the metric(s) the user wants to measure (e.g., 'sales', 'profit', 'quantity').
- Identify the category or dimension(s) for grouping (e.g., 'region', 'segment', 'category').
- Count the number of metrics and the number of grouping dimensions.

**Step 2: Select a `widget_typeofchart`**
Based on your analysis in Step 1, choose the most appropriate `widget_typeofchart` from the list below. This is a critical step.
- **`CARD`**: Use for a single metric with NO grouping dimensions (e.g., "total sales").
- **`PIE`**: Use for a single metric and a single grouping dimension, ONLY if the query contains keywords like "breakdown", "share", "composition", or "proportions".
- **`VBC`**: Use for a single metric and a single grouping dimension (e.g., "sales by region").
- **`VDBC`**: Use for TWO metrics and a single grouping dimension (e.g., "sales and profit by category").
- **`LDRBRD`**: Use for ranked lists (e.g., "top 10...").
- **`TBL`**: Use for TWO OR MORE grouping dimensions (e.g., "sales by region and category") or for high-cardinality data that won't fit a chart.

**Step 3: Generate `pandas_code`**
- Write the Python pandas code that correctly calculates the metrics and groupings identified in Step 1.
- The final result **MUST** be stored in a variable named `result`.
- **CRITICAL**: If you use `groupby`, you **MUST** chain `.reset_index()` at the end. For example: `result = df.groupby('category')['sales'].sum().reset_index()`
- The `result` should be a DataFrame.
- Do NOT include `import pandas as pd`.

**Step 4: Final Output**
Return a single, valid JSON object with the `pandas_code` and `widget_typeofchart` you determined in the steps above.

{format_instructions}
""",
        input_variables=["query", "df_info"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    
    try:
        return chain.invoke({"query": query, "df_info": df_info})
    except Exception as e:
        logger.error(f"LangChain invocation failed for code generation: {e}")
        raise ValueError("Failed to generate analysis from LangChain.")

def generate_summary_and_suggestions(query: str, data: str) -> Dict[str, Any]:
    """
    Generates a summary and proactive suggestions from a query and its resulting data using LangChain and Gemini.
    """
    parser = JsonOutputParser(pydantic_object=SummarySuggestionsResponse)

    prompt = PromptTemplate(
        template="""
You are a data analyst AI. Your task is to provide a concise, data-driven summary and suggest next steps.

The user asked: "{query}"
This produced the data: {data}

{format_instructions}
""",
        input_variables=["query", "data"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser

    try:
        return chain.invoke({"query": query, "data": data})
    except Exception as e:
        logger.error(f"LangChain invocation failed for summary generation: {e}")
        return {
            "summary": "A summary of the results could not be generated.",
            "suggestions": []
        }