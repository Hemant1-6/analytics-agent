import logging
from typing import Dict, Any
from app.core.config import settings
from app.models.analytics import CodeResponse, TitleResponse, SummarySuggestionsResponse

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

# Initialize the OpenAI model via LangChain
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=settings.OPENAI_API_KEY)

# --- Initialize Local LLM via Ollama's OpenAI-Compatible API ---
# llm = ChatOpenAI(
#     # The name of the model you pulled locally
#     model="phi3:mini", 
    
#     # Point the base URL to your local Ollama server
#     base_url="http://localhost:11434/v1",
    
#     # API key is required but can be a placeholder string for local service
#     api_key="ollama" 
# )

def generate_code(query: str, df_info: str) -> Dict[str, Any]:
    parser = JsonOutputParser(pydantic_object=CodeResponse)
    prompt = PromptTemplate(
        template="""
### DataFrame Context & Schema
The DataFrame is loaded from a CSV and has the following structure and key business columns:

| DataFrame Column Name | Business Meaning / Aliases (Use these for interpretation) | Data Type Hint |
| :--- | :--- | :--- |
| **bizline** | Product / Business Line / Loan Type | Categorical (String) |
| **vendor_name** | DSA / Channel Partner / Vendor Name (Note: Names are NOT unique) | Categorical (String) |
| **vendor_code** | DSA / Channel Partner / Vendor Code | Categorical (String) |
| **acct_number** | Loan Account ID | String (Unique Identifier) |
| **total_disb_amount** | Disbursement Value / Disb Amt / **Disbursed Amount** (Use this for value aggregation) | Numeric (Float) |
| **insurance_fee** | Insurance Amount / Insurance Fee / Cross Sell Amount | Numeric (Float) |
| **payout_amount** | Payout Amount / Commission Amount / DSA Payout | Numeric (Float) |
| **cycle_start_date** | Cycle Start Date (Used for Time Analysis) | DateTime |
| *Other columns may be present based on the schema dump below.* | *Interpretation should be derived from the User Query.* | *Varies* |

**Current DataFrame Info (Schema Dump):**
{df_info}

### User Query
"{query}"

###  Code Generation Instructions

1.  **Output Variable:** The final resulting DataFrame **MUST** be stored in a variable named `result`.
2.  **Imports:** **DO NOT** include `import pandas as pd`. Assume the DataFrame is already loaded as `df`.
3.  **Clarity & Structure:** Write the code clearly. You may use intermediate variables (e.g., `temp_df`) if the calculation is complex, but the final line must assign to `result`.
4.  **Aggregation Rule (CRITICAL):**
    * Any use of `.groupby()` **MUST** include all necessary identifying columns (`vendor_code`, `vendor_name`, time period) in the grouping list.
    * The grouping **MUST** be followed by a `.reset_index()` to return a flat DataFrame.
    * **DO NOT** manually add identifying columns back to the result later; group them from the start.
5.  **Handling Edge Cases & Column Naming (CRITICAL):**
    * You **MUST** use the **lowercase snake_case column names** as listed in the schema table (e.g., **`total_disb_amount`**).
    * **For any filtering operation that creates a subset of the DataFrame which is immediately modified (e.g., adding a new column), you MUST use the `.copy()` method to explicitly create an independent DataFrame (e.g., `new_df = df[...].copy()`) to prevent execution errors.**
    * **When performing column-wise arithmetic (`diff`, `+`, `-`), you MUST first slice the DataFrame to include ONLY the numeric columns** (e.g., using `.iloc[:, 2:]` after pivoting) to prevent errors like `'float' and 'str'`.
    * If a required column is missing, assume the request cannot be fulfilled and set `result = pd.DataFrame('Error': ['Column(s) required for the query are missing based on the df_info.'])`.
    * For ranking or top/bottom N queries, use `.nlargest()` or `.nsmallest()` where appropriate, or a sort and slice.
6.  **Complex Calculation Formulas (Prioritize these definitions):**
    * **Distribution/Frequency:** Use `.value_counts()` or a `groupby().size()` and then `reset_index(name='Count')`.
7.  **Date/Time Handling (MOM/QOQ) (CRITICAL FIX):**
    * **For any calculation requiring time-based grouping, you MUST use the column 'cycle_start_date'.**
    * **For conversion, you MUST use `pd.to_datetime(df['cycle_start_date'], format='%d/%m/%y', errors='coerce')`** to ensure robustness against mixed formats.
    * For MOM/QOQ analysis, you **MUST** use the Indian Financial Year quarter system by using the frequency code **`'Q-MAR'`** (e.g., `dt.to_period('Q-MAR')`). Also, **dynamically determine** the comparison periods by sorting the unique periods.
8.  **Variance Magnitude (Highest/Biggest):**
    * For general variance, use the **absolute value** of the difference (`.abs()`) before ranking using `.nlargest(1, 'Abs Variance Column')`.
9.  **Directional Variance (Fall/Rise):**
    * For **"sharpest fall"** (a decrease), calculate the signed difference (**Q2 - Q1**) and use **`.nsmallest(1)`**.
    * For **"sharpest rise"** (an increase), calculate the signed difference (**Q2 - Q1**) and use **`.nlargest(1)`**.
10. **DSA/Vendor Identification (CRITICAL FIX):** For accurate analysis, the primary aggregation key **MUST** include **`vendor_code` and `vendor_name`** alongside the time period.11. **Final Output Presentation (CRITICAL):** The final result DataFrame MUST contain both the unique identifier (**vendor_code**) and the descriptive name (**vendor_name**). Include the amounts for the comparison periods (e.g., Q1 Amt, Q2 Amt) alongside the difference.
12. **Serialization Fix (CRITICAL):**
    * Any column that is a time period or datetime object **in the rows** MUST be converted to a standard string representation using **`.astype(str)`**.
    * **After any pivot operation, the column labels (index) that represent time periods MUST be converted to strings using `.columns.astype(str)` before the final assignment to `result` to ensure JSON compatibility.**

### Final Output Format (CRITICAL)

**You MUST respond ONLY with a single JSON object that adheres to the following format instructions. DO NOT include any explanatory text, markdown outside the JSON block, or extra characters.**

{format_instructions}

""",
        input_variables=["query", "df_info"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "query": query, 
            "df_info": df_info,
        })
    except Exception as e:
        logger.error(f"LangChain invocation failed for code generation: {e}")
        # Add a specific check for JSON parsing errors if possible, otherwise raise
        if "Invalid json output" in str(e):
             # This suggests the model failed to follow the format instructions
             # You might add logic here to retry or refine the prompt with a sample JSON
             pass 
        raise ValueError("Failed to generate pandas code from LangChain: {e}")
    
    
def generate_title(query: str) -> Dict[str, Any]:
    parser = JsonOutputParser(pydantic_object=TitleResponse)
    prompt = PromptTemplate(
        template="""
Based on the user's query, create a short, descriptive title for a chart or table. For example, if the query is "What are the total sales for each customer segment?", a good title would be "Total Sales by Customer Segment".

**User Query:**
"{query}"

{format_instructions}
""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    try:
        return chain.invoke({"query": query})
    except Exception as e:
        logger.error(f"LangChain invocation failed for title generation: {e}")
        return {"widget_title": "Data Analysis"} # Fallback title

def generate_summary_and_suggestions(query: str, data: str, df_info: str) -> Dict[str, Any]:
    parser = JsonOutputParser(pydantic_object=SummarySuggestionsResponse)
    prompt = PromptTemplate(
        template="""
You are a data analyst AI. Your task is to provide a concise, data-driven summary and suggest next steps.

The user asked: "{query}"
This produced the data: {data}

**CRITICAL INSTRUCTION:**
You must generate 3 relevant follow-up questions a data analyst might ask next. These suggestions **MUST** be answerable using the original dataset. The available columns are described below. Do not suggest questions that would require external data or research.

**Available Data Info:**
{df_info}

{format_instructions}
""",
        input_variables=["query", "data", "df_info"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    try:
        return chain.invoke({"query": query, "data": data, "df_info": df_info})
    except Exception as e:
        logger.error(f"LangChain invocation failed for summary generation: {e}")
        return {
            "summary": "A summary of the results could not be generated.",
            "suggestions": []
        }
