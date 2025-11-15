import logging
from typing import Dict, Any
from app.core.config import settings
from app.models.analytics import CodeResponse, TitleResponse, SummarySuggestionsResponse

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

# Initialize the OpenAI model via LangChain
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=settings.OPENAI_API_KEY)

def generate_code(query: str, df_info: str) -> Dict[str, Any]:
    parser = JsonOutputParser(pydantic_object=CodeResponse)
    prompt = PromptTemplate(
        template="""
You are an expert data analyst AI. Your task is to interpret a user's query and generate the correct Python pandas code to answer it.

**DataFrame Info:**
{df_info}

**User Query:**
"{query}"

**Instructions:**
- Your primary goal is to write the pandas code that correctly performs the analysis.
- For complex calculations like 'profit margin', the formula is `(profit / sales) * 100`. You must perform the aggregation first, then define the new column.
- For "distribution" queries, you should count the occurrences of each value. A good approach is `df.groupby(['dimension_column'])['metric_column'].value_counts().reset_index(name='count')`.
- The final result **MUST** be stored in a variable named `result`.
- **CRITICAL**: If you use `groupby`, you **MUST** chain `.reset_index()` at the end.
- The `result` should be a DataFrame.
- Do NOT include `import pandas as pd`.
- Consider Bizline name as Product / Business Line / Loan Type.
- Consider Payee name as (DSA / Channel Partner / Vendor) Name, also different vendors can have same name . 
- Consider payee code as (DSA / Channel Partner / Vendor) Code
- Consider ACCT_NUMBER as loan account id 
- Consider NET_DISB_AMT as disbusement value / disb amt/ disbursed amount
- Consider INSURANCE_FEE as insurance amount / insurance fee / cross sell amount
- Consider Payout Amount as payout amount / commission amount / dsa payout

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
        raise ValueError("Failed to generate pandas code from LangChain.")

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
