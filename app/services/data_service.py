import pandas as pd
import numpy as np
from asteval import Interpreter
import logging
import io
from typing import Any

logger = logging.getLogger(__name__)

def execute_pandas_code(df: pd.DataFrame, code: str) -> Any:
    """Safely execute pandas code using asteval."""
    try:
        aeval = Interpreter()
        aeval.symtable['df'] = df
        aeval.symtable['pd'] = pd
        
        aeval.eval(code)
        
        result = aeval.symtable.get('result')
        if result is None:
            raise ValueError("Code did not produce a 'result' variable.")
        return result
    except Exception as e:
        logger.error(f"Code execution failed: {e}\nCode: {code}")
        raise ValueError(f"Code execution failed: {e}")

def serialize_result(result: Any) -> Any:
    """Formats analysis result into a JSON-serializable structure."""
    if isinstance(result, pd.DataFrame):
        # FIX: Ensure that if the result has an index, it's converted to a column.
        # This prevents the index (e.g., 'category') from being lost during serialization.
        if result.index.name is not None and result.index.name not in result.columns:
            result = result.reset_index()
        return result.to_dict(orient="records")
    
    elif isinstance(result, pd.Series):
        return result.reset_index().to_dict(orient="records")
    
    elif isinstance(result, (int, float, str, bool, np.number)):
        return {"value": result}
        
    elif isinstance(result, dict):
        return result
        
    else:
        return {"data": str(result)}

def get_dataframe_info(df: pd.DataFrame) -> str:
    """Generates a string summary of a DataFrame for the LLM prompt."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    return f"""
    Columns and Data Types:
    {info_str}

    First 5 rows:
    {df.head().to_string()}
    """