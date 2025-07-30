import pandas as pd
import numpy as np
from asteval import Interpreter
import logging
import io
from typing import Any

logger = logging.getLogger(__name__)

def execute_pandas_code(df: pd.DataFrame, code: str) -> Any:
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
    if isinstance(result, pd.DataFrame):
        if isinstance(result.index, pd.MultiIndex):
            result = result.reset_index()
        
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(map(str, col)).strip() for col in result.columns.values]
            
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
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    return f"""
    Columns and Data Types:
    {info_str}

    First 5 rows:
    {df.head().to_string()}
    """
