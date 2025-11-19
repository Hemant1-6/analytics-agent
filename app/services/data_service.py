import pandas as pd
import numpy as np
from asteval import Interpreter
import logging
import io
from typing import Any
import traceback
from pydantic import BaseModel
import ast

logger = logging.getLogger(__name__)

def execute_pandas_code(df: pd.DataFrame, code: str) -> Any:
    try:
        aeval = Interpreter()
        aeval.symtable['df'] = df
        aeval.symtable['pd'] = pd
        
        aeval.eval(code)
        logger.debug(f"Executed code:\n{code}")
        result = aeval.symtable.get('result')
        if result is None:
            raise ValueError("Code did not produce a 'result' variable.")
        return result
    except Exception as e:
        # logger.error(f"Code execution failed: {e}\nCode: {code}")
        logger.error(exec_code_with_debug(code, df))
        raise ValueError(f"Code execution failed: {e}\nCode: {code}")
        

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

# -----------------------
# Diagnostic Helper
# -----------------------
def exec_code_with_debug(code_str: str, df: pd.DataFrame):
    """Syntax-check and execute generated pandas code with detailed diagnostics."""
    # 1) Syntax check
    try:
        ast.parse(code_str)
    except SyntaxError as se:
        return {
            "success": False,
            "error_type": "SyntaxError",
            "message": str(se),
            "syntax_info": {
                "lineno": se.lineno,
                "offset": se.offset,
                "text": se.text.strip() if se.text else None,
            },
        }

    ns = {"df": df, "pd": pd}

    # 2) Runtime execution
    try:
        exec(code_str, ns)
    except Exception as e:
        return {
            "success": False,
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
            "namespace_summary": {k: type(v).__name__ for k, v in ns.items() if k != "__builtins__"},
        }

    # 3) Success — inspect result
    result_exists = "result" in ns
    result_preview = None

    if result_exists:
        res = ns["result"]
        if hasattr(res, "head"):
            try:
                result_preview = res.head(5).to_dict(orient="records")
            except:
                result_preview = repr(res)
        else:
            result_preview = repr(res)

    return {
        "success": True,
        "result_exists": result_exists,
        "result_preview": result_preview,
        "namespace_summary": {k: type(v).__name__ for k, v in ns.items() if k != "__builtins__"},
    }


# -----------------------
# Request Model
# -----------------------
class CodeInput(BaseModel):
    code: str
