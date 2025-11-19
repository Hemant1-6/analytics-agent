from typing import Dict, Any, List
import pandas as pd

def determine_widget_type(result: pd.DataFrame, query: str) -> str:
    """
    Analyzes the result DataFrame to determine the best widget type,
    currently prioritizing 'TBL' (Table) and 'CARD' formats.
    """
    # Safety check: if input is not a DataFrame, return CARD (or TBL, depending on what the non-DataFrame input is)
    # Based on the usage in analyze_data, result should be the execution report dict, 
    # but the function signature expects pd.DataFrame. Assuming the caller handles extraction/conversion.
    if not isinstance(result, pd.DataFrame):
        # We'll assume TBL for generic non-DataFrame output to avoid errors.
        # Note: If your analyze_data function passes the result DICT here, 
        # this check will always fail, and you need to adapt the caller.
        return "TBL" 

    rows, cols = result.shape

    # 1. CARD Case (Single metric/scalar output)
    if rows == 1 and cols == 1:
        return "CARD"

    # 2. DEFAULT Case (All other formats are returned as a standard Table)
    # All logic for PIE, VBC, VDBC, and LDRBRD is now removed.
    return "TBL"


def format_data_into_widget(widget_type: str, data: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
    # ... (Rest of the function remains the same, but it will only hit the TBL/CARD branches) ...
    if not data:
        return {"error": "No data to format."}

    base_widget = {
        "widget_width": "100", "widget_height": "auto", "widget_filters": [],
        "widget_tabs": [], "widget_json": [], "action": {"action_seq": "", "action_name": ""},
    }

    if widget_type == "TBL" or widget_type == "LDRBRD":
        # The LDRBRD logic is harmless here, but if you don't want the '#rank' column,
        # remove the 'if widget_type == "LDRBRD"' block entirely.
        
        # Since we only return "TBL" now, LDRBRD logic is technically dead code.
        
        first_record = data[0]
        columns = [{"label": str(key).replace('_', ' ').title(), "key": str(key)} for key in first_record.keys()]
        
        # NOTE: The original TBL logic had conditional LDRBRD rank insertion.
        # We remove that to simplify.

        return {
            **base_widget,
            "widget_placement": 10, # Always TBL placement
            "widget_data": {"columns": columns, "records": data},
            "widget": {"widget_slug": f"WDG_TBL", "widget_title": title, "widget_typeofchart": "TBL"}
        }

    if widget_type == "CARD":
        first_record = data[0]
        value = list(first_record.values())[0]
        key = list(first_record.keys())[0]
        return {
            **base_widget, "widget_width": "50", "widget_placement": 1,
            "widget_data": {
                "columns": [{"label": title, "key": key, "is_amount": True, "is_percent": False}],
                "records": [{"label": title, "data": value}]
            },
            "widget": {"widget_slug": "WDG_CARD", "widget_title": title, "widget_typeofchart": "CARD"}
        }

    # Fallback is now guaranteed to be TBL
    return format_data_into_widget("TBL", data, title)