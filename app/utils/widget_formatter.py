from typing import Dict, Any, List

def format_data_into_widget(widget_type: str, data: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
    """
    Builds the complete widget JSON using the actual data.
    """
    if not data:
        return {"error": "No data to format."}

    base_widget = {
        "widget_width": "100", "widget_height": "auto", "widget_filters": [],
        "widget_tabs": [], "widget_json": [], "action": {"action_seq": "", "action_name": ""},
    }

    if widget_type == "TBL" or widget_type == "LDRBRD":
        if widget_type == "LDRBRD":
            for i, record in enumerate(data):
                record['rank'] = f"#{i + 1}"

        first_record = data[0]
        columns = [{"label": str(key).replace('_', ' ').title(), "key": str(key)} for key in first_record.keys()]
        
        if widget_type == "LDRBRD":
            rank_col = next((c for c in columns if c['key'] == 'rank'), None)
            if rank_col:
                columns.remove(rank_col)
                columns.insert(0, rank_col)

        return {
            **base_widget,
            "widget_placement": 10 if widget_type == "TBL" else 9,
            "widget_data": {"columns": columns, "records": data},
            "widget": {"widget_slug": f"WDG_{widget_type}", "widget_title": title, "widget_typeofchart": widget_type}
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

    if widget_type in ["VBC", "VDBC", "PIE", "VSBC"]:
        first_record = data[0]
        label_key = list(first_record.keys())[0]
        labels = [str(rec.get(label_key)) for rec in data]
        
        datasets = []
        metric_keys = list(first_record.keys())[1:]
        
        colors = ["#05abf3", "#f3b4b7", "#fdbf16", "#4db3e5", "#756fd7", "#e189b5"]

        if widget_type == "PIE":
             datasets.append({
                "label": str(metric_keys[0]).replace('_', ' ').title(),
                "data": [rec.get(metric_keys[0]) for rec in data],
                "backgroundColor": colors,
             })
        else:
            for i, key in enumerate(metric_keys):
                datasets.append({
                    "label": str(key).replace('_', ' ').title(),
                    "data": [rec.get(key) for rec in data],
                    "backgroundColor": colors[i % len(colors)],
                    "type": "bar"
                })

        return {
            **base_widget, "widget_width": "50", "widget_placement": 2,
            "widget_data": {"labels": labels, "datasets": datasets},
            "widget": {"widget_slug": f"WDG_{widget_type}", "widget_title": title, "widget_typeofchart": widget_type}
        }

    # Fallback
    return format_data_into_widget("TBL", data, title)
