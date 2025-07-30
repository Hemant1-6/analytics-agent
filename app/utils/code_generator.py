from typing import Dict, Any

def generate_pandas_code(plan: Dict[str, Any]) -> str:
    """
    Deterministically generates pandas code from a structured query plan.
    """
    metrics = plan.get('metrics', [])
    dimensions = plan.get('dimensions', [])
    analysis_type = plan.get('analysis_type')

    if not metrics or not dimensions:
        # Handle simple cases like "total sales"
        if metrics and not dimensions and analysis_type == 'aggregation':
             return f"result = pd.DataFrame({{{'total_{m}': [df['{m}'].sum()] for m in metrics}}})"
        # Fallback for more complex queries without clear metrics/dimensions
        if analysis_type == 'distribution' and len(dimensions) > 1:
             # Heuristic for queries like "distribution of sales values for each ship mode"
             grouped_cols = ", ".join([f"'{d}'" for d in dimensions])
             return f"result = df.groupby([{grouped_cols}])['{metrics[0]}'].value_counts().reset_index(name='count')"
        return "" # Cannot generate code

    if analysis_type == 'aggregation':
        agg_funcs = {m: "'sum'" for m in metrics}
        agg_str = ", ".join([f"{k}={v}" for k, v in agg_funcs.items()])
        dim_str = ", ".join([f"'{d}'" for d in dimensions])
        return f"result = df.groupby([{dim_str}]).agg({agg_str}).reset_index()"
        
    if analysis_type == 'distribution':
        dim_str = ", ".join([f"'{d}'" for d in dimensions])
        return f"result = df.groupby([{dim_str}])['{metrics[0]}'].value_counts().reset_index(name='count')"

    return "" # Fallback for unsupported types
