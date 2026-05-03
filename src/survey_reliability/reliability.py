import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


def validate_scale_range(
    min_val: int,
    max_val: int
) -> None:
    """
    校验量表范围的合法性
    
    Args:
        min_val: 最小可能值
        max_val: 最大可能值
        
    Raises:
        ValueError: 当 min_val > max_val 或 min_val == max_val 时
    """
    if min_val > max_val:
        raise ValueError(
            f"量表范围不合法：min-val ({min_val}) 不能大于 max-val ({max_val})"
        )
    if min_val == max_val:
        raise ValueError(
            f"量表范围不合法：min-val ({min_val}) 不能等于 max-val ({max_val})，量表至少需要两个不同的值"
        )


def validate_item_values(
    df: pd.DataFrame,
    item_cols: List[str],
    min_val: int,
    max_val: int,
    strict: bool = True
) -> Optional[Dict[str, Any]]:
    """
    校验题项分数是否在量表范围内
    
    Args:
        df: 数据框
        item_cols: 题项列名列表
        min_val: 最小可能值
        max_val: 最大可能值
        strict: 是否严格模式（严格模式下超出范围会抛出异常，否则返回警告信息）
        
    Returns:
        如果非严格模式且有超出范围的值，返回包含警告信息的字典；否则返回 None
        
    Raises:
        ValueError: 严格模式下，当题项分数超出 [min_val, max_val] 范围时
    """
    validation_results = {
        "out_of_range_items": [],
        "total_out_of_range_count": 0
    }
    
    for col in item_cols:
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        below_min = col_data < min_val
        above_max = col_data > max_val
        
        below_count = int(below_min.sum())
        above_count = int(above_max.sum())
        
        if below_count > 0 or above_count > 0:
            min_observed = float(col_data.min())
            max_observed = float(col_data.max())
            
            validation_results["out_of_range_items"].append({
                "item": col,
                "below_min_count": below_count,
                "above_max_count": above_count,
                "min_observed": min_observed,
                "max_observed": max_observed
            })
            validation_results["total_out_of_range_count"] += below_count + above_count
    
    if validation_results["total_out_of_range_count"] > 0:
        if strict:
            error_details = []
            for item_info in validation_results["out_of_range_items"]:
                details = []
                if item_info["below_min_count"] > 0:
                    details.append(f"{item_info['below_min_count']} 个值小于 {min_val}")
                if item_info["above_max_count"] > 0:
                    details.append(f"{item_info['above_max_count']} 个值大于 {max_val}")
                details_str = "，".join(details)
                error_details.append(
                    f"题项 '{item_info['item']}': {details_str} "
                    f"(观测范围: {item_info['min_observed']} - {item_info['max_observed']})"
                )
            
            raise ValueError(
                f"题项分数超出量表范围 [min-val={min_val}, max-val={max_val}]，"
                f"共发现 {validation_results['total_out_of_range_count']} 个异常值。\n"
                + "\n".join(error_details)
            )
        else:
            return validation_results
    
    return None


def reverse_score(
    values: np.ndarray,
    min_val: int,
    max_val: int
) -> np.ndarray:
    """
    反向计分：将原始分数转换为反向分数
    
    公式：max_val + min_val - original_value
    
    Args:
        values: 原始分数数组
        min_val: 最小可能值
        max_val: 最大可能值
        
    Returns:
        反向计分后的数组
    """
    return max_val + min_val - values


def process_reverse_items(
    df: pd.DataFrame,
    item_cols: List[str],
    reverse_items: Optional[List[str]] = None,
    min_val: int = 1,
    max_val: int = 5
) -> pd.DataFrame:
    """
    处理反向题项
    
    Args:
        df: 数据框
        item_cols: 所有题项列名列表
        reverse_items: 需要反向计分的题项列名列表
        min_val: 最小可能值
        max_val: 最大可能值
        
    Returns:
        处理后的数据框
    """
    validate_scale_range(min_val, max_val)
    
    processed = df.copy()
    
    if reverse_items is None:
        reverse_items = []
    
    for item in reverse_items:
        if item in item_cols and item in processed.columns:
            processed[item] = reverse_score(processed[item].values, min_val, max_val)
    
    return processed


def impute_missing(
    df: pd.DataFrame,
    item_cols: List[str],
    method: str = "mean"
) -> pd.DataFrame:
    """
    缺失值插补
    
    Args:
        df: 数据框
        item_cols: 题项列名列表
        method: 插补方法 ('mean', 'median', 'drop')
            - 'mean': 用该题项的平均值插补
            - 'median': 用该题项的中位数插补
            - 'drop': 删除含有缺失值的行
            
    Returns:
        处理后的数据框
    """
    processed = df.copy()
    
    if method == "drop":
        return processed.dropna(subset=item_cols)
    
    for col in item_cols:
        if col in processed.columns:
            if method == "mean":
                fill_value = processed[col].mean()
            elif method == "median":
                fill_value = processed[col].median()
            else:
                raise ValueError(f"不支持的插补方法: {method}")
            
            processed[col] = processed[col].fillna(fill_value)
    
    return processed


def calculate_scale_scores(
    df: pd.DataFrame,
    item_cols: List[str],
    reverse_items: Optional[List[str]] = None,
    min_val: int = 1,
    max_val: int = 5,
    missing_method: str = "mean"
) -> pd.DataFrame:
    """
    计算量表得分
    
    流程：
    1. 校验量表范围合法性
    2. 校验题项分数范围
    3. 缺失值插补
    4. 反向题处理
    5. 计算总分和平均分
    
    Args:
        df: 数据框
        item_cols: 所有题项列名列表
        reverse_items: 需要反向计分的题项列名列表
        min_val: 最小可能值
        max_val: 最大可能值
        missing_method: 缺失值插补方法 ('mean', 'median', 'drop')
        
    Returns:
        含有量表得分的新数据框，包含：
        - 原始数据
        - 处理后的题项得分（反向计分后）
        - scale_total: 总分
        - scale_mean: 平均分
    """
    validate_scale_range(min_val, max_val)
    validate_item_values(df, item_cols, min_val, max_val)
    
    result = df.copy()
    
    result = impute_missing(result, item_cols, missing_method)
    
    processed_items = process_reverse_items(result, item_cols, reverse_items, min_val, max_val)
    
    for col in item_cols:
        result[f"{col}_processed"] = processed_items[col]
    
    result["scale_total"] = processed_items[item_cols].sum(axis=1)
    result["scale_mean"] = processed_items[item_cols].mean(axis=1)
    
    return result


def cronbach_alpha(
    df: pd.DataFrame,
    item_cols: List[str],
    reverse_items: Optional[List[str]] = None,
    min_val: int = 1,
    max_val: int = 5,
    missing_method: str = "mean"
) -> Dict[str, Any]:
    """
    计算 Cronbach's α 系数
    
    公式：
    α = (k / (k - 1)) * (1 - (Σσ²_i / σ²_total))
    
    其中：
    - k: 题项数量
    - σ²_i: 第 i 个题项的方差
    - σ²_total: 总分的方差
    
    Args:
        df: 数据框
        item_cols: 所有题项列名列表
        reverse_items: 需要反向计分的题项列名列表
        min_val: 最小可能值
        max_val: 最大可能值
        missing_method: 缺失值插补方法 ('mean', 'median', 'drop')
        
    Returns:
        包含以下信息的字典：
        - alpha: Cronbach's α 系数
        - item_count: 题项数量
        - sample_size: 有效样本量
        - item_stats: 每个题项的统计信息
        - alpha_if_deleted: 删除该题项后的 α 系数
    """
    validate_scale_range(min_val, max_val)
    validate_item_values(df, item_cols, min_val, max_val)
    
    data = df.copy()
    data = impute_missing(data, item_cols, missing_method)
    data = process_reverse_items(data, item_cols, reverse_items, min_val, max_val)
    
    items_data = data[item_cols]
    
    k = len(item_cols)
    n = len(items_data)
    
    if k < 2:
        raise ValueError("Cronbach's α 计算至少需要 2 个题项")
    
    if n < 2:
        raise ValueError("有效样本量不足，无法计算 α 系数")
    
    item_variances = items_data.var(ddof=1, axis=0)
    total_scores = items_data.sum(axis=1)
    total_variance = total_scores.var(ddof=1)
    
    if total_variance == 0:
        return {
            "alpha": 0.0,
            "item_count": k,
            "sample_size": n,
            "item_stats": {},
            "alpha_if_deleted": {},
            "note": "总分方差为 0，所有被试得分相同"
        }
    
    alpha = (k / (k - 1)) * (1 - (item_variances.sum() / total_variance))
    
    item_stats = {}
    alpha_if_deleted = {}
    
    for item in item_cols:
        other_items = [col for col in item_cols if col != item]
        
        item_stats[item] = {
            "mean": items_data[item].mean(),
            "std": items_data[item].std(ddof=1),
            "min": items_data[item].min(),
            "max": items_data[item].max(),
            "count": items_data[item].count(),
            "is_reverse": reverse_items is not None and item in reverse_items
        }
        
        if len(other_items) >= 2:
            other_data = items_data[other_items]
            other_item_vars = other_data.var(ddof=1, axis=0)
            other_total_scores = other_data.sum(axis=1)
            other_total_var = other_total_scores.var(ddof=1)
            
            if other_total_var > 0:
                alpha_deleted = (
                    (len(other_items) / (len(other_items) - 1)) *
                    (1 - (other_item_vars.sum() / other_total_var))
                )
            else:
                alpha_deleted = 0.0
        else:
            alpha_deleted = None
        
        alpha_if_deleted[item] = alpha_deleted
    
    return {
        "alpha": float(alpha),
        "item_count": k,
        "sample_size": n,
        "item_stats": item_stats,
        "alpha_if_deleted": alpha_if_deleted
    }


def generate_report(
    result: Dict[str, Any],
    scale_name: str = "量表"
) -> str:
    """
    生成信度分析报告
    
    Args:
        result: cronbach_alpha 返回的结果字典
        scale_name: 量表名称
        
    Returns:
        格式化的报告字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"信度分析报告 - {scale_name}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"样本量: {result['sample_size']}")
    lines.append(f"题项数: {result['item_count']}")
    lines.append(f"Cronbach's α: {result['alpha']:.4f}")
    lines.append("")
    
    if result['alpha'] >= 0.9:
        interpretation = "优秀 (Excellent)"
    elif result['alpha'] >= 0.8:
        interpretation = "良好 (Good)"
    elif result['alpha'] >= 0.7:
        interpretation = "可接受 (Acceptable)"
    elif result['alpha'] >= 0.6:
        interpretation = "有疑问 (Questionable)"
    else:
        interpretation = "较差 (Poor)"
    
    lines.append(f"α 系数解释: {interpretation}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("题项统计信息")
    lines.append("-" * 60)
    lines.append(f"{'题项':<15} {'均值':<10} {'标准差':<10} {'删除后α':<12} {'反向计分':<10}")
    lines.append("-" * 60)
    
    for item, stats in result['item_stats'].items():
        alpha_deleted = result['alpha_if_deleted'].get(item)
        alpha_deleted_str = f"{alpha_deleted:.4f}" if alpha_deleted is not None else "N/A"
        reverse_str = "是" if stats.get('is_reverse') else "否"
        
        lines.append(
            f"{item:<15} {stats['mean']:<10.2f} "
            f"{stats['std']:<10.2f} {alpha_deleted_str:<12} {reverse_str:<10}"
        )
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)
