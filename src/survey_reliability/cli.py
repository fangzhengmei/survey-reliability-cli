import click
import pandas as pd
import json
from pathlib import Path
from typing import Optional, List

from .reliability import (
    cronbach_alpha,
    calculate_scale_scores,
    generate_report
)


def parse_item_list(ctx, param, value) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",")]


@click.group()
@click.version_option(version="0.1.0")
def main():
    """问卷量表信度分析 CLI 工具"""
    pass


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--items", "-i",
    required=True,
    callback=parse_item_list,
    help="题项列名，用逗号分隔（如：q1,q2,q3,q4,q5）"
)
@click.option(
    "--reverse-items", "-r",
    callback=parse_item_list,
    help="需要反向计分的题项列名，用逗号分隔"
)
@click.option(
    "--min-val",
    type=int,
    default=1,
    help="最小可能值（默认：1）"
)
@click.option(
    "--max-val",
    type=int,
    default=5,
    help="最大可能值（默认：5）"
)
@click.option(
    "--missing-method",
    type=click.Choice(["mean", "median", "drop"]),
    default="mean",
    help="缺失值处理方法（默认：mean）"
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False),
    help="输出报告文件路径"
)
@click.option(
    "--json-output",
    type=click.Path(dir_okay=False),
    help="输出 JSON 格式结果的文件路径"
)
@click.option(
    "--scale-name",
    default="量表",
    help="量表名称（用于报告标题）"
)
def alpha(
    input_file,
    items,
    reverse_items,
    min_val,
    max_val,
    missing_method,
    output,
    json_output,
    scale_name
):
    """
    计算 Cronbach's α 系数
    
    INPUT_FILE: 输入数据文件路径（支持 CSV、Excel）
    
    示例：
        survey-reliability alpha data.csv -i q1,q2,q3,q4,q5
        survey-reliability alpha data.csv -i q1,q2,q3 -r q3 --min-val 1 --max-val 7
    """
    input_path = Path(input_file)
    
    if input_path.suffix.lower() in [".csv"]:
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    else:
        raise click.ClickException(f"不支持的文件格式: {input_path.suffix}")
    
    for item in items:
        if item not in df.columns:
            raise click.ClickException(f"题项列不存在: {item}")
    
    if reverse_items:
        for item in reverse_items:
            if item not in items:
                raise click.ClickException(f"反向题项 {item} 不在题项列表中")
    
    result = cronbach_alpha(
        df=df,
        item_cols=items,
        reverse_items=reverse_items,
        min_val=min_val,
        max_val=max_val,
        missing_method=missing_method
    )
    
    report = generate_report(result, scale_name)
    
    click.echo(report)
    
    if output:
        output_path = Path(output)
        output_path.write_text(report, encoding="utf-8")
        click.echo(f"\n报告已保存至: {output_path}")
    
    if json_output:
        json_path = Path(json_output)
        json_result = result.copy()
        json_result["items"] = items
        json_result["reverse_items"] = reverse_items
        json_path.write_text(
            json.dumps(json_result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8"
        )
        click.echo(f"JSON 结果已保存至: {json_path}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--items", "-i",
    required=True,
    callback=parse_item_list,
    help="题项列名，用逗号分隔（如：q1,q2,q3,q4,q5）"
)
@click.option(
    "--reverse-items", "-r",
    callback=parse_item_list,
    help="需要反向计分的题项列名，用逗号分隔"
)
@click.option(
    "--min-val",
    type=int,
    default=1,
    help="最小可能值（默认：1）"
)
@click.option(
    "--max-val",
    type=int,
    default=5,
    help="最大可能值（默认：5）"
)
@click.option(
    "--missing-method",
    type=click.Choice(["mean", "median", "drop"]),
    default="mean",
    help="缺失值处理方法（默认：mean）"
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(dir_okay=False),
    help="输出文件路径（支持 CSV、Excel）"
)
@click.option(
    "--keep-processed/--no-keep-processed",
    default=True,
    help="是否保留处理后的题项列（默认：保留）"
)
def score(
    input_file,
    items,
    reverse_items,
    min_val,
    max_val,
    missing_method,
    output,
    keep_processed
):
    """
    计算量表得分
    
    INPUT_FILE: 输入数据文件路径（支持 CSV、Excel）
    
    输出将包含：
    - 原始数据
    - 处理后的题项得分（可选，通过 --keep-processed 控制）
    - scale_total: 总分
    - scale_mean: 平均分
    
    示例：
        survey-reliability score data.csv -i q1,q2,q3,q4,q5 -o scored.csv
        survey-reliability score data.csv -i q1,q2,q3 -r q3 -o scored.xlsx
    """
    input_path = Path(input_file)
    output_path = Path(output)
    
    if input_path.suffix.lower() in [".csv"]:
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    else:
        raise click.ClickException(f"不支持的文件格式: {input_path.suffix}")
    
    for item in items:
        if item not in df.columns:
            raise click.ClickException(f"题项列不存在: {item}")
    
    if reverse_items:
        for item in reverse_items:
            if item not in items:
                raise click.ClickException(f"反向题项 {item} 不在题项列表中")
    
    result = calculate_scale_scores(
        df=df,
        item_cols=items,
        reverse_items=reverse_items,
        min_val=min_val,
        max_val=max_val,
        missing_method=missing_method
    )
    
    if not keep_processed:
        processed_cols = [f"{col}_processed" for col in items]
        result = result.drop(columns=processed_cols)
    
    if output_path.suffix.lower() in [".csv"]:
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif output_path.suffix.lower() in [".xlsx", ".xls"]:
        result.to_excel(output_path, index=False)
    else:
        raise click.ClickException(f"不支持的输出格式: {output_path.suffix}")
    
    click.echo(f"量表得分已保存至: {output_path}")
    click.echo(f"\n输出列说明:")
    click.echo(f"  - scale_total: 量表总分")
    click.echo(f"  - scale_mean: 量表平均分")
    if keep_processed:
        click.echo(f"  - {{item}}_processed: 处理后的题项得分（反向计分后）")
    
    click.echo(f"\n统计摘要:")
    click.echo(f"  有效样本数: {len(result)}")
    click.echo(f"  总分范围: {result['scale_total'].min():.2f} - {result['scale_total'].max():.2f}")
    click.echo(f"  总分均值: {result['scale_total'].mean():.2f} (SD={result['scale_total'].std():.2f})")
    click.echo(f"  平均分范围: {result['scale_mean'].min():.2f} - {result['scale_mean'].max():.2f}")
    click.echo(f"  平均分均值: {result['scale_mean'].mean():.2f} (SD={result['scale_mean'].std():.2f})")


if __name__ == "__main__":
    main()
