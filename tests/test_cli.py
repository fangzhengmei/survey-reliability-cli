import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from click.testing import CliRunner

from survey_reliability.cli import main


class TestCLIBasic:
    def setup_method(self):
        self.runner = CliRunner()
        
        np.random.seed(42)
        n = 50
        latent = np.random.normal(0, 1, n)
        self.item_cols = ["q1", "q2", "q3", "q4", "q5"]
        
        data = {"id": list(range(1, n + 1))}
        for i, col in enumerate(self.item_cols):
            if i == 2 or i == 4:
                data[col] = (-latent * 0.8 + np.random.normal(0, 0.2, n)).round()
            else:
                data[col] = (latent * 0.8 + np.random.normal(0, 0.2, n)).round()
            data[col] = np.clip(data[col], 1, 5)
        
        self.df = pd.DataFrame(data)
    
    def test_alpha_command_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            
            result = self.runner.invoke(
                main,
                [
                    "alpha",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols)
                ]
            )
            
            assert result.exit_code == 0
            assert "Cronbach's α" in result.output
            assert "样本量" in result.output
            assert "题项数" in result.output
    
    def test_alpha_command_excel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path = Path(tmpdir) / "test_data.xlsx"
            self.df.to_excel(xlsx_path, index=False)
            
            result = self.runner.invoke(
                main,
                [
                    "alpha",
                    str(xlsx_path),
                    "-i",
                    ",".join(self.item_cols)
                ]
            )
            
            assert result.exit_code == 0
            assert "Cronbach's α" in result.output
    
    def test_alpha_command_with_reverse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            
            result_with_reverse = self.runner.invoke(
                main,
                [
                    "alpha",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "-r",
                    "q3,q5",
                    "--min-val",
                    "1",
                    "--max-val",
                    "5"
                ]
            )
            
            result_without_reverse = self.runner.invoke(
                main,
                [
                    "alpha",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols)
                ]
            )
            
            assert result_with_reverse.exit_code == 0
            assert result_without_reverse.exit_code == 0
            
            def extract_alpha(output):
                for line in output.split("\n"):
                    if "Cronbach's α" in line:
                        return float(line.split(":")[1].strip())
                return None
            
            alpha_with = extract_alpha(result_with_reverse.output)
            alpha_without = extract_alpha(result_without_reverse.output)
            
            assert alpha_with is not None
            assert alpha_without is not None
            assert alpha_with > alpha_without
    
    def test_alpha_command_with_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            output_path = Path(tmpdir) / "report.txt"
            
            result = self.runner.invoke(
                main,
                [
                    "alpha",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "-o",
                    str(output_path)
                ]
            )
            
            assert result.exit_code == 0
            assert output_path.exists()
            
            report_content = output_path.read_text(encoding="utf-8")
            assert "Cronbach's α" in report_content
    
    def test_alpha_command_with_json_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            json_path = Path(tmpdir) / "result.json"
            
            result = self.runner.invoke(
                main,
                [
                    "alpha",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "--json-output",
                    str(json_path)
                ]
            )
            
            assert result.exit_code == 0
            assert json_path.exists()
            
            json_content = json.loads(json_path.read_text(encoding="utf-8"))
            assert "alpha" in json_content
            assert "item_count" in json_content
            assert "sample_size" in json_content
            assert "items" in json_content
    
    def test_alpha_command_invalid_item(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            
            result = self.runner.invoke(
                main,
                [
                    "alpha",
                    str(csv_path),
                    "-i",
                    "q1,q2,q100"
                ]
            )
            
            assert result.exit_code != 0
            assert "题项列不存在" in result.output
    
    def test_alpha_command_invalid_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "test_data.txt"
            txt_path.write_text("q1,q2\n1,2\n3,4", encoding="utf-8")
            
            result = self.runner.invoke(
                main,
                [
                    "alpha",
                    str(txt_path),
                    "-i",
                    "q1,q2"
                ]
            )
            
            assert result.exit_code != 0
            assert "不支持的文件格式" in result.output
    
    def test_alpha_command_invalid_range_min_gt_max(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            
            result = self.runner.invoke(
                main,
                [
                    "alpha",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "--min-val",
                    "5",
                    "--max-val",
                    "1"
                ]
            )
            
            assert result.exit_code != 0
            assert "min-val" in result.output
            assert "max-val" in result.output
    
    def test_alpha_command_invalid_range_min_eq_max(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            
            result = self.runner.invoke(
                main,
                [
                    "alpha",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "--min-val",
                    "3",
                    "--max-val",
                    "3"
                ]
            )
            
            assert result.exit_code != 0
            assert "min-val" in result.output
            assert "max-val" in result.output
    
    def test_alpha_command_values_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df_invalid = self.df.copy()
            df_invalid.loc[0, "q1"] = 0
            df_invalid.loc[1, "q2"] = 6
            
            csv_path = Path(tmpdir) / "test_data.csv"
            df_invalid.to_csv(csv_path, index=False)
            
            result = self.runner.invoke(
                main,
                [
                    "alpha",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "--min-val",
                    "1",
                    "--max-val",
                    "5"
                ]
            )
            
            assert result.exit_code != 0
            assert "超出量表范围" in result.output


class TestCLIScore:
    def setup_method(self):
        self.runner = CliRunner()
        
        np.random.seed(42)
        n = 30
        latent = np.random.normal(0, 1, n)
        self.item_cols = ["q1", "q2", "q3", "q4", "q5"]
        
        data = {"id": list(range(1, n + 1))}
        for i, col in enumerate(self.item_cols):
            if i == 2:
                data[col] = (-latent * 0.8 + np.random.normal(0, 0.2, n)).round()
            else:
                data[col] = (latent * 0.8 + np.random.normal(0, 0.2, n)).round()
            data[col] = np.clip(data[col], 1, 5)
        
        self.df = pd.DataFrame(data)
    
    def test_score_command_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            output_path = Path(tmpdir) / "scored.csv"
            
            result = self.runner.invoke(
                main,
                [
                    "score",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "-o",
                    str(output_path)
                ]
            )
            
            assert result.exit_code == 0
            assert output_path.exists()
            
            scored_df = pd.read_csv(output_path)
            
            assert "scale_total" in scored_df.columns
            assert "scale_mean" in scored_df.columns
            assert "scale_total" in result.output
            assert "scale_mean" in result.output
            
            expected_total = scored_df[self.item_cols].sum(axis=1)
            pd.testing.assert_series_equal(
                scored_df["scale_total"],
                expected_total,
                check_names=False
            )
    
    def test_score_command_with_reverse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            output_path = Path(tmpdir) / "scored.csv"
            
            result = self.runner.invoke(
                main,
                [
                    "score",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "-r",
                    "q3",
                    "--min-val",
                    "1",
                    "--max-val",
                    "5",
                    "-o",
                    str(output_path)
                ]
            )
            
            assert result.exit_code == 0
            
            scored_df = pd.read_csv(output_path)
            
            assert "q3_processed" in scored_df.columns
            
            q3_reversed = 5 + 1 - scored_df["q3"]
            pd.testing.assert_series_equal(
                scored_df["q3_processed"],
                q3_reversed,
                check_names=False
            )
    
    def test_score_command_no_keep_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            output_path = Path(tmpdir) / "scored.csv"
            
            result = self.runner.invoke(
                main,
                [
                    "score",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "-r",
                    "q3",
                    "--no-keep-processed",
                    "-o",
                    str(output_path)
                ]
            )
            
            assert result.exit_code == 0
            
            scored_df = pd.read_csv(output_path)
            
            for col in self.item_cols:
                assert f"{col}_processed" not in scored_df.columns
    
    def test_score_command_excel_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            output_path = Path(tmpdir) / "scored.xlsx"
            
            result = self.runner.invoke(
                main,
                [
                    "score",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "-o",
                    str(output_path)
                ]
            )
            
            assert result.exit_code == 0
            assert output_path.exists()
            
            scored_df = pd.read_excel(output_path)
            assert "scale_total" in scored_df.columns
            assert "scale_mean" in scored_df.columns
    
    def test_score_command_with_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df_with_missing = self.df.copy()
            df_with_missing.loc[0, "q1"] = np.nan
            df_with_missing.loc[5, "q3"] = np.nan
            
            csv_path = Path(tmpdir) / "test_data.csv"
            df_with_missing.to_csv(csv_path, index=False)
            output_path = Path(tmpdir) / "scored.csv"
            
            result = self.runner.invoke(
                main,
                [
                    "score",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "--missing-method",
                    "mean",
                    "-o",
                    str(output_path)
                ]
            )
            
            assert result.exit_code == 0
            
            scored_df = pd.read_csv(output_path)
            
            assert scored_df["scale_total"].isna().sum() == 0
            assert scored_df["scale_mean"].isna().sum() == 0
    
    def test_score_command_missing_drop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df_with_missing = self.df.copy()
            df_with_missing.loc[0, "q1"] = np.nan
            df_with_missing.loc[5, "q3"] = np.nan
            
            csv_path = Path(tmpdir) / "test_data.csv"
            df_with_missing.to_csv(csv_path, index=False)
            output_path = Path(tmpdir) / "scored.csv"
            
            result = self.runner.invoke(
                main,
                [
                    "score",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "--missing-method",
                    "drop",
                    "-o",
                    str(output_path)
                ]
            )
            
            assert result.exit_code == 0
            
            scored_df = pd.read_csv(output_path)
            
            assert len(scored_df) == len(self.df) - 2
    
    def test_score_command_invalid_range_min_gt_max(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            output_path = Path(tmpdir) / "scored.csv"
            
            result = self.runner.invoke(
                main,
                [
                    "score",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "--min-val",
                    "5",
                    "--max-val",
                    "1",
                    "-o",
                    str(output_path)
                ]
            )
            
            assert result.exit_code != 0
            assert "min-val" in result.output
            assert "max-val" in result.output
    
    def test_score_command_invalid_range_min_eq_max(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.df.to_csv(csv_path, index=False)
            output_path = Path(tmpdir) / "scored.csv"
            
            result = self.runner.invoke(
                main,
                [
                    "score",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "--min-val",
                    "3",
                    "--max-val",
                    "3",
                    "-o",
                    str(output_path)
                ]
            )
            
            assert result.exit_code != 0
            assert "min-val" in result.output
            assert "max-val" in result.output
    
    def test_score_command_values_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df_invalid = self.df.copy()
            df_invalid.loc[0, "q1"] = 0
            df_invalid.loc[1, "q2"] = 6
            
            csv_path = Path(tmpdir) / "test_data.csv"
            df_invalid.to_csv(csv_path, index=False)
            output_path = Path(tmpdir) / "scored.csv"
            
            result = self.runner.invoke(
                main,
                [
                    "score",
                    str(csv_path),
                    "-i",
                    ",".join(self.item_cols),
                    "--min-val",
                    "1",
                    "--max-val",
                    "5",
                    "-o",
                    str(output_path)
                ]
            )
            
            assert result.exit_code != 0
            assert "超出量表范围" in result.output
