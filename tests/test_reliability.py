import pytest
import pandas as pd
import numpy as np
from survey_reliability.reliability import (
    reverse_score,
    process_reverse_items,
    impute_missing,
    calculate_scale_scores,
    cronbach_alpha,
    generate_report,
    validate_scale_range,
    validate_item_values
)


class TestValidateScaleRange:
    def test_validate_scale_range_min_greater_than_max(self):
        with pytest.raises(ValueError, match="min-val.*不能大于 max-val"):
            validate_scale_range(min_val=5, max_val=1)
    
    def test_validate_scale_range_min_greater_than_max_2(self):
        with pytest.raises(ValueError) as exc_info:
            validate_scale_range(min_val=7, max_val=5)
        assert "min-val (7) 不能大于 max-val (5)" in str(exc_info.value)
    
    def test_validate_scale_range_min_equals_max(self):
        with pytest.raises(ValueError, match="min-val.*不能等于 max-val"):
            validate_scale_range(min_val=3, max_val=3)
    
    def test_validate_scale_range_min_equals_max_2(self):
        with pytest.raises(ValueError) as exc_info:
            validate_scale_range(min_val=5, max_val=5)
        assert "min-val (5) 不能等于 max-val (5)" in str(exc_info.value)
    
    def test_validate_scale_range_valid(self):
        validate_scale_range(min_val=1, max_val=5)
        validate_scale_range(min_val=1, max_val=7)
        validate_scale_range(min_val=0, max_val=10)
        validate_scale_range(min_val=-3, max_val=3)


class TestValidateItemValues:
    def test_validate_item_values_below_min(self):
        df = pd.DataFrame({
            "q1": [0, 2, 3, 4, 5],
            "q2": [1, 2, 3, 4, 5]
        })
        item_cols = ["q1", "q2"]
        
        with pytest.raises(ValueError, match="题项分数超出量表范围"):
            validate_item_values(df, item_cols, min_val=1, max_val=5)
    
    def test_validate_item_values_above_max(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3, 4, 5],
            "q2": [1, 2, 6, 4, 5]
        })
        item_cols = ["q1", "q2"]
        
        with pytest.raises(ValueError) as exc_info:
            validate_item_values(df, item_cols, min_val=1, max_val=5)
        
        assert "q2" in str(exc_info.value)
        assert "大于 5" in str(exc_info.value)
    
    def test_validate_item_values_both_below_and_above(self):
        df = pd.DataFrame({
            "q1": [0, 2, 3, 4, 6],
            "q2": [1, 2, 3, 4, 5]
        })
        item_cols = ["q1", "q2"]
        
        with pytest.raises(ValueError) as exc_info:
            validate_item_values(df, item_cols, min_val=1, max_val=5)
        
        error_msg = str(exc_info.value)
        assert "q1" in error_msg
        assert "小于 1" in error_msg
        assert "大于 5" in error_msg
    
    def test_validate_item_values_multiple_items(self):
        df = pd.DataFrame({
            "q1": [0, 2, 3, 4, 5],
            "q2": [1, 2, 3, 4, 6],
            "q3": [1, 2, 3, 4, 5]
        })
        item_cols = ["q1", "q2", "q3"]
        
        with pytest.raises(ValueError) as exc_info:
            validate_item_values(df, item_cols, min_val=1, max_val=5)
        
        error_msg = str(exc_info.value)
        assert "q1" in error_msg
        assert "q2" in error_msg
    
    def test_validate_item_values_boundary_values(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3, 4, 5],
            "q2": [1, 1, 5, 5, 3]
        })
        item_cols = ["q1", "q2"]
        
        result = validate_item_values(df, item_cols, min_val=1, max_val=5)
        assert result is None
    
    def test_validate_item_values_with_nan(self):
        df = pd.DataFrame({
            "q1": [1, np.nan, 3, 4, 5],
            "q2": [np.nan, 2, 3, np.nan, 5]
        })
        item_cols = ["q1", "q2"]
        
        result = validate_item_values(df, item_cols, min_val=1, max_val=5)
        assert result is None
    
    def test_validate_item_values_non_strict_mode(self):
        df = pd.DataFrame({
            "q1": [0, 2, 3, 4, 6],
            "q2": [1, 2, 3, 4, 5]
        })
        item_cols = ["q1", "q2"]
        
        result = validate_item_values(df, item_cols, min_val=1, max_val=5, strict=False)
        
        assert result is not None
        assert "out_of_range_items" in result
        assert "total_out_of_range_count" in result
        assert len(result["out_of_range_items"]) == 1
        assert result["total_out_of_range_count"] == 2
        
        q1_info = result["out_of_range_items"][0]
        assert q1_info["item"] == "q1"
        assert q1_info["below_min_count"] == 1
        assert q1_info["above_max_count"] == 1
        assert q1_info["min_observed"] == 0
        assert q1_info["max_observed"] == 6
    
    def test_validate_item_values_all_valid(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3, 4, 5],
            "q2": [2, 3, 4, 5, 1],
            "q3": [3, 3, 3, 3, 3]
        })
        item_cols = ["q1", "q2", "q3"]
        
        result = validate_item_values(df, item_cols, min_val=1, max_val=5)
        assert result is None
    
    def test_validate_item_values_different_range(self):
        df = pd.DataFrame({
            "q1": [0, 5, 7, 10, 3],
            "q2": [1, 2, 11, 4, 5]
        })
        item_cols = ["q1", "q2"]
        
        with pytest.raises(ValueError) as exc_info:
            validate_item_values(df, item_cols, min_val=1, max_val=10)
        
        error_msg = str(exc_info.value)
        assert "q1" in error_msg or "q2" in error_msg
        assert "小于 1" in error_msg or "大于 10" in error_msg


class TestReverseScore:
    def test_reverse_score_basic(self):
        values = np.array([1, 2, 3, 4, 5])
        result = reverse_score(values, min_val=1, max_val=5)
        expected = np.array([5, 4, 3, 2, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_reverse_score_middle(self):
        values = np.array([3, 3, 3])
        result = reverse_score(values, min_val=1, max_val=5)
        expected = np.array([3, 3, 3])
        np.testing.assert_array_equal(result, expected)
    
    def test_reverse_score_different_range(self):
        values = np.array([1, 3, 5, 7])
        result = reverse_score(values, min_val=1, max_val=7)
        expected = np.array([7, 5, 3, 1])
        np.testing.assert_array_equal(result, expected)


class TestProcessReverseItems:
    def test_process_reverse_items_basic(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3, 4, 5],
            "q2": [1, 2, 3, 4, 5],
            "q3": [1, 2, 3, 4, 5]
        })
        item_cols = ["q1", "q2", "q3"]
        reverse_items = ["q2"]
        
        result = process_reverse_items(df, item_cols, reverse_items, min_val=1, max_val=5)
        
        assert list(result["q1"]) == [1, 2, 3, 4, 5]
        assert list(result["q2"]) == [5, 4, 3, 2, 1]
        assert list(result["q3"]) == [1, 2, 3, 4, 5]
    
    def test_process_reverse_items_no_reverse(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3],
            "q2": [4, 5, 1]
        })
        item_cols = ["q1", "q2"]
        
        result = process_reverse_items(df, item_cols, reverse_items=None)
        
        pd.testing.assert_frame_equal(result, df)
    
    def test_process_reverse_items_original_unchanged(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3],
            "q2": [4, 5, 1]
        })
        original = df.copy()
        item_cols = ["q1", "q2"]
        reverse_items = ["q2"]
        
        process_reverse_items(df, item_cols, reverse_items, min_val=1, max_val=5)
        
        pd.testing.assert_frame_equal(df, original)
    
    def test_process_reverse_items_invalid_range_min_gt_max(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3],
            "q2": [4, 5, 1]
        })
        item_cols = ["q1", "q2"]
        reverse_items = ["q2"]
        
        with pytest.raises(ValueError, match="min-val.*不能大于 max-val"):
            process_reverse_items(df, item_cols, reverse_items, min_val=5, max_val=1)
    
    def test_process_reverse_items_invalid_range_min_eq_max(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3],
            "q2": [4, 5, 1]
        })
        item_cols = ["q1", "q2"]
        reverse_items = ["q2"]
        
        with pytest.raises(ValueError, match="min-val.*不能等于 max-val"):
            process_reverse_items(df, item_cols, reverse_items, min_val=3, max_val=3)


class TestImputeMissing:
    def test_impute_missing_mean(self):
        df = pd.DataFrame({
            "q1": [1, 2, np.nan, 4, 5],
            "q2": [np.nan, 2, 3, np.nan, 5]
        })
        item_cols = ["q1", "q2"]
        
        result = impute_missing(df, item_cols, method="mean")
        
        assert result["q1"].isna().sum() == 0
        assert result["q2"].isna().sum() == 0
        
        expected_q1_mean = (1 + 2 + 4 + 5) / 4
        assert result["q1"].iloc[2] == expected_q1_mean
        
        expected_q2_mean = (2 + 3 + 5) / 3
        assert result["q2"].iloc[0] == expected_q2_mean
        assert result["q2"].iloc[3] == expected_q2_mean
    
    def test_impute_missing_median(self):
        df = pd.DataFrame({
            "q1": [1, 2, np.nan, 4, 10],
            "q2": [np.nan, 2, 3, np.nan, 100]
        })
        item_cols = ["q1", "q2"]
        
        result = impute_missing(df, item_cols, method="median")
        
        assert result["q1"].isna().sum() == 0
        assert result["q2"].isna().sum() == 0
        
        assert result["q1"].iloc[2] == 3.0
        assert result["q2"].iloc[0] == 3.0
    
    def test_impute_missing_drop(self):
        df = pd.DataFrame({
            "q1": [1, 2, np.nan, 4, 5],
            "q2": [1, np.nan, 3, 4, 5]
        })
        item_cols = ["q1", "q2"]
        
        result = impute_missing(df, item_cols, method="drop")
        
        assert len(result) == 3
        assert list(result["q1"]) == [1.0, 4.0, 5.0]
        assert list(result["q2"]) == [1.0, 4.0, 5.0]
    
    def test_impute_missing_invalid_method(self):
        df = pd.DataFrame({"q1": [1, 2, 3]})
        item_cols = ["q1"]
        
        with pytest.raises(ValueError, match="不支持的插补方法"):
            impute_missing(df, item_cols, method="invalid")
    
    def test_impute_missing_original_unchanged(self):
        df = pd.DataFrame({
            "q1": [1, np.nan, 3],
            "q2": [4, 5, np.nan]
        })
        original = df.copy()
        item_cols = ["q1", "q2"]
        
        impute_missing(df, item_cols, method="mean")
        
        pd.testing.assert_frame_equal(df, original)


class TestCalculateScaleScores:
    def test_calculate_scale_scores_basic(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "q1": [1, 2, 3],
            "q2": [4, 5, 1],
            "q3": [2, 3, 4]
        })
        item_cols = ["q1", "q2", "q3"]
        
        result = calculate_scale_scores(df, item_cols)
        
        assert "scale_total" in result.columns
        assert "scale_mean" in result.columns
        
        assert list(result["scale_total"]) == [7, 10, 8]
        assert list(result["scale_mean"]) == [7/3, 10/3, 8/3]
    
    def test_calculate_scale_scores_with_reverse(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "q1": [1, 2, 3],
            "q2": [4, 5, 1],
            "q3": [2, 3, 4]
        })
        item_cols = ["q1", "q2", "q3"]
        reverse_items = ["q2"]
        
        result = calculate_scale_scores(
            df, item_cols, reverse_items, min_val=1, max_val=5
        )
        
        q2_processed = result["q2_processed"]
        assert list(q2_processed) == [2, 1, 5]
        
        assert result["scale_total"].iloc[0] == 1 + 2 + 2
        assert result["scale_total"].iloc[1] == 2 + 1 + 3
        assert result["scale_total"].iloc[2] == 3 + 5 + 4
    
    def test_calculate_scale_scores_with_missing(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "q1": [1, np.nan, 3, 4],
            "q2": [2, 3, np.nan, 5],
            "q3": [3, 4, 5, np.nan]
        })
        item_cols = ["q1", "q2", "q3"]
        
        result = calculate_scale_scores(df, item_cols, missing_method="mean")
        
        assert "scale_total" in result.columns
        assert result["scale_total"].isna().sum() == 0
        
        q1_mean = (1 + 3 + 4) / 3
        q2_mean = (2 + 3 + 5) / 3
        q3_mean = (3 + 4 + 5) / 3
        
        assert result["q1_processed"].iloc[1] == pytest.approx(q1_mean)
        assert result["q2_processed"].iloc[2] == pytest.approx(q2_mean)
        assert result["q3_processed"].iloc[3] == pytest.approx(q3_mean)
    
    def test_calculate_scale_scores_original_unchanged(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "q1": [1, 2, 3],
            "q2": [4, 5, 1]
        })
        original = df.copy()
        item_cols = ["q1", "q2"]
        
        calculate_scale_scores(df, item_cols)
        
        pd.testing.assert_frame_equal(df, original)
    
    def test_calculate_scale_scores_invalid_range_min_gt_max(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "q1": [1, 2, 3],
            "q2": [4, 5, 1]
        })
        item_cols = ["q1", "q2"]
        
        with pytest.raises(ValueError, match="min-val.*不能大于 max-val"):
            calculate_scale_scores(df, item_cols, min_val=5, max_val=1)
    
    def test_calculate_scale_scores_invalid_range_min_eq_max(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "q1": [1, 2, 3],
            "q2": [4, 5, 1]
        })
        item_cols = ["q1", "q2"]
        
        with pytest.raises(ValueError, match="min-val.*不能等于 max-val"):
            calculate_scale_scores(df, item_cols, min_val=3, max_val=3)
    
    def test_calculate_scale_scores_values_below_min(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "q1": [0, 2, 3],
            "q2": [4, 5, 1]
        })
        item_cols = ["q1", "q2"]
        
        with pytest.raises(ValueError, match="题项分数超出量表范围"):
            calculate_scale_scores(df, item_cols, min_val=1, max_val=5)
    
    def test_calculate_scale_scores_values_above_max(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "q1": [1, 2, 3],
            "q2": [4, 6, 1]
        })
        item_cols = ["q1", "q2"]
        
        with pytest.raises(ValueError, match="题项分数超出量表范围"):
            calculate_scale_scores(df, item_cols, min_val=1, max_val=5)


class TestCronbachAlpha:
    def test_cronbach_alpha_basic(self):
        np.random.seed(42)
        n = 100
        latent = np.random.normal(0, 1, n)
        item_cols = ["q1", "q2", "q3", "q4", "q5"]
        
        data = {}
        for i, col in enumerate(item_cols):
            data[col] = (latent * 0.8 + np.random.normal(0, 0.2, n)).round()
            data[col] = np.clip(data[col], 1, 5)
        
        df = pd.DataFrame(data)
        
        result = cronbach_alpha(df, item_cols)
        
        assert "alpha" in result
        assert "item_count" in result
        assert "sample_size" in result
        assert "item_stats" in result
        assert "alpha_if_deleted" in result
        
        assert result["item_count"] == 5
        assert result["sample_size"] == 100
        assert 0 < result["alpha"] < 1
        
        for col in item_cols:
            assert col in result["item_stats"]
            assert col in result["alpha_if_deleted"]
            assert "mean" in result["item_stats"][col]
            assert "std" in result["item_stats"][col]
    
    def test_cronbach_alpha_with_reverse(self):
        np.random.seed(42)
        n = 100
        latent = np.random.normal(0, 1, n)
        item_cols = ["q1", "q2", "q3", "q4", "q5"]
        reverse_items = ["q3", "q5"]
        
        data = {}
        for i, col in enumerate(item_cols):
            if col in reverse_items:
                data[col] = (-latent * 0.8 + np.random.normal(0, 0.2, n)).round()
            else:
                data[col] = (latent * 0.8 + np.random.normal(0, 0.2, n)).round()
            data[col] = np.clip(data[col], 1, 5)
        
        df = pd.DataFrame(data)
        
        result_with_reverse = cronbach_alpha(
            df, item_cols, reverse_items=reverse_items, min_val=1, max_val=5
        )
        result_without_reverse = cronbach_alpha(df, item_cols)
        
        assert result_with_reverse["alpha"] > result_without_reverse["alpha"]
        
        for col in item_cols:
            assert result_with_reverse["item_stats"][col]["is_reverse"] == (col in reverse_items)
    
    def test_cronbach_alpha_with_missing(self):
        np.random.seed(42)
        n = 50
        latent = np.random.normal(0, 1, n)
        item_cols = ["q1", "q2", "q3", "q4"]
        
        data = {}
        for i, col in enumerate(item_cols):
            data[col] = (latent * 0.7 + np.random.normal(0, 0.3, n)).round()
            data[col] = np.clip(data[col], 1, 5)
        
        df = pd.DataFrame(data)
        df.loc[0, "q1"] = np.nan
        df.loc[5, "q2"] = np.nan
        df.loc[10, "q3"] = np.nan
        
        result = cronbach_alpha(df, item_cols, missing_method="mean")
        
        assert result["sample_size"] == 50
        assert 0 < result["alpha"] < 1
        
        result_drop = cronbach_alpha(df, item_cols, missing_method="drop")
        
        assert result_drop["sample_size"] == 47
    
    def test_cronbach_alpha_insufficient_items(self):
        df = pd.DataFrame({"q1": [1, 2, 3, 4, 5]})
        
        with pytest.raises(ValueError, match="至少需要 2 个题项"):
            cronbach_alpha(df, ["q1"])
    
    def test_cronbach_alpha_insufficient_samples(self):
        df = pd.DataFrame({
            "q1": [1],
            "q2": [2]
        })
        
        with pytest.raises(ValueError, match="有效样本量不足"):
            cronbach_alpha(df, ["q1", "q2"])
    
    def test_cronbach_alpha_zero_variance(self):
        df = pd.DataFrame({
            "q1": [3, 3, 3, 3],
            "q2": [3, 3, 3, 3],
            "q3": [3, 3, 3, 3]
        })
        
        result = cronbach_alpha(df, ["q1", "q2", "q3"])
        
        assert result["alpha"] == 0.0
        assert "note" in result
    
    def test_cronbach_alpha_invalid_range_min_gt_max(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3, 4, 5],
            "q2": [1, 2, 3, 4, 5]
        })
        item_cols = ["q1", "q2"]
        
        with pytest.raises(ValueError, match="min-val.*不能大于 max-val"):
            cronbach_alpha(df, item_cols, min_val=5, max_val=1)
    
    def test_cronbach_alpha_invalid_range_min_eq_max(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3, 4, 5],
            "q2": [1, 2, 3, 4, 5]
        })
        item_cols = ["q1", "q2"]
        
        with pytest.raises(ValueError, match="min-val.*不能等于 max-val"):
            cronbach_alpha(df, item_cols, min_val=3, max_val=3)
    
    def test_cronbach_alpha_values_below_min(self):
        df = pd.DataFrame({
            "q1": [0, 2, 3, 4, 5],
            "q2": [1, 2, 3, 4, 5]
        })
        item_cols = ["q1", "q2"]
        
        with pytest.raises(ValueError, match="题项分数超出量表范围"):
            cronbach_alpha(df, item_cols, min_val=1, max_val=5)
    
    def test_cronbach_alpha_values_above_max(self):
        df = pd.DataFrame({
            "q1": [1, 2, 3, 4, 5],
            "q2": [1, 2, 6, 4, 5]
        })
        item_cols = ["q1", "q2"]
        
        with pytest.raises(ValueError, match="题项分数超出量表范围"):
            cronbach_alpha(df, item_cols, min_val=1, max_val=5)
    
    def test_cronbach_alpha_boundary_values(self):
        df = pd.DataFrame({
            "q1": [1, 1, 5, 5, 3],
            "q2": [1, 5, 1, 5, 3],
            "q3": [3, 3, 3, 3, 3]
        })
        item_cols = ["q1", "q2", "q3"]
        
        result = cronbach_alpha(df, item_cols, min_val=1, max_val=5)
        
        assert "alpha" in result
        assert result["sample_size"] == 5


class TestGenerateReport:
    def test_generate_report_basic(self):
        result = {
            "alpha": 0.85,
            "item_count": 5,
            "sample_size": 100,
            "item_stats": {
                "q1": {"mean": 3.5, "std": 1.2, "is_reverse": False},
                "q2": {"mean": 2.8, "std": 1.5, "is_reverse": True},
            },
            "alpha_if_deleted": {
                "q1": 0.82,
                "q2": 0.84,
            }
        }
        
        report = generate_report(result, scale_name="测试量表")
        
        assert "信度分析报告 - 测试量表" in report
        assert "样本量: 100" in report
        assert "题项数: 5" in report
        assert "Cronbach's α: 0.8500" in report
        assert "良好 (Good)" in report
        assert "q1" in report
        assert "q2" in report
        assert "删除后α" in report
        assert "反向计分" in report
    
    def test_generate_report_different_alpha_levels(self):
        for alpha, expected_interp in [
            (0.95, "优秀 (Excellent)"),
            (0.85, "良好 (Good)"),
            (0.75, "可接受 (Acceptable)"),
            (0.65, "有疑问 (Questionable)"),
            (0.55, "较差 (Poor)"),
        ]:
            result = {
                "alpha": alpha,
                "item_count": 3,
                "sample_size": 50,
                "item_stats": {},
                "alpha_if_deleted": {}
            }
            report = generate_report(result)
            assert expected_interp in report
