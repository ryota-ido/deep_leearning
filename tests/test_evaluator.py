"""Evaluatorのテスト"""

import numpy as np
import pytest

from src.evaluation.evaluator import Evaluator


@pytest.fixture
def regression_evaluator():
    """回帰モデル用のEvaluatorインスタンス"""
    return Evaluator("regression")


@pytest.fixture
def classification_evaluator():
    """分類モデル用のEvaluatorインスタンス"""
    return Evaluator("classification")


def test_evaluator_init(regression_evaluator: Evaluator, classification_evaluator: Evaluator):
    """Evaluatorの初期化テスト"""
    # 回帰評価器が適切なメトリクスを持っているか確認
    assert "mean_squared_error" in regression_evaluator.metrics
    assert "mean_absolute_error" in regression_evaluator.metrics
    assert "r2_score" in regression_evaluator.metrics

    # 分類評価器が適切なメトリクスを持っているか確認
    assert "accuracy_score" in classification_evaluator.metrics
    assert "precision_score" in classification_evaluator.metrics
    assert "recall_score" in classification_evaluator.metrics
    assert "f1_score" in classification_evaluator.metrics


def test_evaluate_regression(regression_evaluator: Evaluator, sample_model, sample_data):
    """回帰モデルの評価テスト"""
    X, y = sample_data

    # モデルをトレーニング
    sample_model.fit(X, y)

    # モデルを評価
    metrics = regression_evaluator.evaluate(sample_model, X, y)

    # 評価結果の検証
    assert isinstance(metrics, dict)
    assert "mean_squared_error" in metrics
    assert "mean_absolute_error" in metrics
    assert "r2_score" in metrics

    # サンプルモデルはサンプルデータに完全に適合するので、誤差は小さいはず
    assert metrics["mean_squared_error"] < 1e-10
    assert metrics["mean_absolute_error"] < 1e-5
    assert metrics["r2_score"] > 0.999


def test_resolve_metric_function(regression_evaluator: Evaluator):
    """メトリクス関数の解決テスト"""
    # 文字列指定からの関数解決をテスト
    func = regression_evaluator._resolve_metric_function("sklearn.metrics.mean_absolute_error")
    assert func is not None
    assert callable(func)

    # 存在しない関数名の場合
    func = regression_evaluator._resolve_metric_function("non_existent_metric")
    assert func is None
