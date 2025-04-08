"""Trainerのテスト"""

import os

import numpy as np
import pytest

from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer


@pytest.fixture
def trainer(sample_model):
    """テスト用のTrainerインスタンス"""
    evaluator = Evaluator("regression")
    return Trainer(sample_model, evaluator)


def test_trainer_init(trainer, sample_model):
    """Trainerの初期化テスト"""
    assert trainer.model is sample_model
    assert trainer.evaluator is not None
    # 以前のtraining_configに関する検証は削除


def test_trainer_train(trainer, sample_data):
    """trainメソッドのテスト"""
    X, y = sample_data

    # モデルをトレーニング
    result = trainer.train(X, y)

    # トレーニング結果の検証
    assert result is trainer.model
    assert trainer.model.is_fitted


def test_trainer_evaluate(trainer, sample_data):
    """evaluateメソッドのテスト"""
    X, y = sample_data

    # モデルをトレーニングしてから評価
    trainer.train(X, y)
    metrics = trainer.evaluate(X, y)

    # 評価結果の検証
    assert isinstance(metrics, dict)
    assert "r2_score" in metrics  # 回帰モデルなのでR2スコアがある
    assert metrics["r2_score"] > 0.9  # サンプルデータに対して良いスコアが出るはず


def test_trainer_save_load_model(trainer, sample_data, temp_model_path):
    """save_model と load_model メソッドのテスト"""
    X, y = sample_data

    # モデルをトレーニングして保存
    trainer.train(X, y)
    trainer.save_model(temp_model_path)

    # ファイルが作成されたことを確認
    assert os.path.exists(temp_model_path)

    # 元の予測結果を取得
    original_predictions = trainer.model.predict(X)

    # 新しいモデルをロード
    trainer.load_model(temp_model_path)

    # ロードされたモデルでの予測結果を取得
    loaded_predictions = trainer.model.predict(X)

    # 予測結果が一致することを確認
    assert np.array_equal(original_predictions, loaded_predictions)


def test_trainer_get_set_params(trainer):
    """get_params と set_params メソッドのテスト"""
    # パラメータを取得
    params = trainer.get_params()
    assert isinstance(params, dict)

    # 新しいパラメータを設定
    result = trainer.set_params(test_param=False)

    # メソッドチェーン用に自身を返すことを確認
    assert result is trainer

    # パラメータが更新されたことを確認
    updated_params = trainer.get_params()
    assert updated_params.get("test_param") == False
