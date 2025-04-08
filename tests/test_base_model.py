"""BaseModelのテスト"""

import os

import numpy as np
import pytest

from src.models.base_model import BaseModel


def test_base_model_init(sample_model):
    """BaseModelの初期化テスト"""
    # 初期化後の状態を確認
    assert sample_model.model_category == "regression"
    assert sample_model.is_fitted == False
    assert sample_model.params == {"test_param": True}
    assert sample_model._model is not None


def test_model_fit(sample_model, sample_data):
    """fitメソッドのテスト"""
    X, y = sample_data
    result = sample_model.fit(X, y)

    # fitメソッドは自身を返すことを確認
    assert result is sample_model
    # fitted状態になっていることを確認
    assert sample_model.is_fitted == True


def test_model_predict(sample_model, sample_data):
    """predictメソッドのテスト"""
    X, y = sample_data

    # fitしてから予測
    sample_model.fit(X, y)
    predictions = sample_model.predict(X)

    # 予測結果の形状と値を確認
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == y.shape
    # SimpleTestModelは既知の重みで単純な線形予測を行うので
    # 予測値はサンプルデータのyと一致するはず
    assert np.allclose(predictions, y)


def test_predict_without_fit(sample_model, sample_data):
    """fitせずにpredictを呼び出した場合のテスト"""
    X, y = sample_data

    # 初期化直後のモデルはfitされていないため例外が発生するはず
    sample_model.reset()
    with pytest.raises(ValueError, match="not fitted yet"):
        sample_model.predict(X)


def test_model_save_load(sample_model, sample_data, temp_model_path):
    """モデルの保存と読み込みテスト"""
    X, y = sample_data

    # モデルをfitして保存
    sample_model.fit(X, y)
    sample_model.save(temp_model_path)

    # ファイルが作成されたことを確認
    assert os.path.exists(temp_model_path)

    # 別のインスタンスとして読み込み
    loaded_model = sample_model.__class__.load(temp_model_path)

    # 読み込んだモデルの状態を確認
    assert loaded_model.is_fitted == True
    assert loaded_model.model_category == sample_model.model_category

    # 読み込んだモデルでの予測が元のモデルと同じ結果になることを確認
    original_predictions = sample_model.predict(X)
    loaded_predictions = loaded_model.predict(X)
    assert np.array_equal(original_predictions, loaded_predictions)


def test_model_save_without_fit(sample_model, temp_model_path):
    """fitしていないモデルを保存したときのテスト"""
    # 初期化直後のモデルはfitされていないため保存時に例外が発生するはず
    sample_model.reset()
    with pytest.raises(ValueError, match="must be fitted before saving"):
        sample_model.save(temp_model_path)


def test_reset(sample_model, sample_data):
    """resetメソッドのテスト"""
    X, y = sample_data

    # モデルをfitしてからリセット
    sample_model.fit(X, y)
    assert sample_model.is_fitted == True

    sample_model.reset()
    # リセット後はfitted状態がFalseに戻ること
    assert sample_model.is_fitted == False
