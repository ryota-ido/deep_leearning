"""テストのための共通フィクスチャ"""

import os
import sys
import tempfile

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.models.base_model import BaseModel


class SimpleTestModel(BaseModel):
    """テスト用の単純なモデル実装"""

    def __init__(self, config=None):
        super().__init__(config)
        self.model_category = "regression"

    def init_model(self):
        """モデルの初期化"""
        self._model = {"weights": np.array([1.0, 2.0, 3.0]), "bias": 0.5}
        return self._model

    def fit(self, X, y, **kwargs):
        """モデルの学習（実際には何もしない）"""
        self.is_fitted = True
        return self

    def predict(self, X):
        """線形予測を行う"""
        self._validate_fitted()
        X_array = X if isinstance(X, np.ndarray) else X.values
        return X_array @ self._model["weights"] + self._model["bias"]

    def get_params(self):
        """モデルパラメータを取得"""
        return self.params

    def set_params(self, **params):
        """モデルパラメータを設定"""
        self.params.update(params)
        return self


@pytest.fixture
def sample_model():
    """テスト用のモデルインスタンス"""
    return SimpleTestModel({"params": {"test_param": True}})


@pytest.fixture
def sample_data():
    """テスト用のサンプルデータ"""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    y = np.array([14.5, 32.5, 50.5, 68.5])  # X @ [1, 2, 3] + 0.5
    return X, y


@pytest.fixture
def sample_dataframe():
    """テスト用のサンプルDataFrame"""
    df = pd.DataFrame(
        {
            "feature1": [1.0, 4.0, 7.0, 10.0],
            "feature2": [2.0, 5.0, 8.0, 11.0],
            "feature3": [3.0, 6.0, 9.0, 12.0],
            "target": [14.5, 32.5, 50.5, 68.5],
        }
    )
    return df


@pytest.fixture
def temp_model_path():
    """モデル保存用の一時ファイルパス"""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        yield tmp.name
    # テスト後にファイルを削除
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)
