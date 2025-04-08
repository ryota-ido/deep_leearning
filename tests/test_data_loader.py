"""DataLoaderのテスト"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data.manager import DataManager


@pytest.fixture
def temp_csv_file():
    """一時的なCSVファイルを作成"""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
                "target": [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )
        df.to_csv(tmp.name, index=False)
        yield tmp.name
    # テスト後にファイルを削除
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def data_manager():
    """テスト用のDataManagerインスタンス"""
    config = {
        "data": {
            "raw_dir": ".",  # テスト用に現在のディレクトリを使用
            "processed_dir": ".",  # テスト用に現在のディレクトリを使用
        }
    }
    return DataManager(config)


def test_load_csv(data_manager, temp_csv_file):
    """load_dataメソッドのテスト（CSV形式）"""
    # ファイルパスから相対パスを取得
    filename = os.path.basename(temp_csv_file)
    data_manager.raw_dir = os.path.dirname(temp_csv_file)

    # CSVファイルをロード
    df = data_manager.load_data("", temp_csv_file, "csv")

    # 読み込まれたデータの検証
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 3)
    assert list(df.columns) == ["feature1", "feature2", "target"]
    assert df["feature1"].iloc[0] == 1.0
    assert df["target"].iloc[-1] == 5.5


def test_split_data(data_manager, sample_dataframe):
    """split_dataメソッドのテスト"""
    # データを分割
    X_train, X_test, y_train, y_test = data_manager.split_data(
        sample_dataframe, target_col="target", test_size=0.5, random_state=42
    )

    # 分割されたデータの検証
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    # サイズの検証
    assert X_train.shape[0] == 2  # 50%に分割
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 2
    assert y_test.shape[0] == 2

    # Xから目的変数が除外されていることを確認
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns


def test_save_and_load_processed(data_manager, sample_dataframe):
    """save_dataとload_dataメソッドのテスト"""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        # 処理済みファイルの保存先を設定
        processed_dir = os.path.dirname(tmp.name)
        filename = os.path.basename(tmp.name)

        # データを保存
        data_manager.save_data(sample_dataframe, filename, processed_dir, format="csv")

        # 保存されたデータを読み込み
        loaded_df = data_manager.load_data(processed_dir, filename, "csv")

        # 元のデータと一致するか検証
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

    # テスト後にファイルを削除
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)
