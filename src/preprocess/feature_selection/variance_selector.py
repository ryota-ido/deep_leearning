from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.registry import register_preprocessor


@register_preprocessor("variance_selector")
class VarianceSelector(BasePreprocessor):
    """
    分散に基づいて特徴量を選択する前処理コンポーネント。

    一定の閾値未満の分散を持つ特徴量を除去します。

    Args:
        config: 設定辞書。以下のパラメータをサポート:
            - threshold: 除去するための分散の閾値 (デフォルト: 0.0)
            - columns: 処理対象の列のリスト (デフォルト: None - 全ての数値列)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.threshold = config.get("threshold", 0.0) if config else 0.0
        self.columns = config.get("columns", None) if config else None
        self.selected_columns: List[str] = []
        self.variances_: Dict[str, float] = {}

    def fit(self, data: pd.DataFrame, target: Optional[Union[pd.Series, np.ndarray]] = None) -> "VarianceSelector":
        """
        データの特徴量分散を計算し、閾値以上の分散を持つ特徴量を選択します。

        Args:
            data: 入力データフレーム
            target: 目的変数（このコンポーネントでは使用されません）

        Returns:
            self: 学習済みのセレクターインスタンス
        """
        # データ列の型を検出
        col_types = self._detect_column_types(data)

        # 処理対象列の確定
        if self.columns is None:
            # 数値列のみを選択
            self.columns = col_types["numeric"]
        else:
            # 指定された列のうち、データフレームに存在するものだけ使用
            self.columns = [col for col in self.columns if col in data.columns]

        if not self.columns:
            self.selected_columns = []
            self.fitted = True
            return self

        # 各列の分散を計算
        self.variances_ = {}
        for col in self.columns:
            self.variances_[col] = data[col].var()

        # 閾値以上の分散を持つ列を選択
        self.selected_columns = [col for col in self.columns if self.variances_[col] > self.threshold]

        self.fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        データから選択された特徴量だけを抽出します。

        Args:
            data: 変換するデータフレーム

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        self._validate_fitted()

        # 元のデータフレームのコピーを作成
        result = data.copy()

        # 選択されなかった列を除去
        cols_to_drop = [col for col in self.columns if col not in self.selected_columns]
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop)

        return result

    def reset(self) -> "VarianceSelector":
        """学習状態をリセットします。"""
        self.selected_columns = []
        self.variances_ = {}
        self.fitted = False
        return self

    def get_feature_variances(self) -> Dict[str, float]:
        """
        学習した各特徴量の分散を返します。

        Returns:
            Dict[str, float]: 特徴量名と分散のマッピング
        """
        self._validate_fitted()
        return self.variances_

    def get_selected_features(self) -> List[str]:
        """
        選択された特徴量のリストを返します。

        Returns:
            List[str]: 選択された特徴量名のリスト
        """
        self._validate_fitted()
        return self.selected_columns.copy()
