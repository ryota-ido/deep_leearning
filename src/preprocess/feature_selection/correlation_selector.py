from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.registry import register_preprocessor


@register_preprocessor("correlation_selector")
class CorrelationSelector(BasePreprocessor):
    """
    特徴量間の相関に基づいて特徴量を選択する前処理コンポーネント。

    指定された閾値以上の相関を持つ特徴量ペアから一方を除外します。

    Args:
        config: 設定辞書。以下のパラメータをサポート:
            - threshold: 除去するための相関の閾値 (デフォルト: 0.8)
            - columns: 処理対象の列のリスト (デフォルト: None - 全ての数値列)
            - method: 相関係数の種類 ('pearson', 'kendall', 'spearman') (デフォルト: 'pearson')
            - selection_method: どの特徴量を残すかの方法 ('first', 'variance') (デフォルト: 'variance')
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        if config is None:
            config = {}
        self.threshold = config.get("threshold", 0.8)
        self.columns = config.get("columns", None)
        self.method = config.get("method", "pearson")
        self.selection_method = config.get("selection_method", "variance")
        self.selected_columns: List[str] = []
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.feature_variances_: Dict[str, float] = {}

    def fit(self, data: pd.DataFrame, target: Optional[Union[pd.Series, np.ndarray]] = None) -> "CorrelationSelector":
        """
        データの特徴量間相関を計算し、閾値以上の相関を持つペアを特定して
        選択方法に基づいて一方を除外します。

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

        if len(self.columns) <= 1:
            self.selected_columns = self.columns.copy()
            self.fitted = True
            return self

        # 相関行列の計算
        self.correlation_matrix_ = data[self.columns].corr(method=self.method)

        # 各特徴量の分散を計算（selection_method='variance'の場合に使用）
        if self.selection_method == "variance":
            self.feature_variances_ = {col: data[col].var() for col in self.columns}

        # 相関の高いペアを見つけて一方を除外
        columns_to_drop = set()

        # 相関行列の上三角行列のみを処理（対角線を含まない）
        for i in range(len(self.columns)):
            for j in range(i + 1, len(self.columns)):
                col_i = self.columns[i]
                col_j = self.columns[j]

                # すでに除外リストにある列はスキップ
                if col_i in columns_to_drop or col_j in columns_to_drop:
                    continue

                # 相関係数のチェック
                corr = abs(self.correlation_matrix_.loc[col_i, col_j])
                if corr > self.threshold:
                    # どちらの特徴量を除外するかを決定
                    if self.selection_method == "first":
                        # 常に2番目の特徴量を除外
                        columns_to_drop.add(col_j)
                    elif self.selection_method == "variance":
                        # 分散が小さい方を除外
                        if self.feature_variances_[col_i] < self.feature_variances_[col_j]:
                            columns_to_drop.add(col_i)
                        else:
                            columns_to_drop.add(col_j)

        # 選択された列を設定
        self.selected_columns = [col for col in self.columns if col not in columns_to_drop]

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

        # 元のデータにない列を除外して選択された列のリストを作成
        valid_cols = [col for col in self.selected_columns if col in data.columns]

        # 選択されなかった列を除去（元のデータに存在する列のみ）
        cols_to_drop = [col for col in self.columns if col in data.columns and col not in valid_cols]
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop)

        return result

    def reset(self) -> "CorrelationSelector":
        """学習状態をリセットします。"""
        self.selected_columns = []
        self.correlation_matrix_ = None
        self.feature_variances_ = {}
        self.fitted = False
        return self

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """
        学習した相関行列を返します。

        Returns:
            Optional[pd.DataFrame]: 相関行列、または未学習の場合はNone
        """
        if not self.fitted:
            return None
        return self.correlation_matrix_.copy() if self.correlation_matrix_ is not None else None

    def get_correlation_pairs(self, min_threshold: Optional[float] = None) -> List[Tuple[str, str, float]]:
        """
        閾値以上の相関を持つ特徴量ペアを返します。

        Args:
            min_threshold: 表示する最小相関係数（指定しない場合はself.thresholdを使用）

        Returns:
            List[Tuple[str, str, float]]: 相関の高いペアと相関係数のリスト
        """
        self._validate_fitted()

        if min_threshold is None:
            min_threshold = self.threshold

        if self.correlation_matrix_ is None:
            return []

        high_corr_pairs = []
        for i in range(len(self.columns)):
            for j in range(i + 1, len(self.columns)):
                col_i = self.columns[i]
                col_j = self.columns[j]
                corr = abs(self.correlation_matrix_.loc[col_i, col_j])
                if corr > min_threshold:
                    high_corr_pairs.append((col_i, col_j, corr))

        # 相関係数の降順でソート
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

        return high_corr_pairs
