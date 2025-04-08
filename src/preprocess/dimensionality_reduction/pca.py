from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA

from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.registry import register_preprocessor


@register_preprocessor("pca")
class PCA(BasePreprocessor):
    """
    主成分分析（PCA）による次元削減を行う前処理コンポーネント。

    高次元データを低次元空間に変換します。

    Args:
        config: 設定辞書。以下のパラメータをサポート:
            - n_components: 次元数または分散説明率 (デフォルト: 0.95)
            - svd_solver: 特異値分解のソルバー ('auto', 'full', 'arpack', 'randomized') (デフォルト: 'auto')
            - columns: 処理対象の列のリスト (デフォルト: None - 全ての数値列)
            - return_original: 元の特徴量も保持するかどうか (デフォルト: False)
            - prefix: 新しい特徴量の接頭辞 (デフォルト: 'PC_')
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        if config is None:
            config = {}

        self.n_components = config.get("n_components", 0.95)
        self.svd_solver = config.get("svd_solver", "auto")
        self.columns = config.get("columns", None)
        self.return_original = config.get("return_original", False)
        self.prefix = config.get("prefix", "PC_")

        self.pca = None
        self.feature_names: List[str] = []
        self.output_feature_names: List[str] = []
        self.original_columns: List[str] = []

    def fit(self, data: pd.DataFrame, target: Optional[Union[pd.Series, np.ndarray]] = None) -> "PCA":
        """
        PCAモデルを学習します。

        Args:
            data: 入力データフレーム
            target: 目的変数（このコンポーネントでは使用されません）

        Returns:
            self: 学習済みのPCAインスタンス
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

        # 処理対象列がなければ何もしない
        if not self.columns:
            self.fitted = True
            return self

        # 元のデータフレームの列名を保存
        self.original_columns = data.columns.tolist()

        # 特徴量名を保存
        self.feature_names = self.columns.copy()

        # PCAインスタンスの初期化と学習
        self.pca = SklearnPCA(n_components=self.n_components, svd_solver=self.svd_solver)
        self.pca.fit(data[self.columns])

        # 出力特徴量名を設定
        n_components = self.pca.n_components_
        self.output_feature_names = [f"{self.prefix}{i+1}" for i in range(n_components)]

        self.fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        データをPCAで変換します。

        Args:
            data: 変換するデータフレーム

        Returns:
            pd.DataFrame: 変換されたデータフレーム
        """
        self._validate_fitted()

        # 処理対象列がなければ元のデータを返す
        if not self.columns or not self.pca:
            return data.copy()

        # 元のデータフレームのコピーを作成
        result = data.copy()

        # 指定された列だけを抽出してPCA変換
        X = data[self.columns]
        pca_result = self.pca.transform(X)

        # 変換結果をデータフレームに変換
        pca_df = pd.DataFrame(pca_result, index=data.index, columns=self.output_feature_names)

        if self.return_original:
            # 元の特徴量を維持する場合、PCA結果を追加
            result = pd.concat([result, pca_df], axis=1)
        else:
            # 元の特徴量を破棄する場合、処理対象列を削除してからPCA結果を追加
            result = result.drop(columns=self.columns)
            result = pd.concat([result, pca_df], axis=1)

        return result

    def reset(self) -> "PCA":
        """学習状態をリセットします。"""
        self.pca = None
        self.feature_names = []
        self.output_feature_names = []
        self.original_columns = []
        self.fitted = False
        return self

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        各主成分の説明分散比を返します。

        Returns:
            Optional[np.ndarray]: 説明分散比、または未学習の場合はNone
        """
        if self.pca is None:
            return None
        return self.pca.explained_variance_ratio_

    def get_cumulative_explained_variance(self) -> Optional[np.ndarray]:
        """
        累積説明分散比を返します。

        Returns:
            Optional[np.ndarray]: 累積説明分散比、または未学習の場合はNone
        """
        if self.pca is None:
            return None
        return np.cumsum(self.pca.explained_variance_ratio_)

    def get_components(self) -> Optional[pd.DataFrame]:
        """
        主成分の係数（負荷量）をデータフレームとして返します。

        Returns:
            Optional[pd.DataFrame]: 主成分の係数、または未学習の場合はNone
        """
        if self.pca is None or not self.feature_names:
            return None

        components_df = pd.DataFrame(self.pca.components_, index=self.output_feature_names, columns=self.feature_names)
        return components_df
