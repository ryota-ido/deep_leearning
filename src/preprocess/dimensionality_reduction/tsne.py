from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE as SklearnTSNE

from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.registry import register_preprocessor


@register_preprocessor("tsne")
class TSNE(BasePreprocessor):
    """
    t-SNE（t-distributed Stochastic Neighbor Embedding）による次元削減を行う前処理コンポーネント。

    高次元データを低次元空間に非線形変換します。主に可視化に適しています。

    Args:
        config: 設定辞書。以下のパラメータをサポート:
            - n_components: 出力次元数 (デフォルト: 2)
            - perplexity: パープレキシティ (デフォルト: 30.0)
            - learning_rate: 学習率 (デフォルト: 200.0)
            - n_iter: 最大反復回数 (デフォルト: 1000)
            - random_state: 乱数シード (デフォルト: 42)
            - columns: 処理対象の列のリスト (デフォルト: None - 全ての数値列)
            - return_original: 元の特徴量も保持するかどうか (デフォルト: False)
            - prefix: 新しい特徴量の接頭辞 (デフォルト: 'TSNE_')
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        if config is None:
            config = {}

        self.n_components = config.get("n_components", 2)
        self.perplexity = config.get("perplexity", 30.0)
        self.learning_rate = config.get("learning_rate", 200.0)
        self.n_iter = config.get("n_iter", 1000)
        self.random_state = config.get("random_state", 42)
        self.columns = config.get("columns", None)
        self.return_original = config.get("return_original", False)
        self.prefix = config.get("prefix", "TSNE_")

        self.tsne = None
        self.feature_names: List[str] = []
        self.output_feature_names: List[str] = []
        self.original_columns: List[str] = []
        self.tsne_results_: Optional[np.ndarray] = None

    def fit(self, data: pd.DataFrame, target: Optional[Union[pd.Series, np.ndarray]] = None) -> "TSNE":
        """
        t-SNEモデルを学習します。fit_transformを内部で呼び出します。

        Args:
            data: 入力データフレーム
            target: 目的変数（このコンポーネントでは使用されません）

        Returns:
            self: 学習済みのt-SNEインスタンス
        """
        # t-SNEでは fit_transform しか使わないため、内部で fit_transform を呼び出す
        _ = self.fit_transform(data, target)
        return self

    def fit_transform(self, data: pd.DataFrame, target: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
        """
        t-SNEモデルを学習し、同時にデータを変換します。

        Args:
            data: 入力データフレーム
            target: 目的変数（このコンポーネントでは使用されません）

        Returns:
            pd.DataFrame: 変換されたデータフレーム
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

        # 処理対象列がなければ元のデータを返す
        if not self.columns:
            self.fitted = True
            return data.copy()

        # 元のデータフレームの列名を保存
        self.original_columns = data.columns.tolist()

        # 特徴量名を保存
        self.feature_names = self.columns.copy()

        # t-SNEインスタンスの初期化と学習
        self.tsne = SklearnTSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        # t-SNEを適用
        X = data[self.columns]
        self.tsne_results_ = self.tsne.fit_transform(X)

        # 出力特徴量名を設定
        self.output_feature_names = [f"{self.prefix}{i+1}" for i in range(self.n_components)]

        # 変換結果をデータフレームに変換
        tsne_df = pd.DataFrame(self.tsne_results_, index=data.index, columns=self.output_feature_names)

        # 結果のデータフレームを作成
        result = data.copy()

        if self.return_original:
            # 元の特徴量を維持する場合、t-SNE結果を追加
            result = pd.concat([result, tsne_df], axis=1)
        else:
            # 元の特徴量を破棄する場合、処理対象列を削除してからt-SNE結果を追加
            result = result.drop(columns=self.columns)
            result = pd.concat([result, tsne_df], axis=1)

        self.fitted = True
        return result

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        データをt-SNEで変換します。

        注意: t-SNEはtransformメソッドをサポートしていないため、新しいデータに対しては
        新たにfit_transformを実行する必要があります。このメソッドは便宜上提供されますが、
        学習済みのt-SNEモデルで新しいデータを変換することはできません。

        Args:
            data: 変換するデータフレーム

        Returns:
            pd.DataFrame: 変換されたデータフレーム
        """
        self._validate_fitted()

        # t-SNEはtransformメソッドをサポートしていないため、警告を出力
        import warnings

        warnings.warn(
            "t-SNE does not support transform on new data. "
            "The original data will be returned with placeholder columns. "
            "For proper t-SNE transformation, use fit_transform on all data at once.",
            UserWarning,
        )

        # 元のデータフレームのコピーを作成
        result = data.copy()

        # 処理対象列がなければ元のデータを返す
        if not self.columns:
            return result

        if self.return_original:
            # 元の特徴量を維持する場合、NaNで埋められたt-SNE結果列を追加
            for col in self.output_feature_names:
                result[col] = np.nan
        else:
            # 元の特徴量を破棄する場合、処理対象列を削除してからNaNで埋められたt-SNE結果列を追加
            result = result.drop(columns=[c for c in self.columns if c in result.columns])
            for col in self.output_feature_names:
                result[col] = np.nan

        return result

    def reset(self) -> "TSNE":
        """学習状態をリセットします。"""
        self.tsne = None
        self.feature_names = []
        self.output_feature_names = []
        self.original_columns = []
        self.tsne_results_ = None
        self.fitted = False
        return self

    def get_tsne_results(self) -> Optional[np.ndarray]:
        """
        学習済みのt-SNE結果を返します。

        Returns:
            Optional[np.ndarray]: t-SNE変換結果、または未学習の場合はNone
        """
        if not self.fitted or self.tsne_results_ is None:
            return None
        return self.tsne_results_.copy()
