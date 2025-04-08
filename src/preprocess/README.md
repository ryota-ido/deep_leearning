# 前処理モジュール (src/preprocess/)

## 概要

このモジュールは、機械学習のための様々なデータ前処理機能を提供します。データクリーニング、特徴量変換、スケーリングなど、モデルトレーニングの前に必要な処理を行うためのコンポーネントが含まれています。すべての前処理クラスは、共通のインターフェースを持ち、パイプラインとして連結できます。

## ディレクトリ構造

```
preprocess/
├── base_preprocessor.py    # 前処理の基底クラス
├── preprocess_pipeline.py  # 前処理パイプライン
├── basic/                  # 基本的な前処理コンポーネント
│   ├── missing_value_handler.py   # 欠損値処理
│   ├── outlier_remover.py         # 外れ値処理
│   ├── scaler.py                  # スケーリング
│   └── label_encoder.py           # ラベルエンコーディング
└── README.md               # このファイル
```

## 主要コンポーネント

### BasePreprocessor

すべての前処理クラスの基底となる抽象クラス。以下のメソッドを定義しています：

- `fit(data, target=None)`: 前処理パラメータをデータから学習する
- `transform(data)`: 学習したパラメータを使用してデータを変換する
- `fit_transform(data, target=None)`: fitとtransformを連続して実行する
- `preprocess(X_train, X_test, y_train)`: トレーニングとテストデータを同時に処理する
- `reset()`: 学習した状態をリセットする
- `_validate_fitted()`: 前処理が学習済みかどうかを検証する
- `_detect_column_types(data)`: データの列型を自動検出する

### PreprocessingPipeline

複数の前処理ステップを順番に適用するパイプライン機能を提供します。

- 設定ファイルから前処理ステップを読み込む
- 各前処理ステップを登録されたプリプロセッサから動的に作成
- トレーニングとテストデータに同じ前処理手順を適用

### 基本前処理コンポーネント

#### MissingValueHandler

欠損値を様々な戦略で処理します：
- 数値データ: `mean`, `median`, `constant`, `drop`
- カテゴリカルデータ: `most_frequent`, `constant`, `drop`
- カラム別の異なる戦略をサポート

```python
from src.preprocess.basic.missing_value_handler import MissingValueHandler

handler = MissingValueHandler({
    "strategy": "mean",
    "categorical_strategy": "most_frequent",
    "column_strategies": {"age": "median", "country": "constant"}
})
```

#### OutlierRemover

異常値を検出して処理します：
- 検出方法: `iqr` (四分位範囲法), `zscore` (Z-スコア法), `threshold` (閾値法)
- 処理方法: `clip` (閾値でクリッピング), `remove` (行の除去), `null` (NULLに置換)
- カラム別の設定が可能

```python
from src.preprocess.basic.outlier_remover import OutlierRemover

remover = OutlierRemover({
    "method": "iqr",
    "threshold": 1.5,
    "treatment": "clip"
})
```

#### Scaler

数値データのスケーリングを行います。3種類のスケーラーが実装されています：
- `StandardScaler`: 平均0、分散1にスケーリング
- `MinMaxScaler`: 指定された範囲（デフォルトは0〜1）にスケーリング
- `RobustScaler`: 外れ値に強いスケーリング（中央値と四分位範囲を使用）

```python
from src.preprocess.basic.scaler import StandardScaler, MinMaxScaler, RobustScaler

# 標準化スケーラー
std_scaler = StandardScaler({"with_mean": True, "with_std": True})

# Min-Maxスケーラー
minmax_scaler = MinMaxScaler({"feature_range": (0, 1)})

# ロバストスケーラー
robust_scaler = RobustScaler({"with_centering": True, "with_scaling": True})
```

#### LabelEncoder

カテゴリカル変数を整数ラベルに変換します：
- 文字列や他のカテゴリカルデータを0からの連番に変換
- 未知のカテゴリを処理するオプション

```python
from src.preprocess.basic.label_encoder import LabelEncoder

encoder = LabelEncoder({
    "handle_unknown": "use_default",
    "default_value": -1
})
```

## 使用例

### 単一の前処理コンポーネントの使用

```python
from src.preprocess.basic.missing_value_handler import MissingValueHandler
import pandas as pd

# サンプルデータ
data = pd.DataFrame({
    'num1': [1.0, 2.0, None, 4.0],
    'num2': [5.0, None, 7.0, 8.0],
    'cat1': ['A', 'B', None, 'B']
})

# 前処理クラスのインスタンス化
handler = MissingValueHandler({"strategy": "mean", "categorical_strategy": "most_frequent"})

# トレーニングデータでパラメータを学習
handler.fit(data)

# 学習したパラメータでデータを変換
transformed_data = handler.transform(data)

print(transformed_data)
# num1  num2 cat1
# 0   1.0   5.0    A
# 1   2.0   6.5    B
# 2   2.3   7.0    B
# 3   4.0   8.0    B
```

### 前処理パイプラインの使用

複数の前処理ステップを組み合わせて使用する場合：

```python
from src.preprocess.preprocess_pipeline import PreprocessingPipeline
import pandas as pd
import numpy as np

# サンプルデータ
X_train = pd.DataFrame({
    'num1': [1.0, 2.0, None, 4.0, 100.0],  # 外れ値あり
    'num2': [5.0, None, 7.0, 8.0, 9.0],
    'cat1': ['A', 'B', None, 'B', 'C']
})

X_test = pd.DataFrame({
    'num1': [3.0, None, 5.0],
    'num2': [None, 6.0, 10.0],
    'cat1': ['A', 'D', None]  # 未知のカテゴリ
})

y_train = np.array([0, 1, 0, 1, 1])

# 前処理パイプラインの設定
preprocessing_config = {
    "steps": [
        {
            "name": "missing_values",
            "type": "missing_value_handler",
            "params": {
                "strategy": "median",
                "categorical_strategy": "most_frequent"
            }
        },
        {
            "name": "outliers",
            "type": "outlier_remover",
            "params": {
                "method": "iqr",
                "threshold": 1.5,
                "treatment": "clip"
            }
        },
        {
            "name": "scaling",
            "type": "standard_scaler",
            "params": {
                "with_mean": True,
                "with_std": True
            }
        },
        {
            "name": "encoding",
            "type": "label_encoder",
            "params": {
                "handle_unknown": "use_default"
            }
        }
    ]
}

# パイプラインの作成
pipeline = PreprocessingPipeline(preprocessing_config)

# パイプラインの実行
X_train_processed, X_test_processed = pipeline.run(X_train, y_train, X_test)

print("処理済みトレーニングデータ:")
print(X_train_processed)

print("\n処理済みテストデータ:")
print(X_test_processed)
```

### MLflowパイプラインとの統合

```python
from src.pipeline.mlflow.components.preprocess import run_preprocess
from src.utils.config_manager import ConfigManager

# 設定マネージャーの作成
config_manager = ConfigManager.from_yaml_files("config/pipeline_config.yaml", "config/experiments/linear_regression_california_housing.yaml")

# 前処理の実行
preprocess_result = run_preprocess(config_manager)
```

## 設定ファイル例

YAML設定ファイルでの前処理設定例：

```yaml
preprocessing:
  steps:
    - name: missing_values
      type: missing_value_handler
      params:
        strategy: mean
        categorical_strategy: most_frequent

    - name: outlier_removal
      type: outlier_remover
      params:
        method: iqr
        threshold: 1.5
        treatment: clip

    - name: scaling
      type: standard_scaler
      params:
        with_mean: true
        with_std: true

    - name: encoding
      type: label_encoder
      params:
        handle_unknown: "use_default"
        default_value: -1
```

## 新しい前処理コンポーネントの追加

1. `src/preprocess/basic/` (または新しいサブディレクトリ) に新しいモジュールを作成
2. `BasePreprocessor`を継承したクラスを実装
3. `@register_preprocessor`デコレータを使用
4. 必要なメソッドを実装

```python
from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.registry import register_preprocessor
import pandas as pd
import numpy as np

@register_preprocessor("polynomial_features")
class PolynomialFeatures(BasePreprocessor):
    """多項式特徴量を生成する前処理コンポーネント"""

    def __init__(self, config=None):
        super().__init__(config)
        self.degree = config.get("degree", 2) if config else 2
        self.interaction_only = config.get("interaction_only", False) if config else False
        self.columns = config.get("columns", None) if config else None
        self.feature_names = None

    def fit(self, data, target=None):
        """多項式特徴量の設定を学習"""
        # 対象カラムの選択
        if self.columns is None:
            # 数値カラムのみを選択
            self.columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        else:
            # 指定されたカラムのうち、データに存在するものだけを使用
            self.columns = [col for col in self.columns if col in data.columns]

        # 特徴量名の生成
        self._generate_feature_names()

        self.fitted = True
        return self

    def transform(self, data):
        """多項式特徴量の生成"""
        self._validate_fitted()

        # 入力データのコピーを作成
        result = data.copy()

        # 対象カラムが存在しない場合は元のデータを返す
        if not self.columns:
            return result

        # 多項式特徴量の生成
        X = data[self.columns].values
        poly_features = self._compute_polynomial_features(X)

        # 元のデータフレームに新しい特徴量を追加
        for i, name in enumerate(self.feature_names):
            if name not in self.columns:  # 元の特徴量は追加しない
                result[name] = poly_features[:, i]

        return result

    def reset(self):
        """学習状態のリセット"""
        self.feature_names = None
        self.fitted = False
        return self

    def _generate_feature_names(self):
        """多項式特徴量の名前を生成"""
        if not self.columns:
            self.feature_names = []
            return

        from itertools import combinations_with_replacement

        self.feature_names = []
        for d in range(1, self.degree + 1):
            if self.interaction_only and d > 1:
                combos = combinations(self.columns, d)
            else:
                combos = combinations_with_replacement(self.columns, d)

            for combo in combos:
                name = '*'.join(combo)
                self.feature_names.append(name)

    def _compute_polynomial_features(self, X):
        """多項式特徴量を計算"""
        from itertools import combinations_with_replacement

        n_samples, n_features = X.shape

        # 次数1の特徴量（元の特徴量）
        poly_features = X.copy()

        # 次数2以上の特徴量
        for d in range(2, self.degree + 1):
            if self.interaction_only:
                combos = list(combinations(range(n_features), d))
            else:
                combos = list(combinations_with_replacement(range(n_features), d))

            for combo in combos:
                new_feature = np.ones(n_samples)
                for i in combo:
                    new_feature *= X[:, i]

                poly_features = np.column_stack((poly_features, new_feature))

        return poly_features
```

## 特徴量エンジニアリングのベストプラクティス

1. **データの理解**:
   - 前処理前にデータの分布や特性を理解する
   - 欠損値、外れ値のパターンを分析

2. **前処理の順序**:
   - 一般的に以下の順序が効果的
     1. 欠損値処理
     2. 外れ値処理
     3. 特徴量変換（対数変換など）
     4. スケーリング
     5. エンコーディング
   - モデルによっては順序が重要な場合がある

3. **データリーケージの防止**:
   - 前処理のパラメータはトレーニングデータからのみ学習
   - テストデータには学習したパラメータを適用
   - 交差検証でも同様に注意

4. **モデル特性の考慮**:
   - 線形モデルはスケーリングが重要
   - 決定木ベースのモデルはスケーリングに比較的影響されない
   - カテゴリ変数の扱い方はモデルによって異なる

5. **ドメイン知識の活用**:
   - 業界特有の知識を活かした特徴量エンジニアリング
   - 特徴量の組み合わせや変換で予測力を高める

## 注意点

- **前処理は再現性を保つ**: `fit()`で学習したパラメータを使って`transform()`を実行し、同じ変換が再現できるようにする
- **パイプラインの検証**: データの形状や型が前処理ステップ間で一貫していることを確認する
- **メモリ効率**: 大きなデータセットでは不要なデータのコピーを避ける
- **カラム名の一貫性**: 前処理後もカラム名を維持し、トレーサビリティを確保する
- **前処理のロギング**: 各ステップの効果を監視し、問題を早期に発見する
- **パラメータの保存**: 前処理パラメータも保存して本番環境で再利用できるようにする
- **特徴量の解釈可能性**: 前処理後も特徴量の意味が理解できるようにする
- **すべてのケースを考慮**: エッジケース（極端な値、稀なカテゴリなど）を適切に処理する
