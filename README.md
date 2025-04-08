# 機械学習フレームワーク

## 概要

このプロジェクトは、柔軟かつ拡張可能な機械学習ワークフローを提供する包括的なフレームワークです。データの前処理から様々なモデルの学習、評価、チューニングまでをシームレスに実行でき、MLflowによる実験管理が統合されています。このフレームワークは、モジュラー設計によって新しいコンポーネントの追加が容易で、再現性の高い機械学習実験をサポートします。

## 主な特徴

- **モジュラーアーキテクチャ**
  - データ処理、モデル、トレーニング、評価の各コンポーネントが独立
  - 簡単に新しいモデルや前処理手法を追加可能
  - レジストリパターンによる動的なクラス管理

- **柔軟な設定管理**
  - YAML設定ファイルによる実験設定
  - パイプライン設定と実験設定の分離
  - コンポーネント単位での有効/無効切り替え

- **MLflowによる実験管理**
  - 実験パラメータ、メトリクス、モデルの自動記録
  - 実験の比較・追跡が容易
  - モデルのバージョン管理とデプロイのサポート
  - カスタムビジュアライゼーションの生成と記録

- **包括的なデータ処理**
  - 様々なデータソースからの読み込み（CSV, Excel, Pickle）
  - 欠損値処理、外れ値検出、スケーリングなどの前処理
  - ラベルエンコーディングなどのカテゴリ変数処理
  - カスタム前処理コンポーネントの追加が容易

- **高度なモデリング機能**
  - 様々なモデルタイプ（回帰、分類）のサポート
  - 交差検証（K-Fold, 層化K-Fold, 時系列分割）
  - ハイパーパラメータチューニング（グリッドサーチ、Optuna）
  - モデルの保存と読み込み

詳細なワークフロー図は[こちら](docs/diagrams/ml_workflow.md)で確認できます。

## インストール

### 必要条件

- Python 3.8+
- 依存ライブラリ（`requirements.txt`参照）
  - NumPy>=1.19.0
  - Pandas>=1.1.0
  - Scikit-learn>=0.24.0
  - PyYAML>=5.4.0
  - MLflow>=1.12.0
  - Optuna>=2.3.0
  - その他

### 通常インストール

1. リポジトリをクローン
```bash
git clone https://github.com/your-username/ml-framework.git
cd ml-framework
```

2. 仮想環境の作成と依存関係のインストール
```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
pip install -r requirements.txt
```

### 開発者向けインストール

開発に参加する場合は、パッケージを開発モードでインストールすることを推奨します：

```bash
# 仮想環境をアクティベートした状態で
pip install -e .

# pre-commitフックのインストール
pip install pre-commit
pre-commit install
```

## プロジェクト構造

```
ml-framework/
├── config/                  # 設定ファイル
│   ├── experiments/        # 実験固有の設定
│   │   ├── linear_regression_california_housing.yaml
│   │   └── logistic_regression_abalone.yaml
│   ├── pipeline_config.yaml # パイプライン全体の設定
│   └── README.md           # 設定ファイルの説明
├── data/                    # データディレクトリ
│   ├── raw/                 # 生データ
│   └── processed/           # 前処理済みデータ
├── docs/                    # ドキュメント
│   └── diagrams/            # 図表
│       └── ml_workflow.md   # ワークフロー図
├── logs/                    # ログファイル
├── models/                  # 保存済みモデル
│   └── saved/               # 学習済みモデル
├── notebooks/               # Jupyter notebooks
│   ├── create_dataset.py    # データセット作成スクリプト
│   └── framework_tutorial.ipynb # チュートリアル
├── src/                     # ソースコード
│   ├── data/                # データ管理モジュール
│   │   ├── loader.py       # データ読み込み
│   │   ├── manager.py      # データ管理
│   │   └── storage.py      # ストレージ抽象化
│   ├── evaluation/          # 評価モジュール
│   │   └── evaluator.py    # モデル評価
│   ├── models/              # モデル実装
│   │   ├── base_model.py   # モデル基底クラス
│   │   └── supervised/     # 教師あり学習モデル
│   │       ├── classification/ # 分類モデル
│   │       └── regression/     # 回帰モデル
│   ├── pipeline/            # パイプラインモジュール
│   │   └── mlflow/          # MLflowパイプライン
│   │       ├── components/  # パイプラインコンポーネント
│   │       ├── core.py      # コア実装
│   │       └── mlflow_pipeline.py # エントリーポイント
│   ├── preprocess/          # 前処理モジュール
│   │   ├── base_preprocessor.py # 前処理基底クラス
│   │   ├── preprocess_pipeline.py # 前処理パイプライン
│   │   └── basic/           # 基本前処理コンポーネント
│   ├── training/            # トレーニングモジュール
│   │   ├── trainer.py       # モデルトレーナー
│   │   ├── cross_validation/ # 交差検証
│   │   └── tuners/          # ハイパーパラメータチューニング
│   └── utils/               # ユーティリティ
│       ├── config_manager.py # 設定管理
│       ├── logger.py        # ロギング
│       └── registry.py      # クラスレジストリ
├── tests/                   # テスト
│   ├── conftest.py          # テスト共通設定
│   └── test_*.py            # 各種テスト
├── main.py                  # メインスクリプト
├── requirements.txt         # 依存ライブラリ
├── pyproject.toml           # プロジェクト設定
├── CONTRIBUTING.md          # コントリビューションガイド
└── README.md                # このファイル
```

## 使用方法

### MLflowパイプラインの実行

MLflowパイプラインを使用すると、実験の構成から実行、追跡までを統一的に管理できます。

```bash
# 設定ファイルを指定してMLflowパイプラインを実行
python src/pipeline/mlflow/mlflow_pipeline.py --config config/pipeline_config.yaml --experiment-config config/experiments/linear_regression_california_housing.yaml

# ログレベルを変更してパイプラインを実行
python src/pipeline/mlflow/mlflow_pipeline.py --config config/pipeline_config.yaml --log-level DEBUG

# 特定のコンポーネントのみを実行
python src/pipeline/mlflow/mlflow_pipeline.py --config config/pipeline_config.yaml --components data_loader,preprocess,training
```

### Pythonコードからの実行

```python
from src.pipeline.mlflow.mlflow_pipeline import run_pipeline
from src.utils.config_manager import ConfigManager

# 設定ファイルから実行
results = run_pipeline("config/pipeline_config.yaml", "config/experiments/linear_regression_california_housing.yaml")

# 結果の表示
print(f"MLflow Run ID: {results['run_id']}")
print(f"実行時間: {results['execution_time']:.2f}秒")
print("評価指標:")
for name, value in results.get("metrics", {}).items():
    print(f"  {name}: {value:.4f}")
```

### 設定ファイルの例

パイプライン設定ファイル (`config/pipeline_config.yaml`):
```yaml
# 実験設定ファイルのパス
experiment_config_path: "config/experiments/logistic_regression_abalone.yaml"

# MLflow設定
mlflow:
  tracking_uri: "sqlite:///mlruns.db"
  experiment_name: "logistic_regression_abalone"
  tags:
    version: "1.0"
    environment: "development"

# パイプラインコンポーネント
pipeline:
  components:
    - name: "data_loader"
      enabled: true
    - name: "data_split"
      enabled: true
    - name: "preprocess"
      enabled: true
    - name: "tuning"
      enabled: true
    - name: "training"
      enabled: true
    - name: "evaluation"
      enabled: true

# 出力ディレクトリ設定
output:
  base_dir: "out/"
  data:
    raw: "out/data/raw/"
    processed: "out/data/processed/"
    split: "out/data/split/"
  models:
    saved: "out/models/saved/"
    tuned: "out/models/tuned/"
  logs: "out/logs/"
  results: "out/results/"
```

実験設定ファイル (`config/experiments/linear_regression_california_housing.yaml`):
```yaml
# データ設定
data:
  file: "data/raw/dataset.csv"
  target_col: "target"
  split:
    test_size: 0.2
    random_state: 42

# 前処理設定
preprocessing:
  steps:
    - name: missing_values
      type: missing_value_handler
      params:
        strategy: median

# モデル設定
model:
  type: "linear_regression"
  params:
    fit_intercept: false

# チューニング設定
tuning:
  enabled: true
  tuner_type: grid_search
  params:
    fit_intercept: [true, false]
    positive: [false, true]

# 評価設定
evaluation:
  metrics:
    mean_squared_error:
      enabled: true
    r2_score:
      enabled: true
```

## データ処理

### データローディング

```python
from src.data.manager import DataManager

# データ設定
data_config = {
    "file": "data/raw/dataset.csv",
    "target_col": "target"
}

# データマネージャーの作成
data_manager = DataManager(data_config)

# データの読み込み
data = data_manager.load_data("", "data/raw/dataset.csv", "csv")

# データの分割
X_train, X_test, y_train, y_test = data_manager.split_data(
    data, target_col="target", test_size=0.2, random_state=42
)

# データの保存
data_manager.save_data([X_train, X_test], ["X_train.pkl", "X_test.pkl"], "data/processed")
```

### 前処理パイプライン

```python
from src.preprocess.preprocess_pipeline import PreprocessingPipeline

# 前処理設定
preprocessing_config = {
    "steps": [
        {
            "name": "missing_values",
            "type": "missing_value_handler",
            "params": {"strategy": "median"}
        },
        {
            "name": "scaling",
            "type": "standard_scaler",
            "params": {"with_mean": True, "with_std": True}
        },
        {
            "name": "outlier_treatment",
            "type": "outlier_remover",
            "params": {"method": "iqr", "threshold": 1.5, "treatment": "clip"}
        }
    ]
}

# パイプラインの作成と適用
pipeline = PreprocessingPipeline(preprocessing_config)
X_train_processed, X_test_processed = pipeline.run(X_train, y_train, X_test)
```

## モデルの学習と評価

### 基本的な学習と評価

```python
from src.models.supervised.regression.linear import LinearRegression
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

# モデルの設定
model_config = {
    "params": {"fit_intercept": True}
}

# モデルの初期化
model = LinearRegression(model_config)

# 評価器の作成
evaluation_config = {
    "metrics": {
        "mean_squared_error": {"enabled": True},
        "r2_score": {"enabled": True}
    }
}
evaluator = Evaluator("regression", evaluation_config)

# トレーナーの作成と学習
trainer = Trainer(model, evaluator)
trainer.train(X_train_processed, y_train)

# モデルの評価
metrics = trainer.evaluate(X_test_processed, y_test)
print(metrics)  # {'mse': 0.123, 'r2': 0.85, ...}

# モデルの保存
trainer.save_model("models/saved/linear_regression.pkl")
```

### 交差検証

```python
from src.training.cross_validation.cross_validation import CrossValidator

# 交差検証の設定
cv_config = {
    "split_method": "kfold",
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
    "scoring_metric": "r2"
}

# 交差検証の実行
cross_validator = CrossValidator(trainer, cv_config)
cv_results = cross_validator.cross_validate(X_train_processed, y_train)

print(f"平均スコア: {cv_results['mean_score']:.4f}")
print(f"各フォールドのスコア: {cv_results['fold_scores']}")
```

### ハイパーパラメータチューニング

#### グリッドサーチ
```python
from src.training.tuners.grid_search import GridSearchTuner

# グリッドサーチの設定
tuning_config = {
    "params": {
        "fit_intercept": [True, False],
        "positive": [False, True]
    },
    "scoring_metric": "r2"
}

# グリッドサーチの実行
tuner = GridSearchTuner(trainer, tuning_config)
tuning_results = tuner.tune(X_train_processed, y_train)

# 最適なパラメータでモデルを更新
best_model = tuner.apply_best_params()
print(f"最適パラメータ: {tuning_results['best_params']}")
print(f"最適スコア: {tuning_results['best_score']:.4f}")
```

#### Optuna
```python
from src.training.tuners.optuna import OptunaTuner

# Optunaの設定
optuna_config = {
    "n_trials": 100,
    "timeout": 600,  # 10分
    "direction": "maximize",
    "scoring_metric": "r2",
    "params": {
        "fit_intercept": ["suggest_categorical", [[True, False]]],
        "positive": ["suggest_categorical", [[True, False]]]
    }
}

# Optunaの実行
optuna_tuner = OptunaTuner(trainer, optuna_config)
optuna_results = optuna_tuner.tune(X_train_processed, y_train)
```

## MLflowでの実験追跡

フレームワークはMLflowと完全に統合されており、実験の追跡と管理が可能です：

```bash
# MLflow UIの起動
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

MLflow UIからは以下の情報が確認できます：
- 実験のパラメータとメトリクス
- 前処理、トレーニング、チューニングの実行時間
- 評価結果とビジュアライゼーション
- モデルのバージョン管理
- 実験の比較

## 新しいコンポーネントの追加

### 新しいモデルの追加

```python
from src.models.base_model import BaseModel
from src.utils.registry import register_model
import numpy as np
from sklearn.ensemble import RandomForestRegressor

@register_model("random_forest_regression")
class RandomForestRegressionModel(BaseModel):
    """RandomForest回帰モデルの実装"""

    def __init__(self, config=None):
        super().__init__(config)
        self.model_category = "regression"

    def init_model(self):
        """モデルの初期化"""
        self._model = RandomForestRegressor(**self.get_params())

    def fit(self, X, y, **kwargs):
        """モデルの学習"""
        X = np.asarray(X)
        y = np.asarray(y)
        self._model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X):
        """予測の実行"""
        self._validate_fitted()
        X = np.asarray(X)
        return self._model.predict(X)
```

### 新しい前処理コンポーネントの追加

```python
from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.registry import register_preprocessor
import pandas as pd

@register_preprocessor("my_custom_preprocessor")
class MyCustomPreprocessor(BasePreprocessor):
    """カスタム前処理コンポーネント"""

    def __init__(self, config=None):
        super().__init__(config)
        self.param1 = config.get("param1", "default") if config else "default"

    def fit(self, data, target=None):
        """パラメータの学習"""
        # 学習ロジックを実装
        self.fitted = True
        return self

    def transform(self, data):
        """データの変換"""
        self._validate_fitted()
        # 変換ロジックを実装
        return transformed_data

    def reset(self):
        """状態のリセット"""
        self.fitted = False
        return self
```

## モジュール詳細

各モジュールの詳細なドキュメントは、それぞれのディレクトリにあるREADMEファイルを参照してください：

- [データ処理モジュール](src/data/README.md)
- [モデルモジュール](src/models/README.md)
- [トレーニングモジュール](src/training/README.md)
- [評価モジュール](src/evaluation/README.md)
- [前処理モジュール](src/preprocess/README.md)
- [パイプラインモジュール](src/pipeline/README.md)
- [設定ファイル](config/README.md)

## テスト

テストはPytestを使用して実行できます：

```bash
# すべてのテストを実行
pytest tests/

# 詳細な出力を表示
pytest tests/ -v

# 特定のテストモジュールのみ実行
pytest tests/test_data_loader.py

# カバレッジレポートの生成
pytest --cov=src tests/
```
