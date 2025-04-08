# 設定ファイル (config/)

## 概要

このディレクトリには、機械学習フレームワークで使用される設定ファイルが含まれています。設定はYAML形式で記述され、データロード、前処理、モデルの選択、トレーニング、チューニング、評価、MLflowとの統合など、機械学習パイプラインの様々な側面を制御します。

## ディレクトリ構造

```
config/
├── pipeline_config.yaml                       # パイプライン全体の設定
├── experiments/                              # 実験固有の設定
│   ├── linear_regression_california_housing.yaml  # 線形回帰の設定例
│   └── logistic_regression_abalone.yaml           # ロジスティック回帰の設定例
└── README.md                                  # このファイル
```

## 設定ファイルの役割

フレームワークでは、2種類の設定ファイルを使用します：

1. **パイプライン設定ファイル**: パイプライン全体の構造、MLflow設定、出力ディレクトリなどを定義
2. **実験設定ファイル**: 特定の機械学習実験に関する設定（データ、モデル、前処理など）を定義

この分離により、パイプライン構造を維持したまま、異なる実験設定を簡単に切り替えることができます。

## パイプライン設定ファイルの構造

`pipeline_config.yaml`は以下のセクションから構成されています:

### MLflow設定

```yaml
# MLflow設定
mlflow:
  tracking_uri: "sqlite:///mlruns.db"   # トラッキングURI
  experiment_name: "experiment_name"    # 実験名
  tags:                                 # 実験タグ
    version: "1.0"
    environment: "development"
```

### パイプラインコンポーネント

```yaml
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
```

### 出力ディレクトリ設定

```yaml
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

## 実験設定ファイルの構造

実験設定ファイル（例：`linear_regression_california_housing.yaml`）は以下のセクションから構成されています:

### データ設定

データの場所と読み込み方法を指定します。

```yaml
data:
  file: "data/raw/dataset.csv"          # データファイルパス
  format: "csv"                         # ファイル形式（csv, excel, pickle等）
  header: 0                             # ヘッダー行 (0: 最初の行がヘッダー)
  target_col: "target"                  # 目的変数のカラム名または索引
  split:                                # データ分割設定
    test_size: 0.2                      # テストセットの割合
    random_state: 42                    # 乱数シード
    stratify: false                     # 層化サンプリングの使用有無
```

### 前処理設定

データの前処理パイプラインを定義します。

```yaml
preprocessing:
  steps:
    - name: missing_values              # ステップ名（任意）
      type: missing_value_handler       # 使用する前処理クラス（登録名）
      params:                           # 前処理クラスのパラメータ
        strategy: "mean"                # 数値列の欠損値を平均値で埋める
        categorical_strategy: "most_frequent"  # カテゴリ列は最頻値

    - name: outlier_removal
      type: outlier_remover
      params:
        method: "iqr"
        threshold: 1.5
        treatment: "clip"

    - name: scaling
      type: standard_scaler
      params:
        with_mean: true
        with_std: true
```

### モデル設定

使用するモデルとそのパラメータを指定します。

```yaml
model:
  type: "linear_regression"    # モデルタイプ（登録名）
  params:                      # モデル初期化パラメータ
    fit_intercept: true
    positive: false
    copy_X: true
```

ロジスティック回帰の例：

```yaml
model:
  type: "logistic_regression"
  params:
    C: 1.0
    max_iter: 100000
    penalty: 'l2'
    solver: 'lbfgs'
```

### 交差検証設定

モデルの交差検証方法を設定します。

```yaml
cross_validation:
  split_method: "kfold"        # 分割方法: kfold, stratifiedkfold, timeseries
  n_splits: 5                  # 分割数
  shuffle: true                # データのシャッフル
  random_state: 42             # 乱数シード
  scoring_metric: "r2_score"   # 評価指標
```

### チューニング設定

ハイパーパラメータチューニングの方法とパラメータ空間を定義します。

#### グリッドサーチの例

```yaml
tuning:
  enabled: true
  tuner_type: "grid_search"
  scoring_metric: "r2_score"
  params:
    fit_intercept: [true, false]
    positive: [false, true]
```

#### Optunaの例

```yaml
tuning:
  enabled: true
  tuner_type: "optuna"
  n_trials: 100                # 試行回数
  timeout: 600                 # タイムアウト（秒）
  direction: "maximize"        # 最適化方向
  scoring_metric: "accuracy_score"
  params:
    C: ["suggest_float", [0.001, 100, {"log": true}]]
    solver: ["suggest_categorical", [["lbfgs", "liblinear"]]]
```

### 評価設定

モデル評価に使用する指標を設定します。

```yaml
evaluation:
  primary_metric: "r2_score"    # 主要評価指標
  metrics:
    mean_squared_error:         # 評価指標名
      enabled: true             # 有効/無効
    mean_absolute_error:
      enabled: true
    r2_score:
      enabled: true
```

分類モデルの例：

```yaml
evaluation:
  primary_metric: "accuracy_score"
  metrics:
    accuracy_score:
      enabled: true
    precision_score:
      enabled: true
      params:                    # 評価指標のパラメータ
        average: "weighted"
        zero_division: 0
    recall_score:
      enabled: true
      params:
        average: "weighted"
    f1_score:
      enabled: true
      params:
        average: "weighted"
```

## 設定ファイルの使用方法

コマンドラインからMLflowパイプラインを実行：

```bash
python src/pipeline/mlflow/mlflow_pipeline.py --config config/pipeline_config.yaml
```

実験設定を明示的に指定：

```bash
python src/pipeline/mlflow/mlflow_pipeline.py --config config/pipeline_config.yaml --experiment-config config/experiments/linear_regression_california_housing.yaml
```

Pythonコードから：

```python
from src.pipeline.mlflow.mlflow_pipeline import run_pipeline

results = run_pipeline("config/pipeline_config.yaml", "config/experiments/linear_regression_california_housing.yaml")
```

## サンプル設定ファイル

### linear_regression_california_housing.yaml

カリフォルニア住宅価格データセットを使用した線形回帰モデルの設定例：

```yaml
data:
  file: "data/raw/dataset.csv"
  target_col: "target"
  split:
    test_size: 0.2
    random_state: 42

preprocessing:
  steps:
    - name: missing_values
      type: missing_value_handler
      params:
        strategy: median

model:
  type: "linear_regression"
  params:
    fit_intercept: false

tuning:
  enabled: true
  tuner_type: grid_search
  scoring_metric: r2
  params:
    fit_intercept: [true, false]
    positive: [false, true]

evaluation:
  metrics:
    mean_squared_error:
      enabled: true
    mean_absolute_error:
      enabled: true
    r2_score:
      enabled: true
```

### logistic_regression_abalone.yaml

アワビデータセットを使用したロジスティック回帰モデルの設定例：

```yaml
data:
  file: "data/raw/abalone.csv"
  header: 0
  target_col: 8

preprocessing:
  steps:
    - name: missing_values
      type: missing_value_handler
      params:
        strategy: mean
    - name: scaling
      type: standard_scaler
      params:
        with_mean: true
        with_std: true

model:
  type: "logistic_regression"
  params:
    C: 1.0
    max_iter: 100000
    penalty: 'l2'

tuning:
  enabled: true
  tuner_type: optuna
  n_trials: 100
  timeout: 600
  direction: maximize
  scoring_metric: accuracy_score
  params:
    C: ["suggest_float", [1, 100, {"log": true}]]
    solver: ["suggest_categorical", [["lbfgs", "liblinear"]]]

evaluation:
  primary_metric: "accuracy_score"
  metrics:
    accuracy_score:
      enabled: true
    precision_score:
      enabled: true
      params:
        average: "weighted"
        zero_division: 0
    recall_score:
      enabled: true
      params:
        average: "weighted"
    f1_score:
      enabled: true
      params:
        average: "weighted"
```

## カスタム設定ファイルの作成方法

新しい設定ファイルを作成する際のガイドライン：

1. 既存のサンプル設定ファイルをテンプレートとして使用
2. データパスとファイル名を正しく設定
3. データの特性に合わせて前処理パイプラインを調整
4. 解決すべき問題に適したモデルタイプを選択
5. チューニング対象のパラメータを適切に設定
6. 評価指標をモデルタイプに合わせて選択
