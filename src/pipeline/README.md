# パイプラインモジュール (src/pipeline/)

## 概要

このモジュールは、機械学習ワークフローの完全なパイプラインを提供します。データの読み込みから前処理、モデルのトレーニング、チューニング、評価まで、一連の処理を統合されたインターフェースで実行できます。MLflowによる実験管理機能が組み込まれており、実験の追跡と再現性を確保します。

## ディレクトリ構造

```
pipeline/
├── mlflow/                   # MLflowパイプライン
│   ├── components/           # パイプラインコンポーネント
│   │   ├── base_component.py # コンポーネント基底機能
│   │   ├── data_loader.py    # データ読み込みコンポーネント
│   │   ├── data_split.py     # データ分割コンポーネント
│   │   ├── preprocess.py     # 前処理コンポーネント
│   │   ├── training.py       # モデルトレーニングコンポーネント
│   │   ├── tuning.py         # ハイパーパラメータチューニングコンポーネント
│   │   └── evaluate.py       # モデル評価コンポーネント
│   ├── core.py               # MLflowパイプラインコア実装
│   └── mlflow_pipeline.py    # MLflowパイプラインのメイン実装
└── README.md                 # このファイル
```

## 主要コンポーネント

### MLflowパイプライン

MLflowパイプラインは、機械学習ワークフローの全体を管理するメインクラスです：

- 設定ファイルベースの実験構成
- パイプラインとモデル実験設定の分離
- MLflowによる実験トラッキング
  - パラメータ、メトリクス、アーティファクトの記録
  - 実験の比較と再現性の確保
- コンポーネントベースの構造
  - 分離されたステップで保守性と拡張性を向上
  - コンポーネントの選択的実行が可能
- エラーハンドリングと実行時間の測定

### パイプラインコンポーネント

パイプラインは以下の個別コンポーネントから構成されています：

#### data_loader.py
- データソース（CSV、Excel、Pickle等）からデータを読み込み
- ファイル形式の自動検出と適切な読み込み方法の選択
- MLflowへのデータ特性のログ記録

#### data_split.py
- 読み込んだデータをトレーニングセットとテストセットに分割
- 層化サンプリングのサポート
- 分割結果のMLflowへのログ記録

#### preprocess.py
- 前処理パイプラインを構成し実行
- トレーニングデータの特性に基づいてテストデータも変換
- 処理されたデータの保存とMLflowへの記録

#### training.py
- モデルのトレーニングと交差検証を実行
- 学習済みモデルの保存
- トレーニングメトリクスのMLflowへのログ記録

#### tuning.py
- 設定に基づいてハイパーパラメータチューニングを実行
- 最適なパラメータを見つけてモデルに適用
- チューニング結果のMLflowへのログ記録

#### evaluate.py
- 学習済みモデルをテストデータで評価
- 様々な評価指標を計算
- 結果の可視化とMLflowへの記録

## 使用例

### コマンドラインからの実行

```bash
# 設定ファイルからMLflowパイプラインを実行
python src/pipeline/mlflow/mlflow_pipeline.py --config config/pipeline_config.yaml

# 実験設定ファイルを明示的に指定
python src/pipeline/mlflow/mlflow_pipeline.py --config config/pipeline_config.yaml --experiment-config config/experiments/linear_regression_california_housing.yaml

# ログレベルを変更して実行
python src/pipeline/mlflow/mlflow_pipeline.py --config config/pipeline_config.yaml --log-level DEBUG

# 特定のコンポーネントのみを実行
python src/pipeline/mlflow/mlflow_pipeline.py --config config/pipeline_config.yaml --components data_loader,preprocess,training
```

### Pythonコードからの実行

```python
from src.pipeline.mlflow.mlflow_pipeline import run_pipeline

# 方法1: run_pipelineヘルパー関数を使用
results = run_pipeline("config/pipeline_config.yaml", "config/experiments/linear_regression_california_housing.yaml")

# 結果の表示
print(f"実行時間: {results['execution_time']:.2f}秒")
print("評価指標:")
for metric, value in results['metrics'].items():
    print(f"  {metric}: {value:.4f}")
```

```python
from src.pipeline.mlflow.core import MLflowPipeline
from src.utils.config_manager import ConfigManager

# 方法2: MLflowPipelineクラスを直接使用
config_manager = ConfigManager.from_yaml_files("config/pipeline_config.yaml", "config/experiments/linear_regression_california_housing.yaml")
pipeline = MLflowPipeline(config_manager)
results = pipeline.run()

# 結果の表示
print(f"MLflow Run ID: {results['run_id']}")
print(f"実行時間: {results['execution_time']:.2f}秒")
```

### コンポーネント単位での実行

```python
from src.pipeline.mlflow.components.data_loader import run_data_loader
from src.pipeline.mlflow.components.preprocess import run_preprocess
from src.utils.config_manager import ConfigManager

# 設定マネージャーの作成
config_manager = ConfigManager.from_yaml_files("config/pipeline_config.yaml", "config/experiments/linear_regression_california_housing.yaml")

# MLflow実験の開始
with mlflow.start_run():
    # データ読み込み
    data_result = run_data_loader(config_manager)

    # データ分割
    split_result = run_data_split(config_manager)

    # 前処理
    preprocess_result = run_preprocess(config_manager)

    # 結果の確認
    print(f"データ読み込み: {data_result['status']}")
    print(f"データ分割: {split_result['status']}")
    print(f"前処理: {preprocess_result['status']}")
```

## 設定ファイルの構造

MLflowパイプラインは、以下のような構造のYAML設定ファイルを使用します：

### パイプライン設定（config/pipeline_config.yaml）

```yaml
# 実験設定ファイルのパス
experiment_config_path: "config/experiments/linear_regression_california_housing.yaml"

# MLflow設定
mlflow:
  tracking_uri: "sqlite:///mlruns.db"
  experiment_name: "linear_regression_california_housing"
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

### 実験設定（config/experiments/linear_regression_california_housing.yaml）

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
    - name: "missing_values"
      type: "missing_value_handler"
      params:
        strategy: "median"
    # 他の前処理ステップ...

# モデル設定
model:
  type: "linear_regression"
  params:
    fit_intercept: true
    # その他のモデルパラメータ...

# チューニング設定
tuning:
  enabled: true
  tuner_type: "grid_search"
  scoring_metric: "r2_score"
  params:
    # チューニング対象パラメータ...
    fit_intercept: [true, false]
    positive: [false, true]

# 評価設定
evaluation:
  primary_metric: "r2_score"
  metrics:
    mean_squared_error:
      enabled: true
    # その他の評価指標...
```

## ConfigManager

`ConfigManager`クラスは、パイプライン設定と実験設定を一元管理し、コンポーネント間の一貫した設定アクセスを提供します：

- パイプライン設定と実験設定の統合
- 標準的な出力パスの管理
- コンポーネントの有効・無効状態の管理
- 設定セクションへの簡単なアクセス

```python
from src.utils.config_manager import ConfigManager

# 設定マネージャーの作成
config_manager = ConfigManager.from_yaml_files("config/pipeline_config.yaml", "config/experiments/linear_regression_california_housing.yaml")

# データ設定の取得
data_config = config_manager.get_data_config()

# 出力パスの取得
processed_dir = config_manager.get_output_path("data", "processed")

# コンポーネントの有効・無効確認
is_tuning_enabled = config_manager.is_component_enabled("tuning")
```

## MLflowとの統合

MLflowパイプラインは以下の情報を自動的に記録します：

1. **パラメータ**:
   - データセット情報（サイズ、分割など）
   - モデルパラメータ
   - 前処理パラメータ
   - チューニング設定

2. **メトリクス**:
   - 全ての評価指標
   - 実行時間（全体、前処理、トレーニング、チューニング、評価）

3. **アーティファクト**:
   - モデルファイル
   - 特徴量重要度（対応モデルのみ）
   - 評価結果の可視化

### MLflow UIの起動

```bash
# MLflow UIの起動
mlflow ui --backend-store-uri sqlite:///mlruns.db

# 特定のポートで起動
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5001
```

## 新しいコンポーネントの追加

### 基本的な手順

1. `src/pipeline/mlflow/components/` に新しいコンポーネントファイルを作成
2. `@component`デコレータを使用してコンポーネント関数を定義
3. `MLflowPipeline`の`_get_pipeline_components`メソッドにコンポーネントを追加

```python
# src/pipeline/mlflow/components/custom_component.py
from src.pipeline.mlflow.components.base_component import component
from src.utils.config_manager import ConfigManager
import mlflow

@component("custom_component")
def run_custom_component(config_manager: ConfigManager):
    """
    カスタムコンポーネントの実装

    Args:
        config_manager: 設定マネージャーインスタンス

    Returns:
        実行結果を含む辞書
    """
    # コンポーネント固有の設定取得
    custom_config = config_manager.get("custom_section", {})

    # 処理の実装
    # ...

    # MLflowへのログ記録
    mlflow.log_param("custom.parameter", "value")
    mlflow.log_metric("custom.metric", 0.95)

    # 結果の返却
    return {"status": "success", "result": "custom_result"}
```

`MLflowPipeline`への追加：

```python
def _get_pipeline_components(self) -> List[Dict[str, Any]]:
    """
    Get list of pipeline components with their configuration.
    """
    return [
        {"name": "data_loader", "function": run_data_loader},
        {"name": "data_split", "function": run_data_split},
        {"name": "preprocess", "function": run_preprocess},
        {"name": "tuning", "function": run_tuning},
        {"name": "training", "function": run_train},
        {"name": "evaluation", "function": run_evaluate},
        {"name": "custom_component", "function": run_custom_component},  # 新しいコンポーネント
    ]
```

## パイプラインのカスタマイズ

パイプラインをカスタマイズするには、`MLflowPipeline`クラスを拡張します：

```python
from src.pipeline.mlflow.core import MLflowPipeline
import mlflow

class CustomPipeline(MLflowPipeline):
    """カスタムパイプラインの実装"""

    def _get_pipeline_components(self):
        """カスタムコンポーネントリストの定義"""
        # 基本コンポーネントを継承
        components = super()._get_pipeline_components()

        # 新しいコンポーネントを追加
        components.append({"name": "my_custom_component", "function": run_my_custom_component})

        return components

    def run(self, selected_components=None):
        """カスタムパイプライン実行ロジック"""
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            # 標準のコンポーネント実行
            results = super().run(selected_components)

            # カスタム後処理
            mlflow.log_metric("custom_pipeline_metric", 1.0)

            return results
```

## エラーハンドリング

MLflowパイプラインは、実行中のエラーを適切に処理し、MLflowに記録します：

- 設定エラー: ファイルが存在しない、YAML構文エラーなど
- データ処理エラー: ファイルが見つからない、形式が不正など
- モデルエラー: 互換性のないパラメータ、学習失敗など
- MLflow接続エラー

エラーが発生した場合でも、可能な限り情報を記録し、適切なエラーメッセージを返します。

## 注意点と制限事項

- MLflowとの統合には、適切なMLflow環境が必要です
- 大規模なデータセットでは、メモリ使用量に注意してください
- 計算負荷の高いチューニングを行う場合は、`timeout`パラメータを設定することをお勧めします
- 本番環境で使用する場合は、適切なエラーハンドリングとロギングを追加してください
- 各コンポーネントは独立して動作するため、データの受け渡しはファイルシステムを介して行われます
