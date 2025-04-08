# 評価モジュール (src/evaluation/)

## 概要
このモジュールは、機械学習モデルの評価とパフォーマンス測定のための機能を提供します。様々な評価指標を通じてモデルの性能を数値化し、モデルの比較や選択をサポートします。また、MLflowとの統合機能を含み、評価結果の追跡と可視化が可能です。

## コンポーネント

### Evaluator
- モデル評価の中心クラス
- モデルタイプに基づいて自動的に適切な評価指標を選択
- カスタム評価指標の追加が可能
- パラメータによる評価指標のカスタマイズ
- 設定ファイルベースの柔軟な構成

#### 主な機能
- `evaluate(model, X, y)`: 設定された評価指標に基づいてモデルを評価
- `apply_evaluation_config(config)`: 評価設定の適用
- `_resolve_metric_function(function_reference)`: 文字列から評価指標関数への解決
- `get_primary_metric()`: 主要評価指標の取得

## モデルタイプ別のデフォルト評価指標

### 回帰モデル (`model_category = "regression"`)
- **Mean Squared Error (MSE)**: 予測と実測値の二乗誤差の平均
  - 大きな誤差に対してより厳しいペナルティを与える
  - スケールに依存するため、異なるスケールのデータセット間で比較する際は注意が必要
- **Mean Absolute Error (MAE)**: 予測と実測値の絶対誤差の平均
  - 外れ値に対してMSEよりも頑健
  - 誤差の大きさを直接解釈しやすい
- **R² Score (決定係数)**: モデルによって説明される分散の割合
  - 0〜1の範囲（1が最良、0が最悪）で解釈しやすい
  - 負の値も取りうる（モデルが定数予測よりも悪い場合）

### 分類モデル (`model_category = "classification"`)
- **Accuracy**: 正確に分類されたサンプルの割合
  - 解釈が簡単でわかりやすい
  - クラス不均衡データでは注意が必要
- **Precision**: 陽性と予測したサンプルのうち、実際に陽性であった割合
  - 偽陽性を最小化したい場合に重要
- **Recall**: 実際の陽性サンプルのうち、正しく陽性と予測できた割合
  - 偽陰性を最小化したい場合に重要
- **F1 Score**: PrecisionとRecallの調和平均
  - Precision/Recallのバランスを取った指標

## 使用例

### 基本的な使用法

```python
from src.evaluation.evaluator import Evaluator
from src.models.supervised.regression.linear import LinearRegression

# モデルの初期化と学習
model = LinearRegression({"params": {"fit_intercept": True}})
model.fit(X_train, y_train)

# 評価器の初期化（モデルタイプに基づいて適切な評価指標が自動設定される）
evaluator = Evaluator(model.model_category)

# モデルの評価
metrics = evaluator.evaluate(model, X_test, y_test)
print(metrics)  # {'mean_squared_error': 0.123, 'mean_absolute_error': 0.098, 'r2_score': 0.85}
```

### 設定ベースの評価

```python
from src.evaluation.evaluator import Evaluator

# 回帰モデル用の評価設定
regression_evaluation_config = {
    "primary_metric": "r2_score",
    "metrics": {
        "mean_squared_error": {"enabled": True},
        "mean_absolute_error": {"enabled": True},
        "r2_score": {"enabled": True},
        "explained_variance_score": {"enabled": True},
        "median_absolute_error": {"enabled": False}  # 無効化された指標
    }
}

# 評価器の初期化と設定適用
evaluator = Evaluator("regression", regression_evaluation_config)
metrics = evaluator.evaluate(model, X_test, y_test)
```

分類モデル用の例：

```python
from src.evaluation.evaluator import Evaluator

# 分類モデル用の評価設定
classification_evaluation_config = {
    "primary_metric": "f1_score",
    "metrics": {
        "accuracy_score": {"enabled": True},
        "precision_score": {
            "enabled": True,
            "params": {"average": "weighted", "zero_division": 0}
        },
        "recall_score": {
            "enabled": True,
            "params": {"average": "weighted"}
        },
        "f1_score": {
            "enabled": True,
            "params": {"average": "weighted"}
        },
        "roc_auc_score": {"enabled": False}  # 無効化された指標
    }
}

# 評価器の初期化と設定適用
evaluator = Evaluator("classification", classification_evaluation_config)
metrics = evaluator.evaluate(model, X_test, y_test)
```

### 外部評価指標の使用

```python
from sklearn.metrics import median_absolute_error
from src.evaluation.evaluator import Evaluator

# 評価設定の作成
config = {
    "metrics": {
        "sklearn.metrics.median_absolute_error": {
            "enabled": True
        }
    }
}

# Evaluatorのインスタンス化と設定適用
evaluator = Evaluator("regression")
evaluator.apply_evaluation_config(config)

# 評価の実行
results = evaluator.evaluate(model, X_test, y_test)
```

## MLflowとの統合

```python
import mlflow
from src.evaluation.evaluator import Evaluator

# Evaluatorのインスタンス化
evaluator = Evaluator("regression")

# モデルの評価
metrics = evaluator.evaluate(model, X_test, y_test)

# 評価指標をMLflowに記録
with mlflow.start_run():
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
```

MLflowパイプラインでの使用例：

```python
from src.pipeline.mlflow.components.evaluate import run_evaluate
from src.utils.config_manager import ConfigManager

# 設定マネージャーの作成
config_manager = ConfigManager.from_yaml_files("config/pipeline_config.yaml", "config/experiments/linear_regression_california_housing.yaml")

# 評価の実行
evaluate_result = run_evaluate(config_manager)

# 結果の出力
for metric_name, metric_value in evaluate_result.get("metrics", {}).items():
    print(f"{metric_name}: {metric_value:.4f}")
```

## 設定ファイルでの評価設定

YAML設定ファイルでの評価設定例：

```yaml
# 回帰モデル用
evaluation:
  primary_metric: "r2_score"
  metrics:
    mean_squared_error:
      enabled: true
    mean_absolute_error:
      enabled: true
    r2_score:
      enabled: true
    explained_variance_score:
      enabled: true
```

```yaml
# 分類モデル用
evaluation:
  primary_metric: "f1_score"
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

## 拡張方法

### カスタム評価指標の追加

#### 直接追加する方法

```python
from src.evaluation.evaluator import COMMON_METRICS

# カスタム評価指標の定義
def custom_weighted_error(y_true, y_pred, weights=None):
    """カスタム重み付きエラー指標"""
    import numpy as np
    if weights is None:
        weights = np.ones_like(y_true)
    return np.average(np.abs(y_true - y_pred), weights=weights)

# グローバル指標辞書に追加
COMMON_METRICS["custom_weighted_error"] = custom_weighted_error

# 評価設定での使用
custom_config = {
    "metrics": {
        "custom_weighted_error": {
            "enabled": True,
            "params": {"weights": sample_weights}
        }
    }
}

# 評価器の初期化と設定適用
evaluator = Evaluator("regression", custom_config)
```

#### 動的なインポートを使用する方法

```python
# カスタムモジュールに評価指標を定義
# my_metrics.py
def geometric_mean_error(y_true, y_pred):
    """予測誤差の幾何平均を計算"""
    import numpy as np
    errors = np.abs(y_true - y_pred)
    return np.exp(np.mean(np.log(errors + 1e-7)))

# 評価設定での使用
custom_config = {
    "metrics": {
        "my_metrics.geometric_mean_error": {
            "enabled": True
        }
    }
}

# 評価器はmy_metrics.pyをインポートして関数を解決する
evaluator = Evaluator("regression", custom_config)
```

## 注意点とベストプラクティス

- **評価指標の選択**: タスクの性質に合った評価指標を選択
  - 回帰: MSE/MAEはエラーの大きさを直接測定、R²は説明力を測定
  - 分類: 精度、適合率、再現率、F1スコアはそれぞれ異なる側面を評価

- **クラス不均衡**: 分類問題でクラスが不均衡な場合は、精度だけでなく適合率や再現率も考慮

- **マルチクラス分類**: マルチクラス分類では、`average`パラメータを適切に設定
  - `macro`: 各クラスを均等に扱う
  - `weighted`: クラスの出現頻度で重み付け
  - `micro`: 個々の予測を集計して計算

- **多面的評価**: 単一の指標だけでなく、複数の指標を用いて総合的に評価

- **コスト考慮**: 実際のビジネス価値やコストを反映した指標を選択

- **主要指標の選定**: モデル選択の基準となる主要指標を明確に定義

- **評価指標の一貫性**: 実験間で一貫した評価指標を使用して比較可能性を確保

- **MLflowとの統合**: 評価指標をMLflowに記録して実験の追跡と比較を容易に
