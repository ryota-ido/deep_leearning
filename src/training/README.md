# トレーニングモジュール (src/training/)

## 概要
このモジュールは、モデルのトレーニング、評価、交差検証、ハイパーパラメータチューニングのための機能を提供します。様々なトレーニング戦略と最適化手法をサポートし、MLflowとの連携機能も含まれています。

## コンポーネント

### Trainer
- トレーニングの基本クラス
- モデルとEvaluatorを組み合わせて使用
- 主な機能:
  - `train(X, y, **kwargs)`: モデルのトレーニング
  - `evaluate(X, y)`: モデルの評価（Evaluatorと連携）
  - `save_model(filepath)`: モデルの保存
  - `load_model(filepath)`: モデルの読み込み
  - `get_params()`, `set_params(**params)`: パラメータの管理

### CrossValidator
- 様々な交差検証戦略を実装
- Trainerと連携して各フォールドでトレーニングと評価を実行
- サポートされる交差検証手法:
  - K-Fold: データをK個に分割
  - 層化K-Fold: クラスの分布を維持
  - 時系列分割: 時系列データ向け
- 主な機能:
  - `cross_validate(X, y, **kwargs)`: 交差検証の実行
  - `_score(trainer, X, y)`: モデルのスコア計算

### ハイパーパラメータチューニング
`src/training/tuners/` ディレクトリに実装されたモジュール化されたチューニング機能:

- `BaseTuner`: すべてのチューナーの基底クラス
  - `tune(X, y)`: チューニングの実行（抽象メソッド）
  - `apply_best_params()`: 最適パラメータの適用
  - `get_results()`: チューニング結果の取得

- `GridSearchTuner`: グリッドサーチによるパラメータチューニング
  - sklearn.model_selection.GridSearchCVのラッパー
  - すべてのパラメータの組み合わせを網羅的に探索

- `OptunaTuner`: Optunaを用いたベイズ最適化
  - ベイズ最適化、進化アルゴリズムなどの高度な手法をサポート
  - カスタムパラメータ探索空間の定義

## 使用例

### 基本的なトレーニング

```python
from src.models.supervised.regression.linear import LinearRegression
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

# モデルの初期化
model = LinearRegression({"params": {"fit_intercept": True}})

# 評価器の初期化
evaluator = Evaluator("regression")

# トレーナーの初期化
trainer = Trainer(model, evaluator)

# モデルのトレーニング
trainer.train(X_train, y_train)

# モデルの評価
metrics = trainer.evaluate(X_test, y_test)
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
    "scoring_metric": "r2_score"
}

# 交差検証器の初期化
cross_validator = CrossValidator(trainer, cv_config)

# 交差検証の実行
cv_results = cross_validator.cross_validate(X_train, y_train)

print(f"各フォールドのスコア: {cv_results['fold_scores']}")
print(f"平均スコア: {cv_results['mean_score']:.4f}")
```

### グリッドサーチによるハイパーパラメータチューニング

```python
from src.training.tuners.grid_search import GridSearchTuner

# チューニングの設定
tuning_config = {
    "params": {
        "fit_intercept": [True, False],
        "positive": [False, True]
    },
    "scoring_metric": "r2_score"
}

# グリッドサーチチューナーの初期化
tuner = GridSearchTuner(trainer, tuning_config)

# チューニングの実行
tuning_results = tuner.tune(X_train, y_train)

# 最適なパラメータでモデルを更新
best_trainer = tuner.apply_best_params()

print(f"最適パラメータ: {tuning_results['best_params']}")
print(f"最適スコア: {tuning_results['best_score']:.4f}")
```

### Optunaによるチューニング

```python
from src.training.tuners.optuna import OptunaTuner

# Optunaチューナーの設定
optuna_config = {
    "n_trials": 100,
    "timeout": 600,  # 10分
    "direction": "maximize",
    "scoring_metric": "r2_score",
    "params": {
        "alpha": ["suggest_float", [0.001, 1.0, {"log": True}]],
        "fit_intercept": ["suggest_categorical", [[True, False]]]
    }
}

# Optunaチューナーの初期化と実行
optuna_tuner = OptunaTuner(trainer, optuna_config)
optuna_results = optuna_tuner.tune(X_train, y_train)

# 最適パラメータの適用
best_trainer = optuna_tuner.apply_best_params()

print(f"最適パラメータ: {optuna_results['best_params']}")
print(f"最適スコア: {optuna_results['best_score']:.4f}")
```

## MLflowとの統合

トレーニングモジュールはMLflowと統合されており、パイプライン内で自動的にパラメータとメトリクスを記録します。

```python
import mlflow
from src.pipeline.mlflow.components.training import run_train
from src.pipeline.mlflow.components.tuning import run_tuning
from src.utils.config_manager import ConfigManager

# 設定マネージャーの作成
config_manager = ConfigManager.from_yaml_files("config/pipeline_config.yaml", "config/experiments/linear_regression_california_housing.yaml")

# MLflow実験の開始
with mlflow.start_run():
    # トレーニングの実行
    train_result = run_train(config_manager)

    # チューニングの実行
    tuning_result = run_tuning(config_manager)

    # モデル評価の実行
    evaluate_result = run_evaluate(config_manager)
```

## チューニングモジュールの詳細

### チューナーレジストリ

すべてのチューナーは`tuner_registry`に登録され、設定から簡単に取得できます：

```python
from src.utils.registry import tuner_registry
from src.training.tuners.base_tuner import BaseTuner

# レジストリからチューナークラスを取得
tuner_registry.discover_modules(base_package="src.training.tuners", base_class=BaseTuner)
tuner_class = tuner_registry.get("grid_search")

# チューナーのインスタンス化
tuner = tuner_class(trainer, tuning_config)
```

### カスタムチューナーの追加

新しいチューニング手法を追加するには、以下の手順で実装します：

```python
from src.training.tuners.base_tuner import BaseTuner
from src.utils.registry import register_tuner

@register_tuner("bayesian_search")
class BayesianSearchTuner(BaseTuner):
    """ベイジアン最適化によるチューニング実装"""

    def __init__(self, trainer, tuning_config=None):
        super().__init__(trainer, tuning_config)
        # チューナー固有の初期化

    def tune(self, X, y):
        """ベイジアン最適化の実行"""
        # チューニングロジックの実装
        # ...

        # 結果の設定
        self.best_params = best_params
        self.best_score = best_score

        # 結果の返却
        return {
            "best_params": self.best_params,
            "best_score": self.best_score
        }
```

## 高度な使用シナリオ

### カスタム交差検証

```python
from src.training.cross_validation.cross_validation import CrossValidator
from sklearn.model_selection import TimeSeriesSplit

# 時系列データ用の交差検証設定
cv_config = {
    "split_method": "timeseries",
    "n_splits": 5,
    "scoring_metric": "neg_mean_squared_error"
}

# 交差検証の実行
ts_validator = CrossValidator(trainer, cv_config)
ts_results = ts_validator.cross_validate(X_train, y_train)
```

### アンサンブルトレーニング

```python
from src.training.cross_validation.cross_validation import CrossValidator
import numpy as np

# 交差検証設定
cv_config = {"split_method": "kfold", "n_splits": 5}
cross_validator = CrossValidator(trainer, cv_config)

# 各フォールドでモデルをトレーニング
cv_results = cross_validator.cross_validate(X_train, y_train)

# 各フォールドのモデルでアンサンブル予測
predictions = []
for fold_model in cv_results["fold_models"]:
    fold_pred = fold_model.predict(X_test)
    predictions.append(fold_pred)

# アンサンブル予測の平均
ensemble_prediction = np.mean(predictions, axis=0)
```

## 設定例

### 交差検証設定

```yaml
cross_validation:
  split_method: kfold
  n_splits: 5
  shuffle: true
  random_state: 42
  scoring_metric: r2_score
```

### グリッドサーチ設定

```yaml
tuning:
  enabled: true
  tuner_type: grid_search
  scoring_metric: r2_score
  params:
    fit_intercept: [true, false]
    positive: [false, true]
```

### Optuna設定

```yaml
tuning:
  enabled: true
  tuner_type: optuna
  n_trials: 100
  timeout: 600
  direction: maximize
  scoring_metric: r2_score
  params:
    C: ["suggest_float", [1, 100, {"log": true}]]
    solver: ["suggest_categorical", [["lbfgs", "liblinear"]]]
```

## 注意点と推奨事項

- **再現性**: 乱数シードを固定して再現性を確保
- **評価メトリクス**: 問題に合った評価メトリクスを選択（回帰ならR2やMSE、分類なら精度やF1など）
- **過学習の防止**: 交差検証を使用して汎化性能を評価
- **計算リソース**: 大規模なグリッドサーチは計算コストが高いため、Optunaなどの効率的な手法を検討
- **チューニング範囲**: 適切なパラメータ範囲を設定し、必要に応じて対数スケールを使用
- **結果の解釈**: チューニング結果をMLflowで視覚化して傾向を分析
- **モデル選択**: チューニングはモデル選択の一部として位置づけ、異なるモデルタイプも比較する
- **検証データの独立性**: チューニングに使用するデータをテストデータから完全に分離する
