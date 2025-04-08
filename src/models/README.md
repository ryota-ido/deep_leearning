# モデルモジュール (src/models/)

## 概要
このモジュールは、様々な機械学習モデルの実装を提供します。拡張性を重視した設計により、新しいモデルを簡単に追加できます。すべてのモデルは共通のインターフェースを持ち、フレームワーク内の他のコンポーネントとシームレスに連携します。

## ディレクトリ構造
```
models/
├── base_model.py        # 全モデルの基底クラス
├── supervised/
│   ├── regression/      # 回帰モデル
│   │   └── linear.py    # 線形回帰
│   └── classification/  # 分類モデル
│       └── logistic_regression.py # ロジスティック回帰
├── unsupervised/        # 教師なし学習モデル（将来の拡張用）
└── ensemble/            # アンサンブルモデル（将来の拡張用）
```

## BaseModel
抽象基底クラス(`BaseModel`)は、すべてのモデルに必要なインターフェースを定義しています。

### 主な機能
- **初期化**: `__init__(config)` - 設定からモデルを初期化
- **モデル作成**: `init_model()` - 基礎となるモデルインスタンスの作成
- **学習**: `fit(X, y, **kwargs)` - モデルをデータで学習
- **予測**: `predict(X)` - 新しいデータの予測
- **保存と読み込み**: `save(filepath)`, `load(filepath)` - モデルの永続化
- **パラメータ管理**: `get_params()`, `set_params(**params)` - モデルパラメータの取得・設定
- **状態管理**: `reset()` - モデルの状態をリセット、`_validate_fitted()` - 学習状態の検証

## サポートされているモデル

### 回帰モデル
- **LinearRegression**: scikit-learnの線形回帰モデルのラッパー（モデルタイプ: `linear_regression`）
  - パラメータ: `fit_intercept`, `positive`, `copy_X`など
  - メソッド: `fit()`, `predict()`

- **MLPRegressionModel**: scikit-learnのMLPRegressorモデルのラッパー（モデルタイプ: `mlp_regression`）
  - パラメータ: `hidden_layer_sizes`, `activation`, `solver`, `alpha`, `learning_rate`など
  - メソッド: `fit()`, `predict()`

### 分類モデル
- **LogisticRegression**: scikit-learnのロジスティック回帰モデルのラッパー（モデルタイプ: `logistic_regression`）
  - パラメータ: `C`, `penalty`, `solver`, `max_iter`など
  - メソッド: `fit()`, `predict()`

- **MLPClassificationModel**: scikit-learnのMLPClassifierモデルのラッパー（モデルタイプ: `mlp_classification`）
  - パラメータ: `hidden_layer_sizes`, `activation`, `solver`, `alpha`, `learning_rate`など
  - メソッド: `fit()`, `predict()`, `predict_proba()`

- **DecisionTreeClassifier**: scikit-learnの決定木分類モデルのラッパー（モデルタイプ: `decision_tree_classifier`）
  - パラメータ: `criterion`, `max_depth`, `min_samples_split`など
  - メソッド: `fit()`, `predict()`, `predict_proba()`, `get_feature_importances()`

- **RandomForestClassifier**: scikit-learnのランダムフォレスト分類モデルのラッパー（モデルタイプ: `random_forest_classifier`）
  - パラメータ: `n_estimators`, `criterion`, `max_depth`など
  - メソッド: `fit()`, `predict()`, `predict_proba()`, `get_feature_importances()`

- **GradientBoostingClassifier**: scikit-learnの勾配ブースティング分類モデルのラッパー（モデルタイプ: `gradient_boosting_classifier`）
  - パラメータ: `n_estimators`, `learning_rate`, `max_depth`など
  - メソッド: `fit()`, `predict()`, `predict_proba()`, `get_feature_importances()`

- **XGBoostClassifier**: XGBoostライブラリの分類モデルのラッパー（モデルタイプ: `xgboost_classifier`）
  - パラメータ: `estimators_`, `learning_rate`, `max_depth`, `objective`など
  - メソッド: `fit()`, `predict()`, `predict_proba()`, `get_feature_importances()`

- **LightGBMClassifier**: LightGBMライブラリの分類モデルのラッパー（モデルタイプ: `lightgbm_classifier`）
  - パラメータ: `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `objective`など
  - メソッド: `fit()`, `predict()`, `predict_proba()`, `get_feature_importances()`

## 使用例

### 線形回帰モデル

```python
from src.models.supervised.regression.linear import LinearRegression
import numpy as np

# モデルの初期化
config = {"params": {"fit_intercept": True, "positive": False}}
model = LinearRegression(config)

# サンプルデータの作成
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([5, 11, 17])  # 1*x1 + 2*x2 = y

# モデルのトレーニング
model.fit(X, y)

# 予測
predictions = model.predict(np.array([[7, 8]]))
print(f"予測値: {predictions[0]}")  # 予想されるOutput: 約 23

# モデルの保存
model.save('models/saved/linear_regression.pkl')

# モデルの読み込み
loaded_model = LinearRegression.load('models/saved/linear_regression.pkl')
```

### ロジスティック回帰モデル

```python
from src.models.supervised.classification.logistic_regression import LogisticRegression
import numpy as np

# モデルの初期化
config = {"params": {"C": 1.0, "penalty": "l2", "max_iter": 1000}}
model = LogisticRegression(config)

# サンプルデータの作成
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # XORに似たパターン

# モデルのトレーニング
model.fit(X, y)

# 予測
predictions = model.predict(np.array([[0, 0], [1, 1]]))
print(f"予測値: {predictions}")  # 予想されるOutput: [0, 1]

# モデルの保存
model.save('models/saved/logistic_regression.pkl')
```

## MLflowとの統合

モデルモジュールはMLflowと統合できます：

```python
import mlflow
from src.models.supervised.regression.linear import LinearRegression

# MLflowトラッキングの設定
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("regression_experiment")

# 実験実行
with mlflow.start_run():
    # パラメータのログ
    mlflow.log_param("fit_intercept", True)

    # モデルの初期化と学習
    model = LinearRegression({"params": {"fit_intercept": True}})
    model.fit(X_train, y_train)

    # メトリクスのログ
    predictions = model.predict(X_test)
    mse = ((predictions - y_test) ** 2).mean()
    mlflow.log_metric("mse", mse)

    # モデルのログ
    mlflow.sklearn.log_model(model._model, "model")
```

## パイプラインとの統合

```python
from src.pipeline.mlflow.components.training import run_train
from src.utils.config_manager import ConfigManager

# 設定マネージャーの作成
config_manager = ConfigManager.from_yaml_files("config/pipeline_config.yaml", "config/experiments/linear_regression_california_housing.yaml")

# モデルのトレーニング実行
run_train(config_manager)
```

## 新しいモデルの追加方法

### 基本的な手順

1. 適切なサブディレクトリ（`supervised/regression/`など）に新しいモデルクラスを作成
2. `BaseModel`を継承
3. 必要なメソッドを実装
   - `init_model()`
   - `fit(X, y, **kwargs)`
   - `predict(X)`
4. モデルタイプを指定（`self.model_category = "regression"` など）
5. `@register_model`デコレータでモデルを登録

### 実装例：RandomForestRegressor

```python
from src.models.base_model import BaseModel
from src.utils.registry import register_model
import numpy as np
from sklearn.ensemble import RandomForestRegressor

@register_model("random_forest_regression")
class RandomForestRegressionModel(BaseModel):
    """ランダムフォレスト回帰モデルの実装"""

    def __init__(self, config=None):
        super().__init__(config)
        self.model_category = "regression"

    def init_model(self):
        """モデルの初期化"""
        params = self.get_params()

        # デフォルトパラメータの設定
        default_params = {
            "n_estimators": 100,
            "random_state": 42
        }

        # パラメータの結合（ユーザー指定のパラメータが優先）
        all_params = {**default_params, **params}

        self._model = RandomForestRegressor(**all_params)

    def fit(self, X, y, **kwargs):
        """モデルの学習"""
        # numpyアレイに変換
        X_array = np.asarray(X)
        y_array = np.asarray(y)

        # モデルの学習
        self._model.fit(X_array, y_array, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X):
        """予測の実行"""
        self._validate_fitted()
        X_array = np.asarray(X)
        return self._model.predict(X_array)

    def get_feature_importances(self):
        """特徴量の重要度を取得（モデル固有の機能）"""
        self._validate_fitted()
        return self._model.feature_importances_
```

### 実装例：SVMClassifier

```python
from src.models.base_model import BaseModel
from src.utils.registry import register_model
import numpy as np
from sklearn.svm import SVC

@register_model("svm_classifier")
class SVMClassifier(BaseModel):
    """SVM分類モデルの実装"""

    def __init__(self, config=None):
        super().__init__(config)
        self.model_category = "classification"

    def init_model(self):
        """モデルの初期化"""
        self._model = SVC(**self.get_params())

    def fit(self, X, y, **kwargs):
        """モデルの学習"""
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        self._model.fit(X_array, y_array, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X):
        """予測の実行"""
        self._validate_fitted()
        X_array = np.asarray(X)
        return self._model.predict(X_array)

    def predict_proba(self, X):
        """確率付き予測（probability=Trueの場合のみ）"""
        self._validate_fitted()
        if not hasattr(self._model, "predict_proba"):
            raise AttributeError("このモデルはpredict_probaをサポートしていません")
        X_array = np.asarray(X)
        return self._model.predict_proba(X_array)
```

## ベストプラクティス

- **一貫したインターフェース**: すべてのモデルは同じインターフェースを提供する
- **設定の検証**: `init_model()`で無効なパラメータを検出する
- **エラーハンドリング**: 適切な例外を発生させ、有用なエラーメッセージを提供する
- **入力の検証**: すべての入力（Xとy）を検証し、必要に応じて変換する
- **状態管理**: `is_fitted`フラグを適切に管理し、`_validate_fitted()`を使用する
- **メモリ効率**: 大きなデータセットに対応するため、不要なコピーを避ける
- **ドキュメンテーション**: 各モデルとメソッドにdocstringsを提供する
- **型ヒント**: 型ヒントを用いてコードの可読性と保守性を向上させる
