# データ処理モジュール (src/data/)

## 概要
このモジュールは、機械学習プロジェクトにおけるデータの読み込み、管理、保存を担当します。異なるストレージバックエンドをサポートし、データの分割やモデルの保存・読み込みなど、データ関連の一連の操作を一貫したインターフェースで提供します。

## コンポーネント

### DataStorage
- データストレージの抽象基底クラス
- 様々なストレージバックエンド（ローカルファイルシステム、クラウドストレージなど）の基盤
- 主な抽象メソッド：
  - `load()`: データの読み込み
  - `save()`: データの保存

### LocalStorage
- ローカルファイルシステムに対するストレージ実装
- サポートするファイル形式：CSV、JSON、TSV、Pickle
- ディレクトリの自動作成機能

### DataManager
- データ操作の高レベルインターフェース
- 設定ベースのデータ管理
- 主な機能：
  - データの読み込みと保存
  - トレーニング/テストデータへの分割
  - モデルの保存と読み込み
  - 結果の保存

## 使用例

### 基本的なデータ操作

```python
from src.data.manager import DataManager

# データ設定
data_config = {
    "file": "data/raw/dataset.csv",
    "target_col": "target",
    "split": {
        "test_size": 0.2,
        "random_state": 42
    }
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
data_manager.save_data(
    [X_train, X_test, y_train, y_test],
    ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"],
    "data/processed"
)
```

### モデルの保存と読み込み

```python
from src.models.supervised.regression.linear import LinearRegression

# モデルの保存
data_manager.save_model(model, "linear_regression_model.pkl")

# モデルの読み込み
loaded_model = data_manager.load_model(LinearRegression, "linear_regression_model.pkl")
```

### 異なるデータ形式の操作

```python
# CSVデータの読み込み
csv_data = data_manager.load_data("", "data/raw/dataset.csv", "csv", header=0, index_col=0)

# JSONデータの読み込み
json_data = data_manager.load_data("", "data/raw/dataset.json", "json")

# Pickleデータの読み込み
pickle_data = data_manager.load_data("", "data/processed/processed_data.pkl", "pickle")

# CSVデータの保存
data_manager.save_data(df, "processed_data.csv", "data/processed", format="csv", index=False)
```

### 複数ファイルの一括操作

```python
# 複数のデータを一度に保存
data_manager.save_data(
    [X_train, X_test, y_train, y_test],
    ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"],
    "data/processed"
)

# 複数のデータを一度に読み込み
X_train, X_test, y_train, y_test = data_manager.load_data(
    "data/processed",
    ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"]
)
```

## MLflowパイプラインとの統合

```python
from src.pipeline.mlflow.components.data_loader import run_data_loader
from src.pipeline.mlflow.components.data_split import run_data_split
from src.utils.config_manager import ConfigManager

# 設定マネージャーの作成
config_manager = ConfigManager.from_yaml_files("config/pipeline_config.yaml", "config/experiments/linear_regression_california_housing.yaml")

# データの読み込みと分割
run_data_loader(config_manager)
run_data_split(config_manager)
```

## 拡張方法

### 新しいストレージバックエンドの追加

新しいストレージバックエンド（例：GCS、S3、Azureなど）をサポートするには、`DataStorage`抽象クラスを継承し、必要なメソッドを実装します：

```python
from src.data.storage import DataStorage

class GCSStorage(DataStorage):
    """Google Cloud Storageの実装"""

    def __init__(self, bucket_name, project_id=None):
        self.bucket_name = bucket_name
        self.project_id = project_id
        # GCSクライアントの初期化

    def load(self, source, format="pickle", **kwargs):
        """GCSからデータを読み込む"""
        # GCSからデータを読み込む実装

    def save(self, data, destination, format="pickle", **kwargs):
        """GCSにデータを保存する"""
        # GCSにデータを保存する実装
```

## 注意点

- データマネージャーは設定から自動的にストレージタイプを検出し、適切なストレージバックエンドを使用します
- 大きなデータセットを扱う場合は、適切なチャンクサイズやメモリ管理を考慮してください
- パスの指定には相対パスと絶対パスの両方が使用できますが、一貫性を保つことが重要です
- データ分割は再現性のために乱数シードを固定することが推奨されます
- モデルの保存時にはディレクトリが自動的に作成されますが、既存のファイルは上書きされます
- 複雑なデータパイプラインでは、データの整合性を確保するためにデータ依存関係を明示的に管理することが重要です
