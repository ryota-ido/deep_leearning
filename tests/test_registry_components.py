"""
レジストリコンポーネントテスト - 登録された全てのモデル、チューナー、前処理コンポーネントが正常に動作することをテスト
"""

import inspect
import os
import sys
from typing import Any, Dict, List, Type

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.evaluation.evaluator import Evaluator
from src.models.base_model import BaseModel
from src.preprocess.base_preprocessor import BasePreprocessor
from src.preprocess.preprocess_pipeline import PreprocessingPipeline
from src.training.trainer import Trainer
from src.training.tuners.base_tuner import BaseTuner
from src.utils.registry import model_registry, preprocessor_registry, tuner_registry


@pytest.fixture
def regression_data():
    """回帰モデルテスト用のサンプルデータ（拡張版）"""
    X = np.random.rand(100, 3) * 10
    coef = np.array([1.0, 2.0, 3.0])
    y = X @ coef + 0.5  # 線形回帰のターゲット
    return X, y


@pytest.fixture
def classification_data():
    """分類モデルテスト用のサンプルデータ（拡張版）"""
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_classes=2, random_state=42)
    return X, y


@pytest.fixture
def pandas_regression_data():
    X = np.random.rand(100, 3) * 10
    coef = np.array([1.0, 2.0, 3.0])
    y = X @ coef + 0.5  # 線形回帰のターゲット
    df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
    y = pd.Series(y, name="target")
    return df, y


@pytest.fixture
def pandas_classification_data():
    """分類モデルテスト用のDataFrameデータ"""
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature{i+1}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    return df, y


@pytest.fixture
def temp_directory(tmpdir):
    """一時ディレクトリ"""
    return tmpdir


class TestModelRegistry:
    """モデルレジストリのテスト"""

    def setup_method(self):
        """テスト前の準備"""
        # モデルレジストリの更新を確認
        model_registry.discover_modules(base_package="src.models", base_class=BaseModel)

    def test_model_registry_has_entries(self):
        """モデルレジストリにエントリが存在するか"""
        assert len(model_registry._registry) > 0, "モデルレジストリが空です"
        print(f"登録されているモデル: {list(model_registry._registry.keys())}")

    def test_all_regression_models(self, regression_data):
        """全ての回帰モデルが正常に機能するか"""
        X, y = regression_data

        for model_name, model_class in model_registry._registry.items():
            # クラスのインスタンスを作成
            try:
                # モデルタイプが回帰のものだけテスト
                model_instance = model_class({"params": {}})
                if model_instance.model_category != "regression":
                    continue

                print(f"回帰モデルをテスト中: {model_name}")

                # モデルの初期化
                model_instance.init_model()

                # 学習
                model_instance.fit(X, y)
                assert model_instance.is_fitted, f"{model_name} の学習に失敗しました"

                # 予測
                predictions = model_instance.predict(X)
                assert len(predictions) == len(y), f"{model_name} の予測サイズが不正です"

                # 保存と読み込み
                temp_path = f"/tmp/{model_name}_test.pkl"
                model_instance.save(temp_path)
                loaded_model = model_class.load(temp_path)
                loaded_predictions = loaded_model.predict(X)
                assert np.array_equal(predictions, loaded_predictions), f"{model_name} の保存/読み込みで予測が変わりました"

                # 後片付け
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                pytest.fail(f"モデル {model_name} のテストに失敗: {str(e)}")

    def test_all_classification_models(self, classification_data):
        """全ての分類モデルが正常に機能するか"""
        X, y = classification_data

        for model_name, model_class in model_registry._registry.items():
            # クラスのインスタンスを作成
            try:
                # モデルタイプが分類のものだけテスト
                model_instance = model_class({"params": {}})
                if model_instance.model_category != "classification":
                    continue

                print(f"分類モデルをテスト中: {model_name}")

                # モデルの初期化
                model_instance.init_model()

                # 学習
                model_instance.fit(X, y)
                assert model_instance.is_fitted, f"{model_name} の学習に失敗しました"

                # 予測
                predictions = model_instance.predict(X)
                assert len(predictions) == len(y), f"{model_name} の予測サイズが不正です"

                # 保存と読み込み
                temp_path = f"/tmp/{model_name}_test.pkl"
                model_instance.save(temp_path)
                loaded_model = model_class.load(temp_path)
                loaded_predictions = loaded_model.predict(X)
                assert np.array_equal(predictions, loaded_predictions), f"{model_name} の保存/読み込みで予測が変わりました"

                # 後片付け
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                pytest.fail(f"モデル {model_name} のテストに失敗: {str(e)}")

    def test_models_with_trainer(self, regression_data, classification_data):
        """全てのモデルがTrainerと連携できるか"""
        reg_X, reg_y = regression_data
        clf_X, clf_y = classification_data

        # 各モデルのテスト
        for model_name, model_class in model_registry._registry.items():
            try:
                model_instance = model_class({"params": {}})

                # モデルタイプに基づいてデータを選択
                if model_instance.model_category == "regression":
                    X, y = reg_X, reg_y
                    evaluator = Evaluator("regression")
                elif model_instance.model_category == "classification":
                    X, y = clf_X, clf_y
                    evaluator = Evaluator("classification")
                else:
                    continue  # サポートされていないモデルタイプはスキップ

                print(f"モデル {model_name} をTrainerでテスト中")

                # トレーナーの作成
                trainer = Trainer(model_instance, evaluator)

                # トレーニングと評価
                trainer.train(X, y)
                metrics = trainer.evaluate(X, y)

                assert model_instance.is_fitted, f"{model_name} の学習に失敗しました"
                assert len(metrics) > 0, f"{model_name} の評価指標が空です"

            except Exception as e:
                pytest.fail(f"モデル {model_name} のTrainerテストに失敗: {str(e)}")


class TestPreprocessorRegistry:
    """前処理レジストリのテスト"""

    def setup_method(self):
        """テスト前の準備"""
        # 前処理レジストリを更新
        preprocessor_registry.discover_modules(base_package="src.preprocess", base_class=BasePreprocessor)

    def test_preprocessor_registry_has_entries(self):
        """前処理レジストリにエントリが存在するか"""
        assert len(preprocessor_registry._registry) > 0, "前処理レジストリが空です"
        print(f"登録されている前処理: {list(preprocessor_registry._registry.keys())}")

    def test_all_preprocessors(self, pandas_regression_data):
        """全ての前処理コンポーネントが正常に機能するか"""
        X, y = pandas_regression_data

        # 各前処理コンポーネントをテスト
        for preprocessor_name, preprocessor_class in preprocessor_registry._registry.items():
            try:
                print(f"前処理コンポーネントをテスト中: {preprocessor_name}")

                # コンストラクタのパラメータを取得
                try:
                    # デフォルトの空設定でインスタンス化
                    preprocessor = preprocessor_class({})
                except TypeError:
                    # パラメータがない場合
                    preprocessor = preprocessor_class()

                # fit
                preprocessor.fit(X, y)
                assert preprocessor.fitted, f"{preprocessor_name} のfit処理に失敗しました"

                # transform
                transformed_X = preprocessor.transform(X)
                assert isinstance(transformed_X, pd.DataFrame), f"{preprocessor_name} のtransformの戻り値がDataFrameではありません"
                assert len(transformed_X) == len(X), f"{preprocessor_name} のtransform後のデータ長が不正です"

                # reset
                preprocessor.reset()
                assert not preprocessor.fitted, f"{preprocessor_name} のreset処理に失敗しました"

            except Exception as e:
                pytest.fail(f"前処理 {preprocessor_name} のテストに失敗: {str(e)}")

    def test_preprocessor_pipeline(self, pandas_regression_data):
        """前処理パイプラインで全ての前処理コンポーネントが利用できるか"""
        X, y = pandas_regression_data

        # 登録されている全前処理を含むパイプライン設定を作成
        preprocessing_config = {"steps": []}

        for i, (preprocessor_name, _) in enumerate(preprocessor_registry._registry.items()):
            step_config = {"name": f"step_{i}", "type": preprocessor_name, "params": {}}
            preprocessing_config["steps"].append(step_config)

        try:
            # パイプラインの作成
            pipeline = PreprocessingPipeline(preprocessing_config)

            # パイプラインの実行
            X_train_processed, X_test_processed = pipeline.run(X, y, X.copy())

            assert isinstance(X_train_processed, pd.DataFrame), "処理後のトレーニングデータがDataFrameではありません"
            assert isinstance(X_test_processed, pd.DataFrame), "処理後のテストデータがDataFrameではありません"
            assert len(X_train_processed) == len(X), "処理後のトレーニングデータ長が不正です"
            assert len(X_test_processed) == len(X), "処理後のテストデータ長が不正です"

        except Exception as e:
            pytest.fail(f"前処理パイプラインのテストに失敗: {str(e)}")


class TestTunerRegistry:
    """チューナーレジストリのテスト"""

    def setup_method(self):
        """テスト前の準備"""
        # チューナーレジストリを更新
        tuner_registry.discover_modules(base_package="src.training.tuners", base_class=BaseTuner)

    def test_tuner_registry_has_entries(self):
        """チューナーレジストリにエントリが存在するか"""
        assert len(tuner_registry._registry) > 0, "チューナーレジストリが空です"
        print(f"登録されているチューナー: {list(tuner_registry._registry.keys())}")

    def test_grid_search_tuner(self, regression_data):
        """GridSearchTunerのテスト"""
        if "grid_search" not in tuner_registry._registry:
            pytest.skip("GridSearchTunerがレジストリに存在しません")

        X, y = regression_data

        try:
            # 回帰モデルを取得
            model_class = model_registry.get("linear_regression")
            model = model_class({"params": {"fit_intercept": True}})

            # 評価器の作成
            evaluator = Evaluator("regression")

            # トレーナーの作成
            trainer = Trainer(model, evaluator)

            # チューニング設定
            tuning_config = {"params": {"fit_intercept": [True, False]}, "scoring_metric": "r2"}

            # チューナーの作成
            tuner_class = tuner_registry.get("grid_search")
            tuner = tuner_class(trainer, tuning_config)

            # チューニングの実行
            tuning_results = tuner.tune(X, y)

            assert "best_params" in tuning_results, "チューニング結果に最適パラメータがありません"
            assert "best_score" in tuning_results, "チューニング結果に最適スコアがありません"

            # 最適パラメータでモデルを更新
            def is_subset(dict1, dict2):
                return all(item in dict2.items() for item in dict1.items())

            best_trainer = tuner.apply_best_params()
            assert is_subset(tuning_results["best_params"], best_trainer.model.get_params()), "最適パラメータが正しく適用されていません"

        except Exception as e:
            pytest.fail(f"GridSearchTunerのテストに失敗: {str(e)}")

    def test_all_tuners_interface(self, regression_data):
        """全てのチューナーが共通インターフェースを実装しているか"""
        X, y = regression_data

        # 各チューナーのインターフェースをチェック
        for tuner_name, tuner_class in tuner_registry._registry.items():
            # 基本的なメソッドが実装されているか
            assert hasattr(tuner_class, "tune"), f"{tuner_name} にtuneメソッドがありません"
            assert hasattr(tuner_class, "apply_best_params"), f"{tuner_name} にapply_best_paramsメソッドがありません"
            assert hasattr(tuner_class, "get_results"), f"{tuner_name} にget_resultsメソッドがありません"

            # コンストラクタの引数を確認
            signature = inspect.signature(tuner_class.__init__)
            params = signature.parameters
            assert "trainer" in params, f"{tuner_name} のコンストラクタがtrainer引数を受け取りません"
            assert "tuning_config" in params, f"{tuner_name} のコンストラクタがtuning_config引数を受け取りません"


if __name__ == "__main__":
    pytest.main(["-v", "test_registry_components.py"])
