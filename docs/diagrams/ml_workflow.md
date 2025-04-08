# 機械学習ワークフローダイアグラム

## MLflowベースの機械学習パイプライン

以下のダイアグラムは、本フレームワークの機械学習ワークフローを示しています。MLflowを中心としたパイプラインの構造と各コンポーネントの連携が表現されています。

```mermaid
flowchart TD
    Config["設定ファイル(YAML)"] --> ConfigManager["ConfigManager設定管理"]

    subgraph MLflowパイプライン
        ConfigManager --> DataLoader["データ読み込み(data_loader)"]
        DataLoader --> DataSplit["データ分割(data_split)"]
        DataSplit --> Preprocess["前処理(preprocess)"]

        Preprocess --> |"前処理済みデータ"| Tuning["ハイパーパラメータチューニング(tuning)"]
        Tuning --> |"最適パラメータ"| Training["モデルトレーニング(training)"]
        Preprocess --> |"前処理済みデータ"| Training

        Training --> |"学習済みモデル"| Evaluation["モデル評価(evaluation)"]

        MLflow["MLflow実験管理"] <-->|"パラメータメトリクスアーティファクト"| DataLoader
        MLflow <-->|"パラメータメトリクスアーティファクト"| DataSplit
        MLflow <-->|"パラメータメトリクスアーティファクト"| Preprocess
        MLflow <-->|"パラメータメトリクスアーティファクト"| Tuning
        MLflow <-->|"パラメータメトリクスアーティファクト"| Training
        MLflow <-->|"パラメータメトリクスアーティファクト"| Evaluation
    end

    subgraph 出力
        Output_Data[(処理済みデータ)]
        Output_Model[(学習済みモデル)]
        Output_Results[(評価結果)]
        Output_Visualizations[(可視化)]
    end

    DataLoader -->|"生データ"| Output_Data
    DataSplit -->|"分割データ"| Output_Data
    Preprocess -->|"前処理済みデータ"| Output_Data
    Training -->|"モデル"| Output_Model
    Evaluation -->|"メトリクス"| Output_Results
    Evaluation -->|"プロット"| Output_Visualizations

    MLflow -->|"UI/ダッシュボード"| MLflow_UI["MLflowトラッキングUI"]
```

## データ処理プロセス

以下のダイアグラムは、データ読み込みから前処理までの詳細なプロセスを示しています。

```mermaid
flowchart TD
    Data[(データソース)] --> DL[データローダー]
    DL --> |"生データ"| Split[データ分割]

    Split --> |"トレーニングデータ"| PP_Train["前処理パイプライン(fit_transform)"]
    Split --> |"テストデータ"| PP_Test["前処理パイプライン(transform)"]

    subgraph 前処理ステップ: Example
        PP_Train --> MV[欠損値処理]
        MV --> OR[外れ値除去]
        OR --> SC[スケーリング]
        SC --> LE[ラベルエンコーディング]
    end

    PP_Train --> |"学習したパラメータ"| PP_Test

    PP_Train --> |"前処理済みトレーニングデータ"| X_train[(X_train, y_train)]
    PP_Test --> |"前処理済みテストデータ"| X_test[(X_test, y_test)]
```

## モデルトレーニングとチューニング

以下のダイアグラムは、モデルトレーニング、交差検証、ハイパーパラメータチューニングのプロセスを示しています。

```mermaid
flowchart TD
    subgraph データ入力
        X_train[(X_train, y_train)]
        X_test[(X_test, y_test)]
    end

    X_train --> |"トレーニングデータ"| Tuning

    subgraph チューニング
        Tuning[ハイパーパラメータチューニング] --> |"パラメータ探索"| CV[交差検証]
        CV --> |"検証スコア"| Tuning
        Tuning --> |"最適パラメータ"| BestParams[(最適パラメータ)]
    end

    BestParams --> |"モデル設定"| ModelInit[モデル初期化]
    ModelInit --> Training[モデルトレーニング]
    X_train --> |"トレーニングデータ"| Training

    Training --> |"学習済みモデル"| TrainedModel[(学習済みモデル)]

    TrainedModel --> |"予測"| Evaluation[モデル評価]
    X_test --> |"テストデータ"| Evaluation

    Evaluation --> |"評価指標"| Metrics[(評価指標)]
    Evaluation --> |"可視化"| Plots[(評価プロット)]
```

## MLflow実験管理

以下のダイアグラムは、MLflowを使用した実験管理とトラッキングを示しています。

```mermaid
flowchart TD
    subgraph MLflowパイプライン
        Components[パイプラインコンポーネント] --> |"実行"| Run[MLflow Run]
    end

    Run --> |"記録"| Params[パラメータ]
    Run --> |"記録"| Metrics[メトリクス]
    Run --> |"記録"| Artifacts[アーティファクト]

    Params --> MLflow[(MLflowデータベース)]
    Metrics --> MLflow
    Artifacts --> MLflow

    MLflow --> |"クエリ"| UI[MLflow UI]

    subgraph MLflow UI
        Experiments[実験リスト]
        RunDetails[実行詳細]
        MetricCharts[メトリクスチャート]
        ModelRegistry[モデルレジストリ]
    end

    UI --> Experiments
    UI --> RunDetails
    UI --> MetricCharts
    UI --> ModelRegistry
```

## コンポーネント間のデータフロー

以下のダイアグラムは、主要コンポーネント間のデータフローとファイル保存を示しています。

```mermaid
flowchart TD
    Data_Source[(データソース)] --> |"読み込み"| DataLoader[データローダー]

    DataLoader --> |"保存"| Raw_Data[/生データout/data/raw/]
    Raw_Data --> |"読み込み"| DataSplit[データ分割]

    DataSplit --> |"保存"| Split_Data[/分割データout/data/split/]
    Split_Data --> |"読み込み"| Preprocess[前処理]

    Preprocess --> |"保存"| Processed_Data[/前処理済みデータout/data/processed/]
    Processed_Data --> |"読み込み"| Tuning[チューニング]

    Tuning --> |"保存"| Best_Params[/最適パラメータout/tuning/best_params/]
    Best_Params --> |"読み込み"| Training[トレーニング]
    Processed_Data --> |"読み込み"| Training

    Training --> |"保存"| Trained_Model[/学習済みモデルout/models/saved/]
    Trained_Model --> |"読み込み"| Evaluation[評価]
    Processed_Data --> |"読み込み"| Evaluation

    Evaluation --> |"保存"| Results[/評価結果out/results/]

    subgraph MLflow
        Tracking[(トラッキングDB)]
    end

    DataLoader --> |"ログ記録"| Tracking
    DataSplit --> |"ログ記録"| Tracking
    Preprocess --> |"ログ記録"| Tracking
    Tuning --> |"ログ記録"| Tracking
    Training --> |"ログ記録"| Tracking
    Evaluation --> |"ログ記録"| Tracking
```
