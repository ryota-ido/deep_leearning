"""サンプルデータセットを作成するスクリプト"""

import os

import pandas as pd
from sklearn.datasets import fetch_california_housing

# カリフォルニア住宅価格データセットを読み込む
# ボストン住宅データセットは最新のscikit-learnでは非推奨のため、カリフォルニアデータセットを使用
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data["target"] = california.target

# データの確認
print("データサイズ:", data.shape)
print("特徴量:", california.feature_names)
print("サンプルデータ:")
print(data.head())

# CSVとして保存
output_dir = "../data/raw"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "dataset.csv")
data.to_csv(output_path, index=False)

print(f"\nデータセットを保存しました: {output_path}")
