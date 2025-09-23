import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error as mae

df = pd.read_csv("oedo_master_2011_2021_clean.csv", sep=",")

df_station = df.iloc[:, 0:4]
df_master_data = df.iloc[:, 3:]
df_lag = df_master_data.copy()
df_lag["passengers_lag1"] = df_master_data.groupby("stationCode")["passengers"].shift(
    1
)  # 下方向に1ずらす
df_lag["land_lag1"] = df_master_data.groupby("stationCode")["land_price"].shift(
    1
)  # groupbyで駅コードごとにまとめてシフトしている


ordered_cols = [
    "stationCode",
    "year",
    "passengers",
    "passengers_lag1",
    "land_price",
    "land_lag1",
]

df_lag = df_lag[ordered_cols]
df_lag = df_lag.dropna(subset=["passengers_lag1", "land_lag1"]).reset_index(drop=True)
df_predict_base = df_lag.copy()

# -------------------------------------------
# 1) 翌年値 target を付与（駅ごとに shift(-1)）
# -------------------------------------------
df_lag["target"] = df_lag.groupby("stationCode")["passengers"].shift(-1)

# lag1 と target が欠損の行（2017 行・2022 行）を除外
df_lag = df_lag.dropna(subset=["target"]).reset_index(drop=True)

df_lag.to_csv("oedo_target_2011_2021.csv", index=False)
# -------------------------------------------
# 2) 特徴量・目的変数の準備
# -------------------------------------------
feature_cols = [
    "stationCode",
    "passengers",
    "passengers_lag1",
    "land_price",
    "land_lag1",
]
categorical_cols = ["stationCode"]

df_trainval = df_lag.copy()  # ← これで 2012-2020 年が残ります


train_mask = df_trainval["year"] < 2020  # 2018–2019
valid_mask = df_trainval["year"] == 2020  # 2020 → 2021

X_train = df_trainval.loc[train_mask, feature_cols]
y_train = df_trainval.loc[train_mask, "target"]

X_valid = df_trainval.loc[valid_mask, feature_cols]
y_valid = df_trainval.loc[valid_mask, "target"]

# LightGBM Dataset
dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
dvalid = lgb.Dataset(
    X_valid, label=y_valid, categorical_feature=categorical_cols, reference=dtrain
)

params = dict(objective="regression", metric="mae")

model = lgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dvalid],
    valid_names=["valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
)

print(f"Best iteration : {model.best_iteration}")
print(f"Valid MAE      : {model.best_score['valid']['l1']:.3f}")

# -------------------------------------------
# 3) train+valid 全部で再学習（best_iteration 使用）
# -------------------------------------------
full_mask = df_lag["year"] < 2022  # 2012–2021
dfull = lgb.Dataset(
    df_lag.loc[full_mask, feature_cols],
    label=df_lag.loc[full_mask, "target"],
    categorical_feature=categorical_cols,
)

final_model = lgb.train(params, dfull, num_boost_round=model.best_iteration)
# Best iteration:検証用データ（valid）で評価指標が最良値をとった時点のブースティング回数

# -------------------------------------------
# 4) 2022 予測（21 行 → 22）
# -------------------------------------------
df_pred21 = df_predict_base[df_predict_base["year"] == 2021].copy()
X_21 = df_pred21[feature_cols]
pred_2022 = final_model.predict(X_21)

pred22_df = df_pred21[["stationCode"]].copy()
pred22_df["year"] = 2022
pred22_df["pred_passengers_2022"] = pred_2022

# -------------------------------------------
# 5) 2023 予測（再帰：22 行を作って → 23）
#    - まず 22 年用特徴量 DataFrame を複製して調整
# -------------------------------------------
# 21 行→22 をコピーして year だけ +1
X_22 = X_21.copy()
X_22["year"] = 2022
X_22 = X_22[feature_cols]
# lag1 は「実 2021 passengers」
# passengers (当年) を pred_2022 で置換
X_22.loc[:, "passengers_lag1"] = X_21["passengers"].values
X_22.loc[:, "passengers"] = pred_2022

pred_2023 = final_model.predict(X_22)

pred23_df = pred22_df[["stationCode"]].copy()
pred23_df["pred_passengers_2023"] = pred_2023

# -------------------------------------------
# 6) 実測データと MAE 評価
# -------------------------------------------
actual22 = pd.read_csv("oedo_2022.csv")
actual23 = pd.read_csv("oedo_2023.csv")

mae22 = mae(
    actual22.set_index("stationCode").loc[pred22_df["stationCode"], "passengers"],
    pred22_df["pred_passengers_2022"],
)
mae23 = mae(
    actual23.set_index("stationCode").loc[pred23_df["stationCode"], "passengers"],
    pred23_df["pred_passengers_2023"],
)

print(f"\nMAE 2022 = {mae22:,.2f}")
print(f"MAE 2023 = {mae23:,.2f}")

# ⑩ 予測・実測・誤差をまとめて Excel へ
# ──────────────────────────────
# 1) 実測ファイルを読み込み、列名をわかりやすく変更
actual22 = pd.read_csv("oedo_2022.csv")  # ← 路線情報も入っている 6 列ファイル
actual23 = pd.read_csv("oedo_2023.csv")

actual22 = actual22.rename(columns={"passengers": "actual_2022"})
actual23 = actual23.rename(columns={"passengers": "actual_2023"})

# 2) 予測と実測をマージ
result = (
    pred22_df.merge(  # stationCode, pred_passengers_2022
        pred23_df, on="stationCode"
    )  # + pred_passengers_2023
    .merge(  # 駅マスタ情報と 2022 実測
        actual22[
            [
                "stationCode",
                "stationName",
                "administrationCompany",
                "routeName",
                "actual_2022",
            ]
        ],
        on="stationCode",
        how="left",
    )
    .merge(  # 2023 実測
        actual23[["stationCode", "actual_2023"]], on="stationCode", how="left"
    )
)

# ③ 予測値を整数に丸め（四捨五入） & 誤差を計算
result["pred_passengers_2022"] = result["pred_passengers_2022"].round().astype(int)
result["pred_passengers_2023"] = result["pred_passengers_2023"].round().astype(int)

result["error_2022"] = (result["actual_2022"] - result["pred_passengers_2022"]).abs()
result["error_2023"] = (result["actual_2023"] - result["pred_passengers_2023"]).abs()

# 2023 誤差 − 2022 誤差（＋なら誤差増、−なら減）
result["error_delta"] = result["error_2023"] - result["error_2022"]

# ④ 欲しい順に並べ替え
result = result[
    [
        "stationCode",
        "stationName",
        "administrationCompany",
        "routeName",
        "actual_2022",
        "pred_passengers_2022",
        "error_2022",
        "actual_2023",
        "pred_passengers_2023",
        "error_2023",
        "error_delta",
    ]
]

# ⑤ Excel 保存
output_path = "oedo_predictions_with_errors.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    result.to_excel(writer, index=False, sheet_name="predictions")

print(f"\nExcel saved ➜ {output_path}")
