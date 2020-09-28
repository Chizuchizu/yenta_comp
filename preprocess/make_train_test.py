import pandas as pd

USER_V = 4
INTERACTION_V = 2
VERSION = 5

df = pd.read_csv("../data/train.csv")


def feature_engineering(df_):
    diff_list = ["1_from_count_x_", "age_", "num_char_", "range_"]
    for diff in diff_list:
        df_[f"{diff}diff"] = df_[f"{diff}x"] - df_[f"{diff}y"]

    return df_


for mode in ["train", "test"]:
    data = pd.read_pickle(f"../data/user_agg_v{USER_V}.pkl")
    data_1 = pd.read_pickle(f"../data/data_1_v{INTERACTION_V}.pkl")
    data.index = data["user_id"]

    df = pd.read_csv(f"../data/{mode}.csv")

    df["from"] = df["from-to"].apply(lambda x: int(x.split("-")[0]))
    df["to"] = df["from-to"].apply(lambda x: int(x.split("-")[1]))

    df = df.drop(columns=["from-to"])

    # userのdataを結合
    """-------------------------"""
    data["from"] = data["user_id"].copy()
    data = data.drop(columns="user_id")
    df = pd.merge(df, data, on="from", how="left")
    data["to"] = data["from"].copy()
    data = data.drop(columns="from")
    df = pd.merge(df, data, on="to", how="left")

    data_1["from"] = data_1["user_id"].copy()
    data_1 = data_1.drop(columns="user_id")
    df = pd.merge(df, data_1, on="from", how="left")
    data_1["to"] = data_1["from"].copy()
    data_1 = data_1.drop(columns="from")
    df = pd.merge(df, data_1, on="to", how="left")

    df = df.drop(columns=["from", "to"])
    """-------------------------"""

    df = feature_engineering(df)

    print(df.info())

    df.to_pickle(f"../data/{mode}_v{VERSION}.pkl")
