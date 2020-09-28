import pandas as pd

VERSION = 4

df = pd.read_csv("../data/train.csv")

# target = train["score"]

for mode in ["train", "test"]:
    data = pd.read_pickle(f"../data/user_agg_v{VERSION}.pkl")
    data_1 = pd.read_pickle("../data/data_1_v1.pkl")
    data.index = data["user_id"]

    df = pd.read_csv(f"../data/{mode}.csv")

    df["from"] = df["from-to"].apply(lambda x: int(x.split("-")[0]))
    df["to"] = df["from-to"].apply(lambda x: int(x.split("-")[1]))

    df = df.drop(columns=["from-to"])

    # userのdataを結合
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

    print(df.info())

    df.to_pickle(f"../data/{mode}_v{VERSION}.pkl")
