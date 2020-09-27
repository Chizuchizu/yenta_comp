import pandas as pd

train = pd.read_csv("../data/train.csv")
data = pd.read_pickle("../data/user_agg_v3.pkl")
data.index = data["user_id"]
target = train["score"]
train["from"] = train["from-to"].apply(lambda x: int(x.split("-")[0]))
train["to"] = train["from-to"].apply(lambda x: int(x.split("-")[1]))

train = train.drop(columns=["score", "from-to"])

data["from"] = data["user_id"].copy()
data = data.drop(columns="user_id")
train = pd.merge(train, data, on="from", how="left")
data["to"] = data["from"].copy()

data = data.drop(columns="from")
train = pd.merge(train, data, on="to", how="left")

train = train.drop(columns=["from", "to"])

print(train.info())

train.to_pickle("../data/train_v1.pkl")
