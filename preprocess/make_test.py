import pandas as pd

test = pd.read_csv("../data/test.csv")
data = pd.read_pickle("../data/user_agg_v3.pkl")
data.index = data["user_id"]
print(test.shape)

test["from"] = test["from-to"].apply(lambda x: int(x.split("-")[0]))
test["to"] = test["from-to"].apply(lambda x: int(x.split("-")[1]))

test = test.drop(columns=["from-to"])

data["from"] = data["user_id"].copy()
data = data.drop(columns="user_id")
test = pd.merge(test, data, on="from", how="left")
data["to"] = data["from"].copy()

data = data.drop(columns="from")
test = pd.merge(test, data, on="to", how="left")

test = test.drop(columns=["from", "to"])

print(test.info())

test.to_pickle("../data/test_v1.pkl")
