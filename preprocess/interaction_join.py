import pandas as pd

base_data = pd.read_csv("../data/user_ages.csv")

# data1 = pd.read_csv("../data/interaction_swipes.csv")
# data2 = pd.read_csv("../data/interaction_review_strengths.csv")
data1 = pd.read_pickle("../data/data_1.pkl")
data2 = pd.read_pickle("../data/data_2.pkl")


def from_to(df):
    df["from"] = df["from-to"].apply(lambda x: int(x.split("-")[0]))
    df["to"] = df["from-to"].apply(lambda x: int(x.split("-")[1]))

    return df.drop(columns="from-to")


def swipes(data, swipe):
    swipe.loc[swipe["swipe_status"] == -1] = 0
    # df = pd.DataFrame(swipe["from"].unique())
    df_list = list()
    for mode in ["from", "to"]:
        print(mode)
        memo = swipe.groupby(mode)["swipe_status"]
        df = pd.DataFrame(memo.count().index.tolist(), columns=["user_id"])
        df[f"1_{mode}_count"] = memo.count().values
        df[f"1_{mode}_1_sum"] = memo.sum().values
        df[f"1_{mode}_0_sum"] = df[f"1_{mode}_count"] - df[f"1_{mode}_1_sum"]
        df[f"1_{mode}_1_ratio"] = df[f"1_{mode}_0_sum"] / df[f"1_{mode}_count"]
        df_list.append(df)

    data = data.merge(df_list[0], on="user_id", how="left")
    data = data.merge(df_list[0], on="user_id", how="left")

    return data.drop(columns="age")


# data1 = from_to(data1)
# data1.to_pickle("../data/data_1.pkl")
# data2 = from_to(data2)
# data2.to_pickle("../data/data_2.pkl")

data1 = swipes(base_data, data1)
print()

data1.to_pickle("../data/data_1_v1.pkl")