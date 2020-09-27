from functools import wraps
import time
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

data_list = ["user_educations", "user_works", "user_skills", "user_strengths", "user_purposes",
             "user_self_intro_vectors_300dims", "user_sessions"]

data = pd.read_csv("../data/user_ages.csv")


def stop_watch(func_):
    @wraps(func_)
    def wrapper(*args, **kargs):
        # 処理開始直前の時間
        start = time.time()

        # 処理実行
        result = func_(*args, **kargs)

        # 処理終了直後の時間から処理時間を算出
        elapsed_time = time.time() - start

        # 処理時間を出力
        print("{} ms in {}".format(elapsed_time * 1000, func_.__name__))
        return result

    return wrapper


@stop_watch
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


@stop_watch
def educations(new_df):
    new_df = new_df.drop(columns="degree_id")
    # new_df["id"] = new_df.index.copy()
    new_df["values"] = 1
    new_df.loc[new_df["school_id"] == 999999] = np.nan
    new_df = new_df.pivot_table(index="user_id", columns="school_id", values="values", fill_value=0).astype("bool")
    new_df.index = new_df.index.astype(int)
    new_df.columns = new_df.columns.astype(int)

    pca = PCA(n_components=100)
    pca_data = pd.DataFrame(pca.fit_transform(new_df))
    new_df = pd.DataFrame(new_df.index)
    new_df = pd.concat([new_df, pca_data], axis=1)

    # print(new_df.info(memory_usage="deep"))
    # 49.2MB
    return new_df


@stop_watch
def works(new_df):
    new_df = new_df.drop(columns=["over_1000_employees", "industry_id"])
    new_df["values"] = 1
    new_df.loc[new_df["company_id"] == 999999] = np.nan
    new_df = new_df.pivot_table(index="user_id", columns="company_id", values="values", fill_value=0).astype("bool")
    new_df.index = new_df.index.astype(int)
    new_df.columns = new_df.columns.astype(int)

    pca = PCA(n_components=100)
    pca_data = pd.DataFrame(pca.fit_transform(new_df))
    new_df = pd.DataFrame(new_df.index)
    new_df = pd.concat([new_df, pca_data], axis=1)

    # 75.9MB
    print("hello")
    return new_df


@stop_watch
def skills(new_df):
    new_df["values"] = 1
    new_df.loc[new_df["skill_id"] == 999999] = np.nan
    new_df = new_df.pivot_table(index="user_id", columns="skill_id", values="values", fill_value=0).astype("bool")
    new_df.index = new_df.index.astype(int)
    new_df.columns = new_df.columns.astype(int)

    pca = PCA(n_components=100)
    pca_data = pd.DataFrame(pca.fit_transform(new_df))
    new_df = pd.DataFrame(new_df.index)
    new_df = pd.concat([new_df, pca_data], axis=1)

    return new_df


def strengths(new_df):
    return new_df


def purposes(new_df):
    return new_df


@stop_watch
def sessions(new_df):
    df = pd.DataFrame(new_df["user_id"].unique(), columns=["user_id"])
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
    memo = new_df.groupby("user_id")["timestamp"]
    del new_df

    df["range"] = (memo.max() - memo.min()).dt.days.reset_index()["timestamp"]
    df["count"] = memo.count().reset_index().reset_index()["timestamp"]
    df["often"] = df["range"] / df["count"]
    return df


def intro(new_df):
    return new_df


data_dict = {
    "user_educations": educations,
    "user_works": works,
    "user_skills": skills,
    "user_strengths": strengths,
    "user_purposes": purposes,
    "user_sessions": sessions,
    "user_self_intro_vectors_300dims": intro
}
debug = 2
for i, (x, func) in enumerate(data_dict.items()):
    print(x)
    # if i != debug:
    #     continue
    output = func(pd.read_csv(f"../data/{x}.csv"))
    print()
    print(output.info())
    data = pd.merge(data, output, on="user_id", how="left")
    print("---------------------------------------------")

print(data.info())
# data = reduce_mem_usage(data)
data.to_pickle("../data/user_agg_v2.pkl")
