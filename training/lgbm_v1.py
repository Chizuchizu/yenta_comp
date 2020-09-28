import lightgbm as lgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc


N_FOLDS = 4
VERSION = 4
DEBUG = False
NUM_CLASSES = 4
SEED = 22
num_rounds = 10 if DEBUG else 1000

params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': NUM_CLASSES,
    'learning_rate': 0.15,
    'max_depth': 7,
    'num_leaves': 31,
    'max_bin': 31,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'nthread': -1,
    'bagging_freq': 1,
    'verbose': -1,
    'seed': SEED,
}

train = pd.read_pickle(f"../data/train_v{VERSION}.pkl")
test = pd.read_pickle(f"../data/test_v{VERSION}.pkl")
target = train["score"]
train = train.drop(columns="score")
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
pred = np.zeros((test.shape[0], NUM_CLASSES))
score = 0
for fold, (train_idx, valid_idx) in enumerate(kfold.split(train, target)):
    x_train, x_valid = train.loc[train_idx], train.loc[valid_idx]
    y_train, y_valid = target[train_idx], target[valid_idx]

    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_valid, label=y_valid)
    del x_train
    del x_valid
    del y_train
    del y_valid
    gc.collect()

    estimator = lgb.train(
        params=params,
        train_set=d_train,
        num_boost_round=num_rounds,
        valid_sets=[d_train, d_valid],
        verbose_eval=100,
        early_stopping_rounds=100
    )

    y_pred = estimator.predict(test)
    pred += y_pred / N_FOLDS

    print(fold + 1, "done")

    score += estimator.best_score["valid_1"]["multi_logloss"] / N_FOLDS
    lgb.plot_importance(estimator, importance_type="gain", max_num_features=25)
    plt.show()

if not DEBUG:
    ss = pd.read_csv("../data/test.csv")
    ss["score"] = np.argmax(pred, axis=1).astype(float)
    ss.to_csv(f"../outputs/lgbm_v{VERSION}_{round(score, 4)}.csv", index=False)
