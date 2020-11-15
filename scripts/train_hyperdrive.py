# -*- coding: utf-8 -*-
import lightgbm as lgb
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run

## Add arguments to script
parser = argparse.ArgumentParser(description='Cardio Train')
parser.add_argument('--data_folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--num_leaves', type=int, default=32, help='Minimun number of leaves')
parser.add_argument('--max_depth', type=int, default=3, help='Maximum tree depth')
parser.add_argument('--min_data_in_leaf', type=int, default=32, help='Minimun amount of data in the leafs')
parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for training')
args = parser.parse_args()

print("Split the data into train and test")
run = Run.get_context()
output_split_train = run.input_datasets['output_split_train']
df = output_split_train.to_pandas_dataframe()

X_train, X_valid, y_train, y_valid = train_test_split(df.drop('cardio',axis=1),df.cardio,test_size=0.20, random_state=42)

## Model
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': { 'AUC' },
    'num_leaves': args.num_leaves,
    'max_depth': args.max_depth,
    'min_data_in_leaf': args.min_data_in_leaf,
    'bagging_freq': 1,
    'feature_fraction': 0.7,
    'verbose': 1,
    'is_unbalance':True,
    'learning_rate': args.learning_rate,
    'bagging_fraction': 0.9,
}

train_set = lgb.Dataset(X_train, y_train)
validation_sets = lgb.Dataset(X_valid, y_valid, reference=train_set)

model_lgbm = lgb.train(
    params,
    train_set,
    num_boost_round=10000,
    valid_sets=validation_sets,
    early_stopping_rounds=500,
    verbose_eval=False
    )

y_pred_lgbm = model_lgbm.predict(X_valid)
print('LGBM– ' + str(roc_auc_score(y_valid, y_pred_lgbm)))

# Log metrics
run = Run.get_context()
run.log("num_leaves:", np.float(args.num_leaves))
run.log("max_depth:", np.int(args.max_depth))
run.log("min_data_in_leaf:", np.int(args.min_data_in_leaf))
run.log("AUC", roc_auc_score(y_valid, y_pred_lgbm))
