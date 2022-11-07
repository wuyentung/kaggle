#%%
# The dataset is from [**Classifying with Logistic Regression**](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data?select=train.csv)
# The analyze flow is inspired by [Simple Logistic Regression for Good Score (0.5837)](https://www.kaggle.com/code/ryanluoli2/simple-logistic-regression-for-good-score-0-5837)
# The mlflow architecture is from https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
# CV example: https://github.com/mlflow/mlflow/blob/master/examples/sklearn_autolog/grid_search_cv.py
#%%
import os
import warnings
import sys

import pandas as pd
import numpy as np
from regex import L
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from pprint import pprint

import logging
from constant import *
from preprocessing import X_train, y_train, df_test
from utils import fetch_logged_data

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
#%%
if __name__ == "__main__":
    
    mlflow.sklearn.autolog()
    
    ## dataset complete
    X = X_train.copy()
    y = y_train.copy()

    Cs = [5, 10]
    l1_ratios = [0, 0.5, 1]
    # Cs = [float(sys.argv[1])] if len(sys.argv) > 1 else [10]
    # l1_ratios = [float(sys.argv[2])] if len(sys.argv) > 2 else [0, 0.5, 1]
    
    parameters = {'C': Cs, 'l1_ratio': l1_ratios}
    
    lr = LogisticRegression(max_iter=500, penalty='elasticnet', random_state=0, solver='saga', n_jobs=4)
    clf = GridSearchCV(lr, parameters)
    clf.fit(X, y)
    run_id = mlflow.last_active_run().info.run_id

    # show data logged in the parent run
    print("========== parent run ==========")
    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)

    # show data logged in the child runs
    filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run_id)
    runs = mlflow.search_runs(filter_string=filter_child_runs)
    param_cols = ["params.{}".format(p) for p in parameters.keys()]
    metric_cols = ["metrics.mean_test_score"] 
    
    

    print("\n========== child runs ==========\n")
    pd.set_option("display.max_columns", None)  # prevent truncating columns
    print(runs[["run_id", *param_cols, *metric_cols]])
    
    
    ## submission
    # Disable autologging after model selection is complete.
    mlflow.sklearn.autolog(disable=True)
    lr_best = LogisticRegression(max_iter=500, penalty='elasticnet', random_state=0, solver='saga', n_jobs=4, C = clf.best_params_['C'], l1_ratio=clf.best_params_['l1_ratio'])

    lr_best.fit(X_train, y_train)

    y_pred = lr_best.predict_proba(df_test.drop([ID, FAILURE], axis=1))
    submission = pd.read_csv('data/sample_submission.csv')
    submission[FAILURE] = y_pred[:,1]
    submission.to_csv("submission_baseline.csv", index=False)
#%%
