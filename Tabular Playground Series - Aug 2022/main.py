#%%
# The dataset is from [**Classifying with Logistic Regression**](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data?select=train.csv)
# The analyze flow is inspired by [Simple Logistic Regression for Good Score (0.5837)](https://www.kaggle.com/code/ryanluoli2/simple-logistic-regression-for-good-score-0-5837)
# The mlflow architecture is from https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
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
from sklearn.model_selection import cross_validate
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging
from constant import *
from preprocessing import X_train, y_train, df_test

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
#%%
if __name__ == "__main__":
    
    
    ## dataset complete
    X = X_train.copy()
    y = y_train.copy()

    # Cs = [10]
    # l1_ratios = [0.5]
    Cs = [float(sys.argv[1])] if len(sys.argv) > 1 else [10]
    l1_ratios = [float(sys.argv[2])] if len(sys.argv) > 2 else [0.5]
    
    for c in Cs:
        for l1_ratio in l1_ratios:
            with mlflow.start_run():
                lr = LogisticRegression(C=c, max_iter=500, penalty='elasticnet', l1_ratio=l1_ratio, random_state=0, solver='saga', n_jobs=4)
                lr.fit(X, y)
                y_pred = lr.predict_proba(df_test.drop([ID, FAILURE], axis=1))
                scores = cross_validate(lr, X, y, scoring=[ACCURACY, RECALL, F1, ROC_AUC], cv=10)
                
                accuracy = np.median(scores[f'test_{ACCURACY}'])
                recall = np.median(scores[f'test_{RECALL}'])
                f1 = np.median(scores[f'test_{F1}'])
                roc_auc = np.median(scores[f'test_{ROC_AUC}'])
                
                print("median loss of elasticnet LogisticRegression model (C=%f, l1_ratio=%f):" % (c, l1_ratio))
                print("  %s: %s" % (ACCURACY, accuracy))
                print("  %s: %s" % (RECALL, recall))
                print("  %s: %s" % (F1, f1))
                print("  %s: %s" % (ROC_AUC, roc_auc))

                mlflow.log_param("C", c)
                mlflow.log_param("l1_ratio", l1_ratio)
                mlflow.log_metric(ACCURACY, accuracy)
                mlflow.log_metric(RECALL, recall)
                mlflow.log_metric(F1, f1)
                mlflow.log_metric(ROC_AUC, roc_auc)

                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(lr, "model", registered_model_name="elasticnet LogisticRegression Aug 22 model")
                else:
                    mlflow.sklearn.log_model(lr, "model")
    submission = pd.read_csv('data/sample_submission.csv')
    submission[FAILURE] = y_pred[:,1]
    submission.to_csv("submission_baseline.csv", index=False)
#%%
