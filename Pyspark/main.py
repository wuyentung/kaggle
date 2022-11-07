#%%
## this work is inspired by https://www.kaggle.com/code/fatmakursun/pyspark-ml-tutorial-for-beginners/notebook
#%%
import os
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

import seaborn as sns
import matplotlib.pyplot as plt
# Visualization
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)

from matplotlib import rcParams
sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (18,4)})
rcParams['figure.figsize'] = 18,4

from constant import *

# setting random seed for notebook reproducability
rnd_seed = 0
np.random.seed = rnd_seed
np.random.set_state = rnd_seed
#%%
spark = SparkSession.builder.master("local[2]").appName("Linear-Regression-California-Housing").getOrCreate()
#%%
# define the schema, corresponding to a line in the csv data file.
schema = StructType([
    StructField(LONGTITUDE, FloatType(), nullable=True),
    StructField(LATITUDE, FloatType(), nullable=True),
    StructField(MED_AGE, FloatType(), nullable=True),
    StructField(TOT_ROOMS, FloatType(), nullable=True),
    StructField(TOT_BDRMS, FloatType(), nullable=True),
    StructField(POPULATION, FloatType(), nullable=True),
    StructField(HOUSEHOLDS, FloatType(), nullable=True),
    StructField(MED_INC, FloatType(), nullable=True),
    StructField(MED_HV, FloatType(), nullable=True)]
)
# Load housing data
housing_df = spark.read.csv(path='data/cal_housing.data', schema=schema).cache()
#%%
med_age_count = housing_df.groupBy(MED_AGE).count().sort(MED_AGE, ascending=False)
med_age_count.show(10)
#%%
med_age_count.toPandas().plot.bar(x=MED_AGE,figsize=(14, 6))
#%%
housing_df.describe().select(["summary"]+ [F.round(c, 4).alias(c) for c in housing_df.columns]).toPandas()
#%%
housing_df.select([F.count(F.when(F.isnan(c) | col(c).isNull(), c)).alias(c) for c in housing_df.columns]).toPandas()
## there is no missing value
#%%
housing_df = housing_df.withColumn(MED_HV, col(MED_HV)/100000)
#%%
# Add the new columns to `df`
housing_df = housing_df.withColumn("rmsperhh", F.round(col(TOT_ROOMS)/col(HOUSEHOLDS), 2)).withColumn("popperhh", F.round(col(POPULATION)/col(HOUSEHOLDS), 2)).withColumn("bdrmsperrm", F.round(col(TOT_BDRMS)/col(TOT_ROOMS), 2))
#%%
# Re-order and select columns
housing_df = housing_df.select(
    MED_HV, 
    TOT_BDRMS, 
    POPULATION, 
    HOUSEHOLDS, 
    MED_INC, 
    "rmsperhh", 
    "popperhh", 
    "bdrmsperrm")
#%%
## Feature Extraction
feature_cols = [TOT_BDRMS, POPULATION, HOUSEHOLDS, MED_INC, "rmsperhh", "popperhh", "bdrmsperrm"]
#%%
# put features into a feature vector df
assembled_df = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(housing_df)
#%%
# Initialize the `standardScaler` and fit the DataFrame to the scaler
scaled_df = StandardScaler(inputCol="features", outputCol="features_scaled").fit(assembled_df).transform(assembled_df)
#%%
# Inspect the result
scaled_df.select("features", "features_scaled").show(10, truncate=True)
#%%
# Split the data into train and test sets
train_data, test_data = scaled_df.randomSplit([.8, .2], seed=rnd_seed)
#%%
# Initialize `lr`
lr = (LinearRegression(featuresCol='features_scaled', labelCol=MED_HV, predictionCol='predmedhv', maxIter=10, regParam=0.3, elasticNetParam=0.8, standardization=False))
#%%
# Fit the data to the model
linearModel = lr.fit(train_data)
#%%
## evaluate model
# Coefficients for the model
linearModel.coefficients
#%%
# Intercept for the model
linearModel.intercept
#%%
coeff_df = pd.DataFrame({"Feature": ["Intercept"] + feature_cols, "Co-efficients": np.insert(linearModel.coefficients.toArray(), 0, linearModel.intercept)})
coeff_df = coeff_df[["Feature", "Co-efficients"]]
#%%
# Generate predictions and extract the predictions and the "known" correct labels
predandlabels = linearModel.transform(test_data).select("predmedhv", MED_HV)
#%%
# Get the RMSE
print("RMSE: {0}".format(linearModel.summary.rootMeanSquaredError))
print("MAE: {0}".format(linearModel.summary.meanAbsoluteError))
# Get the R2
print("R2: {0}".format(linearModel.summary.r2))
#%%
spark.stop()
#%%
