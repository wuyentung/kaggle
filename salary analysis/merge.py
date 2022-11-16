#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.text import TfidfVectorizer

from constant import *
#%%
columns = [JOB_CAT, JOB_TITLE, SALARY_ESTIMATE, JOB_DES, RATING, CO_NAME, LOCATION, HEADQUARTERS, SIZE, FOUNDED, OWNERSHIP_TYPE, INDUSTRY, SECTOR, REVENUE, COMPETITORS, EASY_APPLY]
#%%
PATH = 'data'
data_dir = os.listdir(PATH)
#%%
def create_l0_df(file_name) -> pd.DataFrame:
       def nothing():
              pass
       df = pd.read_csv(os.path.join(PATH, file_name))
       df[JOB_CAT] = file_name[:-4]
       return df[columns]
#%%
dfs = [create_l0_df(file_name=file_name) for file_name in data_dir]
#%%
df_merge = pd.concat(dfs, ignore_index=True)
#%%
s1 = df_merge[SALARY_ESTIMATE].str.split('$', expand=True)
#%%
temp = s1[1].str.split('K', expand=True)
lower = temp[0].str.split('-', expand=True).copy()
#%%
temp = s1[2].str.split('K', expand=True)
upper = temp[0].str.split(' ', expand=True).copy()
#%%
df_merge[SALARY_LOWER] = lower[0]
df_merge[SALARY_UPPER] = upper[0]
#%%
df_merge[SALARY_LOWER] = pd.to_numeric(df_merge[SALARY_LOWER])
df_merge[SALARY_UPPER] = pd.to_numeric(df_merge[SALARY_UPPER])
#%%
df_merge.loc[(df_merge[SALARY_ESTIMATE].str.contains('Hour')), SALARY_LOWER] = df_merge.loc[(df_merge[SALARY_ESTIMATE].str.contains('Hour')), SALARY_LOWER]*40*52/1000
df_merge.loc[(df_merge[SALARY_ESTIMATE].str.contains('Hour')), SALARY_UPPER] = df_merge.loc[(df_merge[SALARY_ESTIMATE].str.contains('Hour')), SALARY_UPPER]*40*52/1000
#%%
df_merge.groupby(JOB_CAT).mean([SALARY_LOWER, SALARY_UPPER])
#%%
df_misplace = df_merge.loc[df_merge[SALARY_UPPER].isna()].copy()
df_merge.drop(index=df_misplace.index, inplace=True)
#%%
df_merge[JOB_TITLE].unique()
#%%
## TODO: one-hot for level of job
## ref: https://www.kaggle.com/code/josephgutstadt/skills-for-a-data-scientist-analyst/notebook

## one-hot for job level
df_merge[SENIOR] = df_merge[JOB_TITLE].str.contains('enior')
df_merge[MID] = df_merge[JOB_TITLE].str.contains('Mid ') | df_merge[JOB_TITLE].str.contains('Mid-') | df_merge[JOB_TITLE].str.contains(' mid ')
df_merge[JUNIOR] = df_merge[JOB_TITLE].str.contains('unior')
## label amount is quite less, should use KNN or just predict salary range from skills
#%%
skills_list = {skill for skill_set in skill_types.values() for skill in skill_set}
#%%
vectorizer = TfidfVectorizer(stop_words=stopward_list, vocabulary=skills_list)
X = vectorizer.fit_transform(df_merge[JOB_DES])
vectorizer.get_feature_names_out()
#%%
