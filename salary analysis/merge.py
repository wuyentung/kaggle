#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

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
