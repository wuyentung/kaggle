#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#%%
JOB_CAT = 'Job Category'
COLUMNS = [JOB_CAT, 'Job Title', 'Salary Estimate', 'Job Description', 'Rating', 'Company Name', 'Location', 'Headquarters', 'Size', 'Founded', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Competitors', 'Easy Apply']
#%%
PATH = 'data'
data_dir = os.listdir(PATH)
#%%
def create_l0_df(file_name) -> pd.DataFrame:
       def nothing():
              pass
       df = pd.read_csv(os.path.join(PATH, file_name))
       df[JOB_CAT] = file_name[:-4]
       return df[COLUMNS]
#%%
dfs = [create_l0_df(file_name=file_name) for file_name in data_dir]
#%%
df_merge = pd.concat(dfs)
df_merge.dropna(inplace=True)
#%%
s1 = df_merge['Salary Estimate'].str.split('$', expand=True)
#%%
temp = s1[1].str.split('K', expand=True)
lower = temp[0].str.split('-', expand=True).copy()
#%%
temp = s1[2].str.split('K', expand=True)
upper = temp[0].str.split(' ', expand=True).copy()
#%%
df_merge['Salary Lower'] = lower[0].astype('int', errors='ignore')
df_merge['Salary Lower'] = pd.to_numeric(df_merge['Salary Lower'])
#%%
df_merge['Salary Upper'] = upper[0].astype('int', errors='ignore')
df_merge['Salary Upper'] = pd.to_numeric(df_merge['Salary Upper'])
#%%
df_merge.groupby(JOB_CAT).mean(['Salary Lower', 'Salary Upper'])
#%%
