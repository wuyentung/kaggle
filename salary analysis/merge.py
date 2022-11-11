#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
COLUMNS = ['Job Title', 'Salary Estimate', 'Job Description',
       'Rating', 'Company Name', 'Location', 'Headquarters', 'Size', 'Founded',
       'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Competitors',
       'Easy Apply']
#%%
df_label = 'df label'
#%%
df1 = pd.read_csv('BusinessAnalyst.csv', )
df1 = df1[COLUMNS]
df1[df_label] = 1
#%%
df2 = pd.read_csv('DataAnalyst.csv', )
df2 = df2[COLUMNS]
df2[df_label] = 2
#%%
df3 = pd.read_csv('DataEngineer.csv',)
df3 = df3[COLUMNS]
df3[df_label] = 3
#%%
df4 = pd.read_csv('DataScientist.csv', )
df4 = df4[COLUMNS]
df4[df_label] = 4
#%%
df_merge = pd.concat([df1, df2, df3, df4])
#%%
df_merge.dropna(inplace=True)
#%%
s1 = df_merge['Salary Estimate'].str.split('$', expand=True)
#%%
temp = s1[1].str.split('K', expand=True)
lower = temp[0].str.split('-', expand=True).copy()
#%%
temp = s1[2].str.split('K', expand=True)
upper = temp[0].str.split('-', expand=True).copy()
#%%
df_merge['Salary Lower'] = lower[0].astype('int', errors='ignore')
df_merge['Salary Lower'] = pd.to_numeric(df_merge['Salary Lower'])
#%%
df_merge['Salary Upper'] = upper[0].astype('int', errors='ignore')
#%%
np.mean(df_merge['Salary Lower'])
#%%
df_merge['Salary Lower'].astype('int')
#%%
df_merge['Salary Lower'].unique()
#%%
#%%
df_merge['Rating'] = pd.to_numeric(df_merge['Rating'])
df_merge.groupby(df_label).mean('Rating')
#%%
