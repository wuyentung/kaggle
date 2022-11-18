#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import plotly.graph_objects as go

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
## term frequency document-term matrix
tf_matrix = X.toarray() * vectorizer.idf_
#%%
## occurance document-term matrix
occurance_matrix = np.array([[True if value else False for value in row] for row in tf_matrix])
occurance_df = pd.DataFrame(occurance_matrix, columns=skills_list)
#%%
## since we have quanty information of each skill types, we use probability instead of association rule
# [1, 0, 1, 1], [0, 0, 1], [1, 0, 1, 0, 0, 0], [1, 1]
# [1, 0, 0, 1], [0, 1, 1], [0, 0, 1, 1, 1, 1], [0, 1]
# [1, 0, 0, 0], [1, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0]

## skill counts among skill types: for each type, sum occurance, sum all descriptions
#%%
skill_df = pd.DataFrame(np.array([occurance_df[skill_types[key]].T.sum() for key in skill_types.keys()]).T, columns=skill_types.keys())
#%%
## skill type summary needed in job describtion
plt.figure(figsize=(6, 8))
plt.pie(skill_df.sum(), labels=skill_df.columns, colors=sns.color_palette('bright'), autopct='%.0f%%')
plt.title('skill type summary needed in job describtion', fontsize=10)
plt.show()
#%%
skill_df[JOB_CAT] = df_merge[JOB_CAT]
#%%
job_skill_types = skill_df.groupby(JOB_CAT).mean()
#%%
def radar_plot(df:pd.DataFrame, row:str):
       fig = go.Figure(data=go.Scatterpolar(
       r=df.loc[row],
       theta=df.columns,
       fill='toself'
       ))

       fig.update_layout(
       polar=dict(
       radialaxis=dict(
       visible=True
       ),
       ),
       showlegend=False
       )

       fig.show()
radar_plot(df=job_skill_types, row=_)
#%%
for row in job_skill_types.index:
       radar_plot(df=job_skill_types, row=row)
#%%
def radar_plot_compare(df:pd.DataFrame):
       fig = go.Figure()

       for row in df.index:
              fig.add_trace(go.Scatterpolar(r=df.loc[row], theta=df.columns, fill='toself', name=row))

       fig.update_layout(
              polar=dict(radialaxis=dict(visible=True,range=[0, df.max().max()*1.1])),
              showlegend=False
              )

       fig.show()
radar_plot_compare(df=job_skill_types)
#%%
## normorlizing each skill type by each max should be more understanderable when comparing job cat
job_skill_types_norm = job_skill_types.copy()
for col in job_skill_types_norm.columns:
       job_skill_types_norm[col] /= job_skill_types_norm[col].max()
radar_plot_compare(df=job_skill_types_norm)
#%%
