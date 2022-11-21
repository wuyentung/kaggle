#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import plotly.graph_objects as go
import networkx as nx 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import Normalizer

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
## skill importance for each job cat: proportion of each skill type occurance for all skills
job_skill_sum = skill_df.groupby(JOB_CAT).sum()
job_skill_importance = job_skill_sum.copy()
for col in job_skill_importance.columns:
       job_skill_importance[col] /= job_skill_sum.T.sum()
#%%
# job_skill_types = skill_df.groupby(JOB_CAT).mean()
#%%
def radar_plot(df:pd.DataFrame, row:str):
       fig = go.Figure(data=go.Scatterpolar(
              r=df.loc[row],
              theta=df.columns,
              fill='toself'
              ))

       fig.update_layout(
              polar=dict(radialaxis=dict(visible=True),), 
              showlegend=False
              )

       fig.show()
       pass
radar_plot(df=job_skill_importance, row=job_skill_importance.index[0])
#%%
for row in job_skill_importance.index:
       radar_plot(df=job_skill_importance, row=row)
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
       pass
radar_plot_compare(df=job_skill_importance)
#%%
## normorlizing each skill type by each max should be more understanderable when comparing job cat
job_skill_importance_norm = job_skill_importance.copy()
for col in job_skill_importance_norm.columns:
       job_skill_importance_norm[col] /= job_skill_importance_norm[col].max()
radar_plot_compare(df=job_skill_importance_norm)
#%%
## TODO: 能力關係圖
#%%
## calculate confidence(A->B)= P(A^B)/P(A)
job_cats = job_skill_sum.index.to_list()
## column-wise: for each job cat, [col][rol]: confidence(col->rol)
def make_confidence_df(confidence_dict:dict):
       d = {i:[confidence_dict[(j, i)] for j in skill_types.keys()] for i in skill_types.keys()}  
       return pd.DataFrame.from_dict(data=d, orient='index', columns=skill_types.keys())
def cal_job_confidence(df:pd.DataFrame, job:str)->dict:
       confidence_dict = {}
       new_df = df.loc[df[JOB_CAT] == job].copy()
       for i in skill_types.keys():
              for j in skill_types.keys():
                     if i == j:
                            confidence_dict[(i, j)] = np.nan
                            continue
                     confidence_dict[(i, j)] = new_df[[i, j]].T.min().sum()/job_skill_sum[i][job]
       return confidence_dict
job_confidence_dicts = {job:make_confidence_df(cal_job_confidence(df=skill_df, job=job)) for job in job_cats}
#%%
## then the job_confidence_dicts stores confidence(A->B) by jobs, and we should remains the maximize of confidence(A->B) and confidence(B->A) when we visualizing it, i.e., the weight between skills for each job cat
def find_edge_weight(df:pd.DataFrame, i:str, j:str):
       return (i, j, df[i][j]) if df[i][j] > df[j][i] else (j, i, df[j][i])
## since the edges for n nodes is n(n+1)/2, we pick two edges to represent each node
def top_2_node_edge(edge_weight_list:list):
       ## remove duplicates
       edge_weight_list = list(dict.fromkeys(edge_weight_list))
       
       top_2_edge_weight_list = []
       for skill_type in skill_types.keys():
              largest, sec_largest = (0, 0, 0), (0, 0, 0)
              for edge_weight in edge_weight_list:
                     if skill_type in edge_weight[0] or skill_type in edge_weight[1]:
                            # print(edge_weight)
                            if edge_weight[2] > sec_largest[2]:
                                   if edge_weight[2] > largest[2]:
                                          largest = edge_weight
                                   else:
                                          sec_largest = edge_weight
              top_2_edge_weight_list.append(largest)
              top_2_edge_weight_list.append(sec_largest)
              # print(largest)
              # print(sec_largest)
       return top_2_edge_weight_list
#%%
def job_skill_graph(job:str):
       G = nx.DiGraph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
       df = job_confidence_dicts[job]
       G.add_weighted_edges_from(top_2_node_edge([find_edge_weight(df=df, i=i, j=j) for i in skill_types.keys() for j in skill_types.keys()]))
       pos = nx.circular_layout(G)
       edge_width = [10*G.get_edge_data(u, v)['weight'] for u, v in G.edges()]
       node_size = [40000*job_skill_importance[skill_type][job] for skill_type in skill_types.keys()]

       plt.figure(figsize=(20, 16))
       nx.draw_networkx_nodes(G, pos, node_size = node_size, node_color='lightblue', alpha=.7)
       nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=50, edge_color='grey', arrowstyle="-|>", width = edge_width, style='-', connectionstyle='arc3, rad = .03')
       nx.draw_networkx_labels(G, pos, font_size=30)
       plt.title(f'{job} Skills', size = 50)
       plt.show()
job_skill_graph(job_cats[0])
#%%
## TODO: 薪水級距預測
## feature engineering
df_modeling = pd.DataFrame(X.toarray(), columns=skills_list)
df_modeling[[JOB_CAT, SIZE, SALARY_LOWER, SALARY_UPPER]] = df_merge[[JOB_CAT, SIZE, SALARY_LOWER, SALARY_UPPER]]
df_modeling[JUNIOR]  = [1 if _ else 0 for _ in df_merge[JUNIOR]]
df_modeling[MID]  = [1 if _ else 0 for _ in df_merge[MID]]
df_modeling[SENIOR]  = [1 if _ else 0 for _ in df_merge[SENIOR]]
df_modeling = df_modeling.loc[df_modeling[SIZE].str.contains('employees').fillna(False)]
## one-hot encoding
df_modeling = pd.get_dummies(df_modeling, prefix=[JOB_CAT, SIZE], drop_first=True)
#%%
## train-test split
X_train, X_test, y_train, y_test = train_test_split(df_modeling.drop(columns=[SALARY_LOWER, SALARY_UPPER]), df_modeling[[SALARY_LOWER, SALARY_UPPER]], test_size=0.2, random_state=0)
#%%
## since most of the features are range in [0, 1], we normalize the y to get a better mapping, and evaluate in MAPE
transformer = Normalizer().fit(y_train)
y_train = pd.DataFrame(transformer.transform(y_train), columns=y_train.columns)
y_test = pd.DataFrame(transformer.transform(y_test), columns=y_train.columns)
#%%
## predict with XGBoost (two models)
model_lower = GradientBoostingRegressor(random_state=0, warm_start=True, n_estimators=200)
model_lower.fit(X_train, y_train[SALARY_LOWER])
mape_lower = mean_absolute_percentage_error(model_lower.predict(X_test), y_test[SALARY_LOWER])
#%%
upper_model = GradientBoostingRegressor(random_state=0, warm_start=True, n_estimators=200)
upper_model.fit(X_train, y_train[SALARY_UPPER])
mape_upper = mean_absolute_percentage_error(upper_model.predict(X_test), y_test[SALARY_UPPER])
#%%
def feature_importances_transform(model:GradientBoostingRegressor, n:int=10):
       feature_importances_df = pd.DataFrame.from_dict({'Feature': X_train.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
       return feature_importances_df.head(n)
feature_importances_transform(model=model_lower)
#%%
feature_importances_transform(model=upper_model)
#%%
