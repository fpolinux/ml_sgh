## wrapper based approach for feature selection
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# %% Defining data and target variables
data = load_breast_cancer(as_frame=True) # Frame makes it load nicer
X = data.data
y = data.target
# %% Setup a model
model = LogisticRegression()
sfs = SFS(model, 
          k_features=30, #Stop at 30 features
          forward=True, #Forward regression
          floating=False, #Adds/removes features to see if the score can be improved
          scoring='accuracy', #Scoring method, could be r2 etc.
          cv=5)  
#Cross validation, dataset is split into n equal parts, 
# for each part, n-1 parts are used for training, 1 is used for testing.
# the model then iterates over this n times and the next part 
# is used as the testing blocl
sfs.fit(X,y)
# %% Validate model
accuracy = sfs.k_score_ #output of model just ran
sfs_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
fig, ax = plt.subplots()
sfs_df.plot(kind='line', y='avg_score', ax=ax)
# %%
# %% A second model using backward regression
model = LogisticRegression()
sbs = SFS(model, 
          k_features=1, 
          forward=False, #Backward regression; start with all features and remove one by one
          floating=False, #Adds/removes features to see if the score can be improved
          scoring='accuracy', #Scoring method, could be r2 etc.
          cv=5)  
sbs.fit(X,y)
#%% Test the backwaes model accuracy
accuracy = sbs.k_score_ #output of model just ran
sbs_df = pd.DataFrame.from_dict(sbs.get_metric_dict()).T
fig, ax = plt.subplots()
sbs_df.plot(kind='line', y='avg_score', ax=ax)
# %% Best subset selection/exhaustive selection
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
model = LogisticRegression()
## In an ideal world, we would use 1-30 for features but this would take a long time.
efs = EFS(model, 
          min_features=9, 
          max_features=10,
          scoring='accuracy', #Scoring method, could be r2 etc.
          cv=5)  
efs.fit(X,y)
#%% Test the backwaes model accuracy
accuracy = efs.k_score_ #output of model just ran
efs_df = pd.DataFrame.from_dict(efs.get_metric_dict()).T
fig, ax = plt.subplots()
efs_df.plot(kind='line', y='avg_score', ax=ax)
# %%