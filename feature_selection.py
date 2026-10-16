#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import mutual_info_regression
#%% Data Import and define
data = load_diabetes()
X = data.data
y = data.target
#%% Feature setup and sorting

mi = mutual_info_regression(X,y)
feature_scores = {}
for i in range(len(data.feature_names)):
    feature_scores[data.feature_names[i]] = mi[i]

## sort by the second element
sorted_features = sorted(feature_scores.items(), key=lambda x:x[1])

#%% Plotting subplots
fig,ax = plt.subplots()
y_pos = np.arange(len(sorted_features))
ax.barh(y_pos, [score for feature, score in sorted_features])
ax.set_yticks(y_pos)
ax.set_yticklabels([feature for feature, score in sorted_features])
plt.show()
# %%
