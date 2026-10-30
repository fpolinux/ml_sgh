#%% Imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_csv(r"Wage.csv")
#%% Slice, dice, and display
columns = ['ID', 'year', 'age', 'sex', 'maritl', 'race', 'education', 'region',
       'jobclass', 'health', 'health_ins', 'logwage', 'wage']
data_x = data["age"]
data_y = data["wage"]
plt.scatter(data_x, data_y, facecolor = "None", alpha=0.3, edgecolor = "k") # non-linear 
plt.ylabel("Wage")
plt.xlabel("Age")

#%% smf ols regression to simple polynomial
import statsmodels.formula.api as smf
model = smf.ols("wage ~ age + np.power(age,2)+np.power(age,3)+np.power(age,4)", data = data).fit()
model.summary()
#%% Test multiple polynomials; with different degrees; anova_lm is for analysis of variance
from statsmodels.stats.api import anova_lm 
# Note that anova compares every model to the previous model. m1 will return null whilst 
# all others will be compared to n-1
# p-value contained in Pr(>F) --> F test since we are analysing variances
# in our case, reject at m4 since p_4 = 0.051 > 0.05
m1 = smf.ols("wage ~ age", data = data).fit()
m2 = smf.ols("wage ~ age + np.power(age,2)", data = data).fit()
m3 = smf.ols("wage ~ age + np.power(age,2)+np.power(age,3)", data = data).fit()
m4 = smf.ols("wage ~ age + np.power(age,2)+np.power(age,3)+np.power(age,4)", data = data).fit()
m5 = smf.ols("wage ~ age + np.power(age,2)+np.power(age,3)+np.power(age,4)+ np.power(age,5)", data = data).fit()
anova_lm(m1,m2,m3,m4,m5)
# m3 is our selected curve since m4 is cutoff. 
#%% 
from statsmodels.stats.outliers_influence import summary_table
st, data1, ss2 = summary_table(m3, alpha=0.05)
#print(st) # st is the summary table
predictedvalues = data1[:,2]
predict_mse = data1[:, 3]
predict_mean_ci_low, predict_mean_ci_up = data1[:,4:6].T
order = np.argsort(data_x.values.flatten()) # flattens data into a 1D vector
x_o = data_x.values.flatten()[order]
#print(data_x) # not ordered
# print(x_o) # ordered
plt.scatter(data_x, data_y, facecolor = "None", alpha=0.3, edgecolor = "k") # non-linear 
plt.plot(x_o, predictedvalues[order])

#%% Tidy plotting with CI lower and upper bound
# Interesting that because cubic functions are global functions, 
# they perform worsely at extremes/where there is sparse data 
plt.scatter(data_x, data_y, facecolor = "None", alpha=0.3, edgecolor = "k") # non-linear 
plt.plot(x_o, predictedvalues[order], "b")
plt.plot(x_o, predict_mean_ci_low[order], "r--")
plt.plot(x_o, predict_mean_ci_up[order], "r--")

#%% Fixing the global issue by using piecewise functions; creating dummies for bins
df_cut, bins = pd.cut(data_x, 4, retbins=True, right=True) #equally spaced intervals..maybe not the best method
df_cut.value_counts() 
# To see which bin each individual falls into. axis=1 for columns
df_steps = pd.concat([data_x, df_cut, data_y], keys=["age", "age_cuts", "wage"], axis=1)
df_steps_dummies = pd.get_dummies(df_cut) #generating dummy variables for each bin

#%% Fit, prepare data, and then plot
model2 = sm.GLM(df_steps.wage, df_steps_dummies).fit() # format is (target, input)
xp = np.linspace(data_x.min(), data_x.max())
bin_mapping = np.digitize(xp, bins, right=True)  #indices on the bins
x_pred = pd.get_dummies(bin_mapping)
pred = model2.predict(x_pred)

plt.scatter(data_x, data_y, facecolor = "None", alpha=0.3, edgecolor = "k") # non-linear
plt.plot(xp, pred, "r") # displays stepped

#%% 
from patsy import dmatrix
# 'design matrix' sets our axis; matrix for the obs of our axis
# transformed acis in the bs() block. 
transformed_x1 = dmatrix("bs(train, knots=(25,40,60), degree=3, include_intercept=False)", {"train":data_x}, return_type="dataframe")
transformed_x2 = dmatrix("bs(train, knots=(25,40,50, 65), degree=3, include_intercept=False)", {"train":data_x}, return_type="dataframe")

fit1 = sm.GLM(data_y, transformed_x1).fit()
fit2 = sm.GLM(data_y, transformed_x2).fit()
# Validate predictions... we should split into train/test..ignore the memory leak for now
pred1 = fit1.predict(dmatrix("bs(valid, knots=(25,40,60), degree=3, include_intercept=False)", {"valid":data_x}, return_type="dataframe"))
pred2 = fit2.predict(dmatrix("bs(valid, knots=(25,40, 50,65), degree=3, include_intercept=False)", {"valid":data_x}, return_type="dataframe"))

#%% RMSE scoring
from sklearn.metrics import mean_squared_error
from math import sqrt
rms1 = sqrt(mean_squared_error(data_y, pred1)) #39.878
rms2 = sqrt(mean_squared_error(data_y, pred2)) #29.839
# rms2 < rms1 and therefore rms2 is better

#%% Plotting a "spline"
xp = np.linspace(data_x.min(), data_x.max(), 100)
pred1 = fit1.predict(dmatrix("bs(xp, knots=(25,40,60), degree=3, include_intercept=False)", {"xp":xp}, return_type="dataframe"))
pred2 = fit2.predict(dmatrix("bs(xp, knots=(25,40, 50,65), degree=3, include_intercept=False)", {"xp":xp}, return_type="dataframe"))
plt.scatter(data_x, data_y, facecolor = "None", alpha=0.3, edgecolor = "k") # non-linear
plt.plot(xp, pred1, label="pred1")
plt.plot(xp, pred2, label="pred2")
plt.legend()


## Spline: breaking polynomial fit down into a piecewise function. Then at each
# 'edge' of the piecewise, force the sub-polynomial to conform to the next one.
# conformity is considered for the co-ordinate and nth-derivative.
# n-1th-derivative is chosen by the power, n, of the sub-polynomial
# in terms of choosing the knots, this is based on whatever logic/reasoning you prefer
# 