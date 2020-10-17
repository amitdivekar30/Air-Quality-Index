# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 19:00:39 2020

@author: aad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data/Real-Data/Real_Combine.csv')

#check for nullvalues
sns.heatmap(df.isnull(), yticklabels=False,cbar=False,cmap='viridis')

df=df.dropna()

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

sns.pairplot(df)

df.corr()

#correlation matrixwith heatmap
cormat=df.corr()
top_corr_features=cormat.index
plt.figure(figsize=(28,28))
#plot heatmp
g=sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='RdYlGn')

cormat.index


#FeatureImportance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)

print(model.feature_importances_)

#plot graph

feat_importances= pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(8).plot(kind='barh')
plt.show()

sns.distplot(y)

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression

reg =LinearRegression()

reg.fit(X_train,y_train)

reg.coef_
reg.intercept_


print("Coefficient of determination R^2 <-- on train set: {}".format(reg.score(X_train, y_train)))

print("Coefficient of determination R^2 <-- on train set: {}".format(reg.score(X_test, y_test)))

from sklearn.model_selection import cross_val_score
score=cross_val_score(reg,X,y,cv=5)

score.mean()

coeff_df = pd.DataFrame(reg.coef_,X.columns,columns=['Coefficient'])
coeff_df

prediction=reg.predict(X_test)


sns.distplot(y_test-prediction)

plt.scatter(y_test,prediction)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

import pickle


# open a file, where you ant to store the data
file = open('regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(reg, file)


########################## Ridge Regression #######################
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

########################## Lasso Regression #######################
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

prediction=lasso_regressor.predict(X_test)
sns.distplot(y_test-prediction)

plt.scatter(y_test,prediction)

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# open a file, where you ant to store the data
file = open('lasso_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(lasso_regressor, file)


########################## Decision Tree #######################
from sklearn.tree import DecisionTreeRegressor
reg_dt=DecisionTreeRegressor(criterion="mse")

reg_dt.fit(X_train,y_train)

print("Coefficient of determination R^2 <-- on train set: {}".format(reg_dt.score(X_train, y_train)))

print("Coefficient of determination R^2 <-- on test set: {}".format(reg_dt.score(X_test, y_test)))

from sklearn.model_selection import cross_val_score
score=cross_val_score(reg_dt,X,y,cv=5)

score.mean()

# Tree Visualization

##conda install pydotplus
## conda install python-graphviz

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus

features = list(df.columns[:-1])
features

import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


dot_data = StringIO()  
export_graphviz(reg_dt, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

#Model Evaluation
prediction=reg_dt.predict(X_test)

sns.distplot(y_test-prediction)
plt.scatter(y_test,prediction)

# Hyperparameter Tuning Decision Tree Regressor

DecisionTreeRegressor()

## Hyper Parameter Optimization

params={
 "splitter"    : ["best","random"] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_samples_leaf" : [ 1,2,3,4,5 ],
"min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
 "max_features" : ["auto","log2","sqrt",None ],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70]
    
}

## Hyperparameter optimization using GridSearchCV
from sklearn.model_selection import GridSearchCV

random_search=GridSearchCV(reg_dt,param_grid=params,scoring='neg_mean_squared_error',n_jobs=-1,cv=10,verbose=3)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        
        
from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,y)
timer(start_time) # timing ends here for "start_time" variable

random_search.best_params_

random_search.best_score_

predictions=random_search.predict(X_test)

sns.distplot(y_test-predictions)



from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# open a file, where you ant to store the data
file = open('decision_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(random_search, file)


########################## Random Forest #######################
from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor()
regressor.fit(X_train,y_train)

print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))
from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)

score.mean()

#Model Evaluation

prediction=regressor.predict(X_test)
sns.distplot(y_test-prediction)

plt.scatter(y_test,prediction)

RandomForestRegressor()

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Method of selecting samples for training each tree
# bootstrap = [True, False]


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)

rf_random.best_params_

rf_random.best_score_

predictions=rf_random.predict(X_test)

sns.distplot(y_test-predictions)

plt.scatter(y_test,prediction)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)