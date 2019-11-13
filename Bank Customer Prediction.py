#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Package Importing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, model_selection, metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
import warnings
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from sklearn.metrics import classification_report
from yellowbrick.target import FeatureCorrelation


# In[2]:


#Dataset Importing
bank = pd.read_csv("/Users/di/Desktop/UTD/sem4/Applied Machine Learning/Project 2/Bank_Full.csv")
bank_df = pd.DataFrame(bank)


# In[3]:


#categorical -numerical
from sklearn.preprocessing import LabelEncoder

categorical_column = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                      'day', 'poutcome','y']

for i in categorical_column:
    le = LabelEncoder()
    bank_df[i] = le.fit_transform(bank_df[i])


# In[4]:


#Feature and Target
array=bank_df.values
X = array[:,0:16] 
y = array[:,16]

#Resampling the data with Smote algorithm
X, y = SMOTE().fit_resample(X, y)
print(sorted(Counter(y).items()))


# In[5]:


#Dataset Seperating
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[6]:


#Data Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#y_train = sc_X.transform(y_train)
#y_test = sc_X.transform(y_test)


# In[7]:


#Modeling
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier


# In[8]:


models = [
           ['Lasso: ', Lasso()],
           ['Ridge: ', Ridge()],
           ['KNeighborsRegressor: ',  neighbors.KNeighborsRegressor()],
           ['SVR:' , SVR(kernel='rbf')],
           ['RandomForest ',RandomForestRegressor()],
           ['ExtraTreeRegressor :',ExtraTreesRegressor()],
           ['GradientBoostingClassifier: ', GradientBoostingRegressor()] ,
           ['XGBRegressor: ', xgb.XGBRegressor()] ,
           ['MLPRegressor: ', MLPRegressor(  activation='relu', solver='adam',learning_rate='adaptive',max_iter=1000,learning_rate_init=0.01,alpha=0.01)],
           ['KNN',KNeighborsClassifier( n_neighbors = 6, metric = 'minkowski', p = 2)]
]


# In[9]:


# Run all the proposed models and update the information in a list model_data
import time
from math import sqrt
from sklearn.metrics import mean_squared_error

model_data = []
for name,curr_model in models :
    curr_model_data = {}
    curr_model.random_state = 78
    curr_model_data["Name"] = name
    start = time.time()
    curr_model.fit(X_train,y_train)
    end = time.time()
    curr_model_data["Train_Time"] = end - start
    curr_model_data["Train_R2_Score"] = metrics.r2_score(y_train,curr_model.predict(X_train))
    curr_model_data["Test_R2_Score"] = metrics.r2_score(y_test,curr_model.predict(X_test))
    curr_model_data["Test_RMSE_Score"] = sqrt(mean_squared_error(y_test,curr_model.predict(X_test)))
    model_data.append(curr_model_data)


# In[10]:


model_data


# In[11]:


df = pd.DataFrame(model_data)
df.plot(x="Name", y=['Test_R2_Score' , 'Train_R2_Score' , 'Test_RMSE_Score'], kind="bar" , title = 'R2 Score Results' , figsize= (10,8)) ;


# In[12]:


#Modeling Improvement 
from sklearn.model_selection import GridSearchCV
param_grid = [{
              'max_depth': [80, 150, 200,250],
              'n_estimators' : [100,150,200,250],
              'max_features': ["auto", "sqrt", "log2"]
            }]
reg = ExtraTreesRegressor(random_state=40)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = reg, param_grid = param_grid, cv = 3, n_jobs = -1 , scoring='r2' , verbose=2)
grid_search.fit(X_train, y_train)


# In[13]:


#Show the best parameters 
grid_search.best_params_


# In[14]:


#Trainging with all features 
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0,max_depth=80,max_features='auto')
forest.fit(X_train, y_train)


# In[15]:


#Feature Selection
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[17]:


#Delete the least 2 features 
#feature selection
X_Dropped = bank_df.drop('default', axis=1)
X_Dropped = X_Dropped.drop('loan', axis=1)


#print(sorted(Counter(y).items()))


#Feature and Target
array=X_Dropped.values
X_New = array[:,0:13] 
y_New = array[:,14]

#Resampling the data with Smote algorithm
X_New, y_New = SMOTE().fit_resample(X_New, y_New)
print(sorted(Counter(y_New).items()))

#Dataset Seperating
from sklearn.model_selection import train_test_split
X_train_New, X_test_New, y_train_New, y_test_New = train_test_split(X_New, y_New, test_size = 0.3, random_state = 0)


# In[25]:


#Retrain the model with new data

forest_Dropped = ExtraTreesClassifier(n_estimators=250,
                              random_state=0,max_depth=80,max_features='auto')


c.fit(X_train_New, y_train_New)


y_train_pred_New=forest_Dropped.predict(X_train_New)
y_test_pred_New=forest_Dropped.predict(X_test_New)



# In[37]:


#Original Data and model
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0,max_depth=80,max_features='auto')


forest.fit(X_train, y_train)


y_train_pred=forest.predict(X_train)
y_test_pred=forest.predict(X_test)


# In[28]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_New, y_test_pred_New)


# In[24]:


confusion_matrix(y_test, X_test_pred)


# In[29]:


from sklearn.metrics import roc_curve, auc


# In[33]:


# Run classifier with cross-validation and plot ROC curves
from scipy import interp
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=3)
classifier =  forest


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example with original data')
plt.legend(loc="lower right")
plt.show()


# In[34]:


from scipy import interp
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


cv = StratifiedKFold(n_splits=3)
classifier =  forest_Dropped


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X_New, y_New):
    probas_ = classifier.fit(X_train_New, y_train_New).predict_proba(X_test_New)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test_New, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example with new data')
plt.legend(loc="lower right")
plt.show()


# In[43]:


#Training and Testing error with original data
print(classification_report(y_train,y_train_pred))
print(classification_report(y_test,y_test_pred))


# In[42]:


#Training and Testing error with new data
print(classification_report(y_train_New,y_train_pred_New))
print(classification_report(y_test_New,y_test_pred_New))


# In[44]:


#Learning Curve

from sklearn.model_selection import cross_validate
from yellowbrick.model_selection import LearningCurve
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=12)
sizes = np.linspace(0.3, 1.0, 10)
model = ExtraTreesRegressor(n_estimators=250,
                              random_state=0,max_depth=80,max_features='auto')
visualizer = LearningCurve(
    model, cv=cv, scoring='roc_auc',train_sizes=sizes, n_jobs=5)

visualizer.fit(X_train_New, y_train_New)        # Fit the data to the visualizer
visualizer.show()                           # Finalize and render the figure

