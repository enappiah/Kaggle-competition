#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 20:29:46 2020

@author: enoch_macbook
"""
'''
Overview
PassengerId is the unique id of the row and it doesn't have any effect on target
Survived is the target variable we are trying to predict (0 or 1):
1 = Survived
0 = Not Survived
Pclass (Passenger Class) is the socio-economic status of the passenger and it is a categorical ordinal feature which has 3 unique values (1, 2 or 3):
1 = Upper Class
2 = Middle Class
3 = Lower Class
Name, Sex and Age are self-explanatory
SibSp is the total number of the passengers' siblings and spouse
Parch is the total number of the passengers' parents and children
Ticket is the ticket number of the passenger
Fare is the passenger fare
Cabin is the cabin number of the passenger
Embarked is port of embarkation and it is a categorical feature which has 3 unique values (C, Q or S):
C = Cherbourg
Q = Queenstown
S = Southampton
'''
#https://elitedatascience.com/python-seaborn-tutorial

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Visualising the data
train.info()
#create a new column Is_alone which will tell us whether 
#the person was accompanied (1) or not (0).
def is_alone(x):
    if  (x['SibSp'] + x['Parch'])  > 0:
        return 1
    else:
        return 0

train['Is_alone'] = train.apply(is_alone, axis = 1)
test['Is_alone'] = test.apply(is_alone, axis = 1)

g = sns.barplot(x = "Is_alone",y = "Survived", data=train, palette='pastel')

#how age correct with survival
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")

#Sex, Pclass and Embarked
fig, axes = plt.subplots(1, 3, figsize =(8,4))
attribute = ['Sex', 'Pclass', 'Embarked']
fig.suptitle("Probability of Surviving", fontsize=16)

for i, j in zip(range(3), attribute):
    g = sns.barplot(x= j, y="Survived", data=train, ax= axes[i], palette='pastel')

plt.tight_layout()
fig.subplots_adjust(top= 0.85)


train = train.drop(['PassengerId','Name','SibSp','Parch'], axis = 1)
test = test.drop(['Name','SibSp','Parch'], axis = 1)

print("TRAIN DATA:")
train.isnull().sum()

print("TEST DATA:")
test.isnull().sum()

train.dtypes 

numerical = ['Pclass','Age','Is_alone','Fare']
categorical = ['Sex','Ticket','Cabin','Embarked']
features = numerical + categorical
target = ['Survived']

plt.figure(figsize=(8,4))
correlation_map = sns.heatmap(train.corr(), annot=True, fmt = ".2f", cmap = "coolwarm")
plt.show()

from sklearn.model_selection import train_test_split
train_set, valid_set = train_test_split(train, test_size = 0.3, random_state = 0)

#Transforming the data Numerical:	IterativeImputer & StandardScaler
#Categorical:SimpleImputer & OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

numerical_transformer = Pipeline(steps=[('iterative', IterativeImputer(max_iter = 10, 
                                                           random_state=0)), 
                                        ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer(
    transformers=[('num', numerical_transformer, numerical),
                  ('cat', categorical_transformer, categorical)])

#.............................................................................
#Defining Models

'''
1.1 Ensembling
Ensemble methods are techniques that create multiple models and then combine them to produce improved results. Ensemble methods usually produces more accurate solutions than a single model would. The models used to create such ensemble models are called ‘base models’.

We will use Linear SVM, Radial SVM, Logistic Regression and Random Forest Classifier, and use their results to predict.

We will do ensembling with the Voting Ensemble. Voting is one of the simplest ways of combining the predictions from multiple machine learning algorithms. It works by first creating two or more standalone models from your training dataset. A Voting Classifier can then be used to wrap your models and average the predictions of the sub-models when asked to make predictions for new data.

We will be using weighted Voting Classifier. We will assign to the classifiers according to their accuracies. So the classifier with single accuracy will be assigned the highest weight and so on.

But before directly moving to using Voting Classifier, let's take a look at how the above mentioned classification algorithms work individually.
'''
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

accuracy = []
classifiers = ['Linear SVM', 'Radial SVM', 'LogisticRegression', 'RandomForestClassifier']
models = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(), 
                 RandomForestClassifier(n_estimators=200, random_state=0)]

for i in models:
    model = i
    pipe = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])
    pipe.fit(train_set[features], np.ravel(train_set[target]))
    prediction = pipe.predict(valid_set[features])
    accuracy.append(pipe.score(valid_set[features], valid_set[target]))

observations = pd.DataFrame(accuracy, index=classifiers, columns=['Score'])
observations.sort_values(by = 'Score', ascending = False)

#=============================================================================
from sklearn.ensemble import VotingClassifier

linear_svm = svm.SVC(kernel='linear', C=0.1,gamma=10, probability=True)
pipe_linear = Pipeline(steps=[('preprocessor', preprocessor),  ('model', linear_svm)])

radial_svm = svm.SVC(kernel='rbf', C=0.1,gamma=10, probability=True)
pipe_radial = Pipeline(steps=[('preprocessor', preprocessor),  ('model', radial_svm)])

rand = RandomForestClassifier(n_estimators=200, random_state=0)
pipe_rand = Pipeline(steps=[('preprocessor', preprocessor),  ('model', rand)])


ensemble_all = VotingClassifier(estimators=[('Linear_svm', pipe_linear),
                                            ('Radial_svm', pipe_radial), 
                                            ('Random Forest Classifier', pipe_rand)],
                                voting='soft', weights=[1,1,4])

ensemble_all.fit(train_set[features], np.ravel(train_set[target]))
pred_valid = ensemble_all.predict(valid_set[features])

from sklearn.metrics import confusion_matrix, classification_report 

acc_train = round(ensemble_all.score(train_set[features], train_set[target]) * 100, 2)
acc_valid = round(ensemble_all.score(valid_set[features], valid_set[target]) * 100, 2)

print("Train set Accuracy: ", acc_train, "%\nValidation set Accuracy: ", acc_valid, "%")
print("\nConfusion Matrix:\n", confusion_matrix(valid_set[target], pred_valid))
print("\nClassification Report:\n", classification_report(valid_set[target], pred_valid))

#.............................................................................
#RF
model_RF = RandomForestClassifier(n_estimators=200, random_state = 0)
pipe_RF = Pipeline(steps=[('preprocessor', preprocessor),('model', model_RF)])
pipe_RF.fit(train_set[features], np.ravel(train_set[target]))
pred_valid = pipe_RF.predict(valid_set[features])

from sklearn.metrics import confusion_matrix, classification_report

acc_ran_train = round(pipe_RF.score(train_set[features], train_set[target]) * 100, 2)
acc_ran_valid = round(pipe_RF.score(valid_set[features], valid_set[target]) * 100, 2)
print("Train set Accuracy: ", acc_ran_train, "%\nValidation set Accuracy: ", acc_ran_valid, "%")
print("\nConfusion Matrix:\n", confusion_matrix(valid_set[target], pred_valid))
print("\nClassification Report:\n", classification_report(valid_set[target], pred_valid))

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#save model
from sklearn.externals import joblib
joblib_filename = 'joblib_tatinic_rf_model.pkl'
joblib.dump(pipe_RF,joblib_filename)

#make predictions on test data
model = joblib.load(joblib_filename)
pred_test = model.predict(test[features])
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred_test})
output.to_csv('submission.csv', index=False)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth':[10,20,30,50],
    'n_estimators':[150,200,300]
    }
rfc= RandomForestClassifier()
grid_search_rf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose= 2)
grid_search = Pipeline(steps=[('preprocessor', preprocessor),('model', grid_search_rf)])

grid_search.fit(train_set[features], np.ravel(train_set[target]))
best_grid = grid_search
#best_grid = grid_search.best_estimator_

pred_valid = best_grid.predict(valid_set[features])

from sklearn.metrics import confusion_matrix, classification_report
acc_ran_train = round(best_grid.score(train_set[features], train_set[target]) * 100, 2)
acc_ran_valid = round(best_grid.score(valid_set[features], valid_set[target]) * 100, 2)
print("Train set Accuracy: ", acc_ran_train, "%\nValidation set Accuracy: ", acc_ran_valid, "%")
print("\nConfusion Matrix:\n", confusion_matrix(valid_set[target], pred_valid))
print("\nClassification Report:\n", classification_report(valid_set[target], pred_valid))

'''
###XGB
from xgboost import XGBClassifier
model_xgb = XGBClassifier()
pipe_xgb = Pipeline(steps=[('preprocessor', preprocessor),('model', model_xgb)])
pipe_xgb.fit(train_set[features], np.ravel(train_set[target]))
pred_valid_xgb = pipe_xgb.predict(valid_set[features])

from sklearn.metrics import confusion_matrix, classification_report

acc_ran_train = round(pipe_xgb.score(train_set[features], train_set[target]) * 100, 2)
acc_ran_valid = round(pipe_xgb.score(valid_set[features], valid_set[target]) * 100, 2)
print("Train set Accuracy: ", acc_ran_train, "%\nValidation set Accuracy: ", acc_ran_valid, "%")
print("\nConfusion Matrix:\n", confusion_matrix(valid_set[target], pred_valid_xgb))
print("\nClassification Report:\n", classification_report(valid_set[target], pred_valid_xgb))
'''
