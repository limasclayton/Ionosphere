import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, auc, roc_curve, confusion_matrix

# path
dataset_path = "input/ionosphere.data"

# variables
RANDOM_STATE = 1

# reading data and giving prefix duo to no column names
df = pd.read_csv(dataset_path, prefix='sensor_', header=None)


# PREPROCESS
# Changing target's name from 'sensor_34' to 'good' and values from bad to 0 and good to 1
df['good'] = [1 if s == 'g' else 0 for s in df.sensor_34]
df.drop('sensor_34', inplace=True, axis=1)

# Dropping column sensor_1 for adding no value the dataset
df.drop('sensor_1', inplace=True, axis=1)

# Noise features to remove
#df.drop(['sensor_23', 'sensor_25','sensor_29'], inplace=True, axis=1)

# FEATURE SELECTION

# Featues with low correlation to test the model with and without
#df.drop(['sensor_19', 'sensor_27', 'sensor_31'], inplace=True, axis=1)

# Separating X and y
X = df.drop('good', axis=1)
y = df.good
print(y.value_counts())

# Train, validation, test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=RANDOM_STATE) 
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# MODEL

# Logistic Regresison pipeline
lr_pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression())
    ])

lr_param_grid = [{
    'classifier__random_state' : [RANDOM_STATE],
    'classifier__C' : np.logspace(-3, 0, 25),
    'classifier__max_iter' : np.arange(300, 500, 100)    
    }]

lr_cv = RandomizedSearchCV(lr_pipe, lr_param_grid, n_jobs=-1, cv=10, random_state=RANDOM_STATE)
lr_cv.fit(X_train, y_train)

print('-' * 100)
print('Logistic Regression pipeline train score: {:.3f}'.format(lr_cv.score(X_train, y_train)))
print('Logistic Regression pipeline validation score: {0}'.format(lr_cv.best_score_))
print('Logistic Regression pipeline best params: {0}'.format(lr_cv.best_params_))
print('Logistic Regression pipeline coeficients: {0}'.format(lr_cv.best_estimator_.named_steps['classifier'].coef_))

print('Logistic Regression pipeline test score: {:.3f}'.format(lr_cv.score(X_test, y_test)))
y_pred_lr = lr_cv.predict(X_test)
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))


# SVM Pipeline
svm_pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', SVC())
    ])

svm_param_grid = [{
    'classifier__random_state' : [RANDOM_STATE],
    'classifier__C' : np.logspace(-3, 0, 25),
    'classifier__kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
    }]

svm_cv = RandomizedSearchCV(svm_pipe, svm_param_grid, n_jobs=-1, cv=10, random_state=RANDOM_STATE)
svm_cv.fit(X_train, y_train)

print('-' * 100)
print('SMV pipeline train score: {:.3f}'.format(svm_cv.score(X_train, y_train)))
print('SMV pipeline validation score: {0}'.format(svm_cv.best_score_))
print('SMV pipeline best params: {0}'.format(svm_cv.best_params_))

print('SMV pipeline test score: {:.3f}'.format(svm_cv.score(X_test, y_test)))
y_pred_svm = svm_cv.predict(X_test)
print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))

# Random Forest Classifier
rf_pipe = Pipeline([
    ('classifier', RandomForestClassifier())
    ])

rf_param_grid = [{
    'classifier__random_state' : [RANDOM_STATE],
    'classifier__n_jobs' : [-1],
    'classifier__n_estimators' : [250],
    #'classifier__n_estimators' : np.arange(100, 500, 50),
    'classifier__min_samples_split' : [6],
    #'classifier__min_samples_split' : np.arange(2, 10, 1),
    'classifier__min_samples_leaf' : [4]
    #'classifier__min_samples_leaf' : np.arange(1, 10, 1)
    }]


rf_cv = RandomizedSearchCV(rf_pipe, rf_param_grid, n_jobs=-1, cv=10, random_state=RANDOM_STATE)
rf_cv.fit(X_train, y_train)

print('-' * 100)
print('RF pipeline train score: {:.3f}'.format(rf_cv.score(X_train, y_train)))
print('RF pipeline validation score: {0}'.format(rf_cv.best_score_))
print('RF pipeline best params: {0}'.format(rf_cv.best_params_))

print('RF pipeline test score: {:.3f}'.format(rf_cv.score(X_test, y_test)))
y_pred_rf = rf_cv.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# XGboost Classifier
param_distributions_xgb = {
    'learning_rate' : np.linspace(0, 1, 50),
    'min_split_loss' : np.logspace(1, 3, 10),
    'max_depth' : np.arange(2, 10, 1),
    'min_child_weight' : np.arange(0, 5, 1),
    'subsample' : np.linspace(0.01, 1, 10),
    'colsample_bytree' : np.linspace(0.01, 1, 10),
    'colsample_bylevel' : np.linspace(0.01, 1, 10),
    'colsample_bynode' : np.linspace(0.01, 1, 10)
}

xgb = XGBClassifier()
xgb_CV = RandomizedSearchCV(xgb, param_distributions=param_distributions_xgb, n_jobs=-1, cv=10, random_state=RANDOM_STATE)
xgb_CV.fit(X_train, y_train)
print('-' * 100)
print('XGB train score: {:.3f}'.format(xgb_CV.score(X_train, y_train)))
print('XGB validation score: {0}'.format(xgb_CV.best_score_))
print('XGB best params: {0}'.format(xgb_CV.best_params_))

print('XGB test score: {:.3f}'.format(xgb_CV.score(X_test, y_test)))
y_pred_xgb = xgb_CV.predict(X_test)
print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))


# HistGradientBoost Classifier

param_distributions_hgb = {
    'learning_rate' : np.logspace(-4, 0, 25),
    'max_iter' : np.arange(100, 500, 50),
    'min_samples_leaf' : np.arange(10, 50, 5),
    'max_leaf_nodes' : np.arange(10, 50, 5),
    'l2_regularization' : np.linspace(0, 1, 5)    
}

hgb = HistGradientBoostingClassifier()
hgb_CV = RandomizedSearchCV(hgb, param_distributions=param_distributions_hgb, n_jobs=-1, cv=10, random_state=RANDOM_STATE)
hgb_CV.fit(X_train, y_train)
print('-' * 100)
print('HGB train score: {:.3f}'.format(hgb_CV.score(X_train, y_train)))
print('HGB validation score: {0}'.format(hgb_CV.best_score_))
print('HGB best params: {0}'.format(hgb_CV.best_params_))

print('HGB test score: {:.3f}'.format(hgb_CV.score(X_test, y_test)))
y_pred_hgb = hgb_CV.predict(X_test)
print(classification_report(y_test, y_pred_hgb))
print(confusion_matrix(y_test, y_pred_hgb))