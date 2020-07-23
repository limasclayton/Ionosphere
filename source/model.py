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
df.drop(['sensor_23', 'sensor_25','sensor_29'], inplace=True, axis=1)

# FEATURE ENGINEERING

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

param_grid = [{
    'classifier__random_state' : [RANDOM_STATE],
    'classifier__C' : np.logspace(-3, 0, 25),
    'classifier__max_iter' : np.arange(300, 500, 100)    
    }]

lr_cv = RandomizedSearchCV(lr_pipe, param_grid, n_jobs=-1, cv=10, random_state=RANDOM_STATE)
lr_cv.fit(X_train, y_train)

print('-' * 100)
print('Logistic Regression pipeline train score: {:.3f}'.format(lr_cv.score(X_train, y_train)))
print('Logistic Regression pipeline validation score: {0}'.format(lr_cv.best_score_))
print('Logistic Regression pipeline best params: {0}'.format(lr_cv.best_params_))
print('Logistic Regression pipeline coeficients: {0}'.format(lr_cv.best_estimator_.named_steps['classifier'].coef_))

print('Logistic Regression pipeline test score: {:.3f}'.format(lr_cv.score(X_test, y_test)))
y_pred = lr_cv.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# SVM Pipeline


# Random Forest Classifier


# HistGradientBoost Classifier
