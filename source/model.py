import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, auc, roc_curve, confusion_matrix

# caminhos
dataset_path = "input/ionosphere.data"

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

# FEATURE ENGINEERING

# FEATURE SELECTION

X = df.drop('good', axis=1)
y = df.good
print(y.value_counts())

# Featues with low correlation to test the model with and without
features_1 = ['sensor_19', 'sensor_27', 'sensor_31']
#df.drop(['sensor_19', 'sensor_27', 'sensor_31'], inplace=True, axis=1)

# Train, validation, test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=123) 
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# MODEL

# Logistic Regresison pipeline


# SVM Pipiline


# Random Forest Classifier


# HistGradientBoost Classifier
