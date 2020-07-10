import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier

# sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# caminhos
dataset_path = "input/ionosphere.data"

#names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI','target']

# reading data and giving prefix duo to no column names
df = pd.read_csv(dataset_path, prefix='sensor_', header=None)

# PREPROCESS
# Changing target's name from 'sensor_34' to 'good' and values from bad to 0 and good to 1
df['good'] = [1 if s == 'g' else 0 for s in df.sensor_34]
df.drop('sensor_34', inplace=True, axis=1)

# Scale features

# FEATURE ENGINEERING

# FEATURE SELECTION

# Dropping column sensor_1 for adding no value the dataset
df.drop('sensor_1', inplace=True, axis=1)

# Noise features to remove
df.drop(['sensor_23', 'sensor_25','sensor_29'], inplace=True, axis=1)

# Featues with low correlation to test the model with and wothout
features_1 = ['sensor_19', 'sensor_23', 'sensor_25', 'sensor_27', 'sensor_29', 'sensor_31']

# Train, validation, test split
X = df.drop('good', axis=1)
y = df.good
print(y.value_counts())

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y==1, stratify=y, test_size=0.3, random_state=123) 
print(X_train_full.shape, y_train_full.shape)
print(X_test.shape, y_test.shape)

# This step is necessary only on Keras
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, stratify=y_train_full, test_size=0.1, random_state=123)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

# MODEL
# parameters grid to search
neurons = np.arange(16, 128, 16)
learning_rate = np.logspace(-4, 1, 20)
epochs = np.arange(100, 1000, 100)

# Training with GridSearch on Keras
input_shape_train_full = (X_train_full.shape[0],)
print(input_shape_train_full)
param_grid_keras = {
    'neurons' : neurons,
    'learning_rate' : learning_rate,
    'epochs' : epochs
}


def get_model(neurons, learning_rate):
    opt = Adam(learning_rate=learning_rate)
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(245,)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[])
    return model

model = KerasClassifier(build_fn=get_model)
grid = GridSearchCV(estimator=model, param_grid=param_grid_keras, n_jobs=-1)
grid.fit(X_train_full, y_train_full)
results = pd.DataFrame(data=grid.cv_results_)
results.to_csv('resultados_keras.csv')
print(grid.best_estimator_)
print(grid.best_score_)
print(grid.best_params_)

# Early stop and validation on Keras with best training params

# GridSearch with MLPClassifier ()
param_grid_sklearn = {
    'hidden_layer_sizes' : neurons,
    'learning_rate_init' : learning_rate,
    'max_iter' : epochs
}
clf = MLPClassifier()
grid = GridSearchCV(clf, param_grid=param_grid_sklearn, n_jobs=-1)
grid.fit(X_train_full, y_train_full)
#print(grid.cv_results_)
results = pd.DataFrame(data=grid.cv_results_)
results.to_csv('resultados_sklearn.csv')
print(grid.best_estimator_)
print(grid.best_score_)
print(grid.best_params_)
#{'hidden_layer_sizes': 80, 'learning_rate_init': 0.1, 'max_iter': 300}

# Using Early Stop with best params from training
best_params = {
    'hidden_layer_sizes': 80,
    'learning_rate_init': 0.1, 
    'max_iter': 300, 
    'random_state':123
    }
mlp = MLPClassifier(early_stopping=True, n_iter_no_change=10, **best_params)
mlp.fit(X_train_full, y_train_full)
print('Loss:', mlp.loss_)
print('Test score:', mlp.score(X_test, y_test))