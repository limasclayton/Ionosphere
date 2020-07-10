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
from keras.callbacks import EarlyStopping

# sklearn
from sklearn.preprocessing import StandardScaler
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

# Dropping column sensor_1 for adding no value the dataset
df.drop('sensor_1', inplace=True, axis=1)

# Noise features to remove
df.drop(['sensor_23', 'sensor_25','sensor_29'], inplace=True, axis=1)

# Separating features before scale
X = df.drop('good', axis=1)
y = df.good
print(y.value_counts())

# Scaling features
for c in X.columns:
    #print(X[c].describe())
    X[c] = StandardScaler().fit_transform(X[c].values.reshape(-1,1))
    #print(X[c].describe())


# FEATURE ENGINEERING

# FEATURE SELECTION

# Featues with low correlation to test the model with and wothout
features_1 = ['sensor_19', 'sensor_23', 'sensor_25', 'sensor_27', 'sensor_29', 'sensor_31']

# Train, validation, test split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=123) 
print(X_train_full.shape, y_train_full.shape)
print(X_test.shape, y_test.shape)

# This step is necessary only on Keras
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, stratify=y_train_full, test_size=0.1, random_state=123)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

# MODEL
# parameters grid to search
neurons = np.arange(16, 128, 16)
learning_rate = np.logspace(-4, 1, 10)
epochs = np.arange(100, 500, 100)

# funcion that returns a keras model
def get_model(neurons=10, learning_rate=0.001):
    opt = Adam(learning_rate=learning_rate)
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(30,)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'mse'])
    return model

# Manually GridSearching
columns = ['neurons', 'learning_rate', 'epochs', 'accuracy', 'mse']
manual_results = pd.DataFrame(columns=columns)

'''
for n in neurons:
    for lr in learning_rate:
        for e in epochs:
            model = get_model(n, lr)
            history = model.fit(X_train_full, y_train_full, epochs=e)
            acc = history.history['accuracy'][-1]
            mse = history.history['mse'][-1]
            results = pd.DataFrame([[n, lr, e, acc, mse]], columns=columns)
            manual_results = manual_results.append(results, ignore_index=True)
            print(n, lr, e, acc, mse)

manual_results.to_csv('manual_results.csv')
'''

# Possible best configurations with acc 1 and error ~ 0
# Neurons: 16, LR: 0.05994842503189409, Epochs: 400.
# Neurons: 32, LR: 0.7742636826811278, Epochs: 200.
# Neurons: 48, LR: 2.782559402207126, Epochs: 200.
# Neurons: 64, LR: 0.05994842503189409, Epochs: 400.
# Neurons: 80, LR: 0.7742636826811278, Epochs: 100. 

# Early stop and validation on Keras with best training params
callbacks = EarlyStopping(monitor='val_mse', patience=10)
model = get_model(16, 0.05994842503189409)
history = model.fit(X_train, y_train, epochs=400, validation_data=[X_val, y_val], callbacks=[callbacks], verbose=0)

# Plotting history for loss
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plotting history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plotting history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

'''
# Training with GridSearch on Keras. Didn't work and the issue is still open for discussion
input_shape_train_full = (X_train_full.shape[1],)
print(input_shape_train_full)
param_grid_keras = {
    'neurons' : neurons
    'learning_rate' : learning_rate,
    'epochs' : epochs
}

model = KerasClassifier(build_fn=get_model)
grid = GridSearchCV(estimator=model, param_grid=param_grid_keras, n_jobs=-1)
grid_results = grid.fit(X_train_full, y_train_full)

results = pd.DataFrame(data=grid_results.cv_results_)
results.to_csv('resultados_keras.csv')
print(grid_results.best_estimator_)
print(grid_results.best_score_)
print(grid_results.best_params_)

# GridSearch with MLPClassifier() didn't work either for the problem specifics
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
'''
