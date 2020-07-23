import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, auc, roc_curve, confusion_matrix

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

# Featues with low correlation to test the model with and without
features_1 = ['sensor_19', 'sensor_27', 'sensor_31']
df.drop(['sensor_19', 'sensor_27', 'sensor_31'], inplace=True, axis=1)

# Train, validation, test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=123) 
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, stratify=y_train_full, test_size=0.1, random_state=123)
#print(X_train.shape, y_train.shape)
#print(X_val.shape, y_val.shape)

# MODEL
# parameters grid to search
neurons = np.arange(16, 128, 16)
learning_rate = np.logspace(-4, 1, 10)
epochs = np.arange(100, 500, 100)

# Funcion that returns a keras model
def get_model(neurons=16, learning_rate=0.001):
    opt = Adam(learning_rate=learning_rate)
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'mse'])
    return model

# Early stop and validation on Keras with best training params
callbacks = EarlyStopping(monitor='val_loss', patience=25)
model = get_model(32)
history = model.fit(X_train, y_train, epochs=500, validation_split=0.1, callbacks=[callbacks], verbose=0)


# Plotting train history for mse
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Plotting train history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Plotting train history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Evaluating the model
evaluation = model.evaluate(X_test, y_test)
print('Test loss:', evaluation[0])
print('Test acc:', evaluation[1])
print('Test mse:', evaluation[2])

# METRICS
# Calculating 
y_pred = np.round(model.predict(X_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Calculating roc_auc curve
probs = model.predict_proba(X_test)
print(probs)
fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
print(list(zip(tpr,fpr,threshold)))

# Plotting roc_auc curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Changing threshold to see if 0 recall increases
# It does and the other metrics don't fall too much.
# Sounds like a good ideia to do it
y_pred_threshold = model.predict_proba(X_test) > 0.6
print(classification_report(y_test, y_pred_threshold))
print(confusion_matrix(y_test, y_pred_threshold))