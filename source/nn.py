import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# caminhos
dataset_path = "input/ionosphere.data"

#names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI','target']

# reading data and giving prefix duo to no column names
df = pd.read_csv(dataset_path, prefix='sensor_', header=None)

# PREPROCESS
# Changing target's name from 'sensor_34' to 'good' and values from bad to 0 and good to 1
df['good'] = [1 if s == 'g' else 0 for s in df.sensor_34]

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
#X_train, train_test_split

# MODEL
# parameters grid to seach
neurons = [20, 40, 60, 80]
learning_rate = [0.5, 0.3, 0.1]
epochs = [300, 1000, 5000]
params_grid = {
    'hidden_layer_sizes' : neurons,
    'learning_rate_init' : learning_rate,
    'max_iter' : epochs
}