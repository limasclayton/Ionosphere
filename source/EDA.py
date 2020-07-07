import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import LabelEncoder
# caminhos
dataset_path = "input/ionosphere.data"

#names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI','target']

# reading data and giving prefix duo to no column names
df = pd.read_csv(dataset_path, prefix='sensor_', header=None)

# initial EDA
# info(), head(), tail() e describe() from data
print(df.info())
print(df.shape)
print(df.head())
print(df.tail())
for c in df.columns:
    print(df[c].describe())

# Changing target (sensor_34) from bad to 0 and good to 1
df['good'] = [1 if s == 'g' else 0 for s in df.sensor_34]
print(df.good.value_counts())

# sensor_1 appears to be irrelevant
# And it's, only 0 values
print(df.sensor_1.value_counts())
# Dropping column 1
df.drop('sensor_1', inplace=True, axis=1)

# Correlation betweem features
corr = df.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, fmt='.1f')
#plt.show()

# Features that appears to have low (<0.1) correlation with target
# 19, 23, 25, 27, 29, 31
features = ['sensor_19', 'sensor_23', 'sensor_25', 'sensor_27', 'sensor_29', 'sensor_31']
print(corr.good[features])

# Conclusions
# Columns doesnt have names. Infer to be numbers.
# Min/max values already scaled between -1 and 1.
# Remove noise features (sensor_23, sensor_25, sensor_29). Correlation with 'good' < 0.01
# See how model performs with and without features sensor_19, sensor_27 and sensor_31. Wich have correlation with 'good' < 0.05