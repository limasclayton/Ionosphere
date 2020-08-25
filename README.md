# Ionosphere

### Input
Instances: 351

Attributes: 34

Year Published: 1989

Link: [Ionosphere Data Set](https://archive.ics.uci.edu/ml/datasets/Ionosphere)


### Objective
Predict whether a structure in the ionosphere is "good" or "bad" given the data collected by sensors.

### Output
source/EDA.py has the exploratory data analysis done in the dataset, changing on target variable for model interpretability feature selection based correlation between features and conclusions about the use of that features on the model.

source/model.py has the feature selection and the pipelines tested on the data.

source/nn.csv has the preprocessing, feature selection, sequential neural network for the dataset, plotting results for analysis and changing on neuron threshold to increase model recall.

model_results.csv has the chosen models and their selected features, details, results and parameters.

rr_results.csv has the configuration and detailed scores of the chosen neural networks.
