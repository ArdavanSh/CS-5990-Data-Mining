# -------------------------------------------------------------------------
# AUTHOR: Ardavan Sherafat
# FILENAME: roc_curve.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 10 minutes
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# function to transform the data
def transform_data(dataset):

    X = np.zeros((dataset.shape[0], dataset.shape[1] - 2))

    X[:, 0] = np.where(dataset[:, 0] == 'Yes', 1, 0)

    onehot_encoder = OneHotEncoder(categories=[['Single', 'Divorced', 'Married']])
    marital_status_encoded = onehot_encoder.fit_transform(dataset[:, 1].reshape(-1, 1)).toarray()

    for i in range(marital_status_encoded.shape[1]-1, -1, -1):
        X = np.insert(X, 1, marital_status_encoded[:, i], axis=1)

    dataset_str = dataset[:, 2].astype(str)
    X[:, -1] = np.char.replace(dataset_str, 'k', '').astype(float)

    Y = np.zeros((dataset.shape[0], 1))
    Y[:,0] = np.where(dataset[:, -1] == 'Yes', 1, 0)

    return X,Y

# read the dataset cheat_data.csv and prepare the data_training numpy array
# --> add your Python code here

df = pd.read_csv("cheat_data.csv", sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
data_training = np.array(df.values)

# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
# --> add your Python code here
X, y = transform_data(data_training) # The function is implemented above

# transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
# --> add your Python code here
# y = ?

# split into train/test sets using 30% for test
# --> add your Python code here
trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.30)

# generate a no skill prediction (random classifier - scores should be all zero)
# --> add your Python code here
ns_probs = np.zeros(len(testX)) + 0.5

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

# keep probabilities for the positive outcome only
# --> add your Python code here
dt_probs = dt_probs[:, 1]

# calculate scores by using both classifiers (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()