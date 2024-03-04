# -------------------------------------------------------------------------
# AUTHOR: Ardavan Sherafat
# FILENAME: Decision_Tree.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']


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


for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    # transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    # Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    # be converted to a float.
    X, Y = transform_data(data_training) # The function is implemented above


    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y = 

    accuracies = []
    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)

        #plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        plt.show()

        #read the test data and add this data to data_test NumPy
        #--> add your Python code here
        df_test = pd.read_csv('cheat_test.csv', sep=',', header=0)  
        data_test = np.array(df_test.values)[:,1:]

        X_test, Y_test = transform_data(data_test)
        
        correct_predictions = 0

        for data,target in zip(X_test, Y_test):
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            class_predicted = clf.predict([data])[0]

            #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            #--> add your Python code here
            correct_predictions += 1 if class_predicted == target else 0

        #find the average accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
        accuracy = correct_predictions / len(data_test)
        accuracies.append(accuracy)

        # print(clf.score(X_test,Y_test))
    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    #--> add your Python code here
    final_accuracy = np.mean(accuracies)
    print(f'Final accuracy when training on {ds}: {final_accuracy}')



