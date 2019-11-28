from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_predict
from sklearn import linear_model
import numpy as np


def tenfold_cross_validation(X, y):
    """
    Method using the 10-fold cross validation
    as a method for linear classification.
    Required in task 3.
    :return:
    """
    for train_index, test_index in KFold(10).split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    linear_classifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    linear_classifier.fit(x_train, y_train)
    print("\nResults from ten cross validation: \n", linear_classifier.score(x_test, y_test))

    # create and plot the confusion matrix
    y_train_pred = cross_val_predict(linear_classifier, x_test, y_test, cv=3)
    print("\nConfusion matrix from ten cross validation: \n", confusion_matrix(y_test, y_train_pred))


def using_testset(X_trainset, y_trainset, X_testset, y_testset):
    """
    Method using the test set for the classification task.
    Required in task 5.
    :return:
    """
    classifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    classifier.fit(X_trainset, y_trainset)
    print("\n\nResults using test set: \n", classifier.score(X_testset, y_testset))

    # create and plot the confusion matrix
    y_test_pred = cross_val_predict(classifier, X_testset, y_testset, cv=3)
    print("\nConfusion matrix using test set: \n", confusion_matrix(y_testset, y_test_pred))


def main():
    # load data and randomise it
    x_train_data = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl.csv", delimiter=",").sample(frac=1)
    x_test_data = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl.csv", delimiter=",").sample(frac=1)

    # separate training and test data from labels
    y_trainset = x_train_data.iloc[:, -1].values # read them as numpy array, it's needed for the kvalues
    y_trainset = y_trainset.astype(int)
    X_trainset = x_train_data.iloc[:, [0, 1600]].values

    y_testset = x_test_data.iloc[:, -1].values  # read them as numpy array, it's needed for the kvalues
    y_testset = y_testset.astype(int)
    X_testset = x_test_data.iloc[:, [0, 1600]].values

    # ten-fold cross validation, task 3.
    tenfold_cross_validation(X_trainset, y_trainset)

    # test set validation, task 5
    using_testset(X_trainset, y_trainset, X_testset, y_testset)



if __name__ == "__main__":
    main()