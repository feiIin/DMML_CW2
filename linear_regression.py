from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_predict
from sklearn import linear_model
import numpy as np
from pandas_ml import ConfusionMatrix
from sklearn.neural_network import MLPClassifier

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

    # change the parameters to see how each parameter affects the l1inear classifier
    linear_classifier = linear_model.SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                  early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                  l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=1000,
                  n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
                  random_state=None, shuffle=False, tol=0.001, validation_fraction=0.1,
                  verbose=0, warm_start=False)

    # start training the classifier
    linear_classifier.fit(x_train, y_train)
    print("\nResults from ten cross validation: \n", linear_classifier.score(x_test, y_test))

    # create and plot the confusion matrix
    # cross validation done with cross_val_
    y_train_pred = cross_val_predict(linear_classifier, x_test, y_test, cv=10)
    print("\nConfusion matrix from ten cross validation: \n", confusion_matrix(y_test, y_train_pred))
    print("\n")
    print("Classification report")
    print(classification_report(y_test, y_train_pred))

    print("\n with pandas_ml: \n")
    cm = ConfusionMatrix(y_test, y_train_pred)
    cm.print_stats()


def using_testset(X_trainset, y_trainset, X_testset, y_testset):
    """
    Method using the test set for the classification task.
    Required in task 5.
    :return:
    """
    classifier = linear_model.SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                  early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                  l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=1000,
                  n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
                  random_state=None, shuffle=False, tol=0.001, validation_fraction=0.1,
                  verbose=0, warm_start=False)
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

    nn=MLPClassifier(hidden_layer_sizes=(10,))




if __name__ == "__main__":
    main()