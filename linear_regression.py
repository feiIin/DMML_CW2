import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_predict
from sklearn import linear_model
from pandas_ml import ConfusionMatrix
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def tenfold_cross_validation(X, y):
    """
    Method using the 10-fold cross validation
    as a method for linear classification.
    Required in task 3.
    :return:
    """

    i = 0
    x_score = []
    y_score = []

    for i in range(1, 11):
        for train_index, test_index in KFold(10).split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        # change the parameters to see how each parameter affects the l1inear classifier
        linear_classifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

        # start training the classifier
        linear_classifier.fit(x_train, y_train)

        # create and plot the confusion matrix
        # cross validation done with cross_val_
        y_train_pred = cross_val_predict(linear_classifier, x_test, y_test, cv=10)

        print("\n Statistics and Confusion matrix obtained with pandas_ml: \n")
        cm = ConfusionMatrix(y_test, y_train_pred)
        stats = cm.stats()

        file = open("linear_classification_9000_cross_validation_" + str(i) + ".txt", "w")
        file.write(str(stats))
        file.close()

        # cm.print_stats()
        # print confusion matrix
        cm.plot(normalized=True)
        plt.show()


def using_testset(X_trainset, y_trainset, X_testset, y_testset):
    """
    Method using the test set for the classification task.
    Required in task 5.
    :return:
    """

    i = 0
    x_score = []
    y_score = []

    for i in range(1, 11):
        classifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
        classifier.fit(X_trainset, y_trainset)
        print("\n\n\n\n\n\nResults using test set: \n", classifier.score(X_testset, y_testset))
        y_predict = classifier.predict(X_testset)

        print("\n Statistics and Confusion matrix obtained with pandas_ml: \n")
        cm = ConfusionMatrix(y_testset, y_predict)
        stats = cm.stats()

        file = open("linear_classification_9000_testset_" + str(i) + ".txt", "w")
        file.write(str(stats))
        file.close()

    # cm.print_stats()
    # print confusion matrix
    cm.plot(normalized=True)
    plt.show()




def main():
    # load data and randomise it
    x_train_data = pd.read_csv("./new_9000_all_classes/x_test_9000_with_y_test_smpl.csv", delimiter=",").sample(frac=1)
    x_test_data = pd.read_csv("./new_9000_all_classes/x_train_9000_with_y_train_smpl.csv", delimiter=",").sample(frac=1)

    # separate training and test data from labels
    y_trainset = x_train_data.iloc[:, -1].values # read them as numpy array, it's needed for the kvalues
    y_trainset = y_trainset.astype(int)
    X_trainset = x_train_data.iloc[:, [0, 1600]].values

    y_testset = x_test_data.iloc[:, -1].values  # read them as numpy array, it's needed for the kvalues
    y_testset = y_testset.astype(int)
    X_testset = x_test_data.iloc[:, [0, 1600]].values

    """
    LINEAR CLASSIFIER
    """
    # ten-fold cross validation, task 3.
    tenfold_cross_validation(X_trainset, y_trainset)

    """
    TEST SET
    """
    # test set validation, task 5
    using_testset(X_trainset, y_trainset, X_testset, y_testset)




if __name__ == "__main__":
    main()