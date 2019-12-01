"""
The aim of this class is to test small and huge NN
to see if they overfit and in which cases.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')

x_train_gr_smpl = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl.csv", delimiter=",")
x_test_data = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl.csv", delimiter=",")

x_test = x_test_data.iloc[:, 0:1600]
y_test = x_test_data.iloc[:, -1].astype(int)

x = x_train_gr_smpl.iloc[:, 0:1600]
y = x_train_gr_smpl.iloc[:, -1].astype(int)
X_train, X_test_from_train,  y_train,  y_test_from_train = train_test_split(x, y, test_size=.33, random_state=17)

def printReports(nn):
    nn.fit(X_train, y_train)
    y_train_pred = cross_val_predict(nn, X_test_from_train, y_test_from_train, cv=10)
    print("Results using train set")
    print(nn.score(X_train, y_train))
    print("\n")
    print("Confusion matrix from ten cross validation:")
    print(confusion_matrix(y_test_from_train, y_train_pred))
    print("\n")
    print("Classification report")
    print(classification_report(y_test_from_train, y_train_pred))
    print("Using test set ------------------------------------------------------------")
    print("Results using test set:")
    print(nn.score(x_test, y_test))
    y_test_pred = nn.predict(x_test)
    print("\nConfusion matrix using test set:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred))

print("\n--------------------------------------------------------\n")
print("\n--------------------------------------------------------\n")
print("\n--------------CHANGE NUM LAYER & NEURONS----------------\n")
print("\n--------------------------------------------------------\n")
print("\n--------------------------------------------------------\n")

nn = MLPClassifier(hidden_layer_sizes=1, activation='relu',
                   momentum=0.9, learning_rate_init=0.001, max_iter=200)
printReports(nn)

print("\n--------------------------------------------------------\n")

nn1 = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu',
                   momentum=0.9, learning_rate_init=0.001, max_iter=200)
printReports(nn)

print("\n--------------------------------------------------------\n")
nn2 = MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation='relu',
                   momentum =0.9, learning_rate_init=0.001, max_iter=200)
printReports(nn2)

print("\n--------------------------------------------------------\n")
nn3 = MLPClassifier(hidden_layer_sizes=20, activation='relu',
                   momentum =0.9, learning_rate_init=0.001, max_iter=200)
printReports(nn3)

print("\n--------------------------------------------------------")
nn4 = MLPClassifier(hidden_layer_sizes=30, activation='relu',
                   momentum =0.9, learning_rate_init=0.001, max_iter=200)
printReports(nn4)

print("\n--------------------------------------------------------")
nn5 = MLPClassifier(hidden_layer_sizes=50, activation='relu',
                   momentum =0.9, learning_rate_init=0.001, max_iter=200)
printReports(nn5)

print("\n--------------------------------------------------------\n")
print("\n--------------------------------------------------------\n")
print("\n-------------------CHANGE MAX ITERATION-----------------\n")
print("\n--------------------------------------------------------\n")
print("\n--------------------------------------------------------\n")


# change max iteration
nn6 = MLPClassifier(hidden_layer_sizes=20, activation='relu',
                   momentum =0.9, learning_rate_init=0.001, max_iter=50)
printReports(nn6)

print("\n--------------------------------------------------------\n")
nn7 = MLPClassifier(hidden_layer_sizes=(20,), activation='relu',
                   momentum=0.9, learning_rate_init=0.001, max_iter=100)
printReports(nn7)

print("\n--------------------------------------------------------\n")
# changing learning rate
nn8 = MLPClassifier(hidden_layer_sizes=(20,), activation='relu',
                   momentum=0.9, learning_rate_init=0.01, max_iter=200)
printReports(nn8)

print("\n--------------------------------------------------------\n")
nn9 = MLPClassifier(hidden_layer_sizes=(20,), activation='relu',
                   momentum=0.9, learning_rate_init=0.1, max_iter=200)
printReports(nn9)

print("\n--------------------------------------------------------\n")

print("\n--------------------------------------------------------\n")
print("\n--------------------------------------------------------\n")
print("\n---------------------CHANGE MOMENTUM--------------------\n")
print("\n--------------------------------------------------------\n")
print("\n--------------------------------------------------------\n")


#change momentum
nn10 = MLPClassifier(hidden_layer_sizes=(20,), activation='relu',
                   momentum=0.7, learning_rate_init=0.001,max_iter=200)
printReports(nn10)

print("\n--------------------------------------------------------\n")
nn11 = MLPClassifier(hidden_layer_sizes=(20,), activation='relu',
                   momentum=0.5,learning_rate_init=0.001,max_iter=200)

printReports(nn11)

print("\n--------------------------------------------------------\n")
# change validation fraction
nn12 = MLPClassifier(hidden_layer_sizes=(20,),momentum=0.5,learning_rate_init=0.001,
                     max_iter=200, early_stopping=True,validation_fraction=0.7)
printReports(nn12)

print("\n--------------------------------------------------------\n")
nn13 = MLPClassifier(hidden_layer_sizes=(20,),momentum=0.5,learning_rate_init=0.001,
                     max_iter=200, early_stopping=True,validation_fraction=0.5)
printReports(nn13)

print("\n--------------------------------------------------------\n")


