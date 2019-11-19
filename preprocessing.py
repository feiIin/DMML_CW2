#!/usr/bin/env python

"""
Preprocessing: new datasets are created to be used in Weka and Python
"""
import csv
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import preprocessing


def get_array_of_matrix(dataset):
    """ This method transforms the data set from matrix
        to array of matrices, where each matrix is ​​an image.
        """
    array_of_images = []
    for row in dataset:
        row = np.asarray(row)
        matrix = np.reshape(row, (48, 48))
        array_of_images.append(matrix)
    return array_of_images


def crop_dataset(dataset, row, clmn):
    """ This method cropped each image from 48*48 to the new
        "row" and "clmn" selected. It finds the center of the
         image and from there it cuts it.
        """
    copped_dataset = []
    for image in dataset:
        y, x = image.shape
        first_x = x//2-(row//2)
        first_y = y//2-(clmn//2)
        image = image[first_y:first_y + clmn, first_x:first_x + row]
        copped_dataset.append(image)
    return copped_dataset


def normalise_dataset(dataset):
    new_matrix = preprocessing.normalize(dataset)
    return new_matrix


def get_matrix_from_array(dataset):
    """ This method transform the dataset from array of matrices
        to matrix
        """
    dataset_flatten = []
    for image in dataset:
        image = image.flatten()
        dataset_flatten.append(image)
    if len(image) == 40:
        dataset_flatten = np.reshape(dataset_flatten, (12660, 1600))
    if len(image) == 48:
        dataset_flatten = np.reshape(dataset_flatten, (12660, 2304))
    return dataset_flatten


def get_csvfile(dataset, labels, name):
    """This method saves the dataset in a csv file
        """
    dataset = np.append(dataset, labels, axis=1)
    headers = []
    columns = len(dataset[0])
    for i in range(columns):
        i += 1
        headers.append(i)

    with open(name, mode='w') as file:
        my_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        my_writer.writerow(headers)
        for row in dataset:
            my_writer.writerow(row)

    return print("done")


def main():

    # Load datasets x_train_gr_smpl and y_train_smpl
    x_train_gr_smpl = pd.read_csv("./training_sets/x_train_gr_smpl.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl = pd.read_csv("./training_sets/y_train_smpl.csv", delimiter=",", dtype=np.uint8)

    # Load dataset train_smpl_<label>. with 0 and 1 (0 = class; 1 = no class)
    y_train_smpl_0 = pd.read_csv("./training_sets/y_train_smpl_0.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_1 = pd.read_csv("./training_sets/y_train_smpl_1.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_2 = pd.read_csv("./training_sets/y_train_smpl_2.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_3 = pd.read_csv("./training_sets/y_train_smpl_3.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_4 = pd.read_csv("./training_sets/y_train_smpl_4.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_5 = pd.read_csv("./training_sets/y_train_smpl_5.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_6 = pd.read_csv("./training_sets/y_train_smpl_6.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_7 = pd.read_csv("./training_sets/y_train_smpl_7.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_8 = pd.read_csv("./training_sets/y_train_smpl_8.csv", delimiter=",", dtype=np.uint8)
    y_train_smpl_9 = pd.read_csv("./training_sets/y_train_smpl_9.csv", delimiter=",", dtype=np.uint8)



    """ Preprocessing:
    To create and save new datasets, comment and un-comment code based on the 
    desired final dataset"""


    dataset = np.asmatrix(x_train_gr_smpl)
    norm_dataset = normalise_dataset(dataset)

    aom_dataset = get_array_of_matrix(norm_dataset)
    # plt.imshow(aom_dataset[0], cmap="gray")
    # plt.show()
    # print(aom_dataset)

    cropped_dataset = crop_dataset(aom_dataset, 40, 40) # un po' bruttino
    # plt.imshow(cropped_dataset[0], cmap="gray")
    # plt.show()

    new_dataset = get_matrix_from_array(cropped_dataset)
    complete_labels = y_train_smpl

    # smaller_dataset = cropped_dataset[::2]
    # smaller_labels = y_train_smpl[::2]

    # get csv files of smaller datasets
    get_csvfile(new_dataset, complete_labels, "x_train_normalised_with_y_train_smpl.csv")

    # # get csv of cropped datasets
    get_csvfile(new_dataset, y_train_smpl_0, "x_train_normalised_with_y_train_smpl_0.csv")
    get_csvfile(new_dataset, y_train_smpl_1, "x_train_normalised_with_y_train_smpl_1.csv")
    get_csvfile(new_dataset, y_train_smpl_2, "x_train_normalised_with_y_train_smpl_2.csv")
    get_csvfile(new_dataset, y_train_smpl_3, "x_train_normalised_with_y_train_smpl_3.csv")
    get_csvfile(new_dataset, y_train_smpl_4, "x_train_normalised_with_y_train_smpl_4.csv")
    get_csvfile(new_dataset, y_train_smpl_5, "x_train_normalised_with_y_train_smpl_5.csv")
    get_csvfile(new_dataset, y_train_smpl_6, "x_train_normalised_with_y_train_smpl_6.csv")
    get_csvfile(new_dataset, y_train_smpl_7, "x_train_normalised_with_y_train_smpl_7.csv")
    get_csvfile(new_dataset, y_train_smpl_8, "x_train_normalised_with_y_train_smpl_8.csv")
    get_csvfile(new_dataset, y_train_smpl_9, "x_train_normalised_with_y_train_smpl_9.csv")


if __name__ == "__main__":
    main()