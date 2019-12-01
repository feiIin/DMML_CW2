import csv
import random

import pandas as pd
import numpy as np


def get_csvfile(dataset, name):
    """
    This method saves the dataset in a csv file
    :return csv with the dataset
    """
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


def split_dataset(train_set, test_set):
    """
    This method changes the amount of data in train and test set
    :return new train and test sets
    """
    train_set = train_set.iloc[1:]
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    train = train_set.iloc[4000:]
    test = train_set.iloc[:4000]
    test = pd.concat([test_set, test])

    return train, test

def main():
    x_train_gr_smpl = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl.csv", delimiter=",")
    x_test_smpl = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl.csv", delimiter=",")

    # x_train_gr_smpl_0 = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl_0.csv", delimiter=",")
    # x_train_gr_smpl_1 = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl_1.csv", delimiter=",")
    # x_train_gr_smpl_2 = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl_2.csv", delimiter=",")
    # x_train_gr_smpl_3 = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl_3.csv", delimiter=",")
    # x_train_gr_smpl_4 = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl_4.csv", delimiter=",")
    # x_train_gr_smpl_5 = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl_5.csv", delimiter=",")
    # x_train_gr_smpl_6 = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl_6.csv", delimiter=",")
    # x_train_gr_smpl_7 = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl_7.csv", delimiter=",")
    # x_train_gr_smpl_8 = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl_8.csv", delimiter=",")
    # x_train_gr_smpl_9 = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl_9.csv", delimiter=",")
    #
    # y_test_smpl_0 = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl_0.csv", delimiter=",")
    # y_test_smpl_1 = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl_1.csv", delimiter=",")
    # y_test_smpl_2 = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl_2.csv", delimiter=",")
    # y_test_smpl_3 = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl_3.csv", delimiter=",")
    # y_test_smpl_4 = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl_4.csv", delimiter=",")
    # y_test_smpl_5 = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl_5.csv", delimiter=",")
    # y_test_smpl_6 = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl_6.csv", delimiter=",")
    # y_test_smpl_7 = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl_7.csv", delimiter=",")
    # y_test_smpl_8 = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl_8.csv", delimiter=",")
    # y_test_smpl_9 = pd.read_csv("./normalised_test_sets/x_test_normalised_with_y_test_smpl_9.csv", delimiter=",")


    # create random index array to use to randomise data always in the same way
    # new_random_index = random.sample(range(8999), 8999)

    # take new data from existing data set, all labels
    [train_smpl, test_smpl] = split_dataset(x_train_gr_smpl, x_test_smpl)

    train_smpl = np.asarray(train_smpl)
    test_smpl = np.asarray(test_smpl)

    get_csvfile(train_smpl, "x_train_4000_with_y_train_smpl.csv")
    get_csvfile(test_smpl, "x_test_4000_with_y_test_smpl.csv")

    # take new data from existing data set, 0 1 labels
    #
    # [train_smpl_0, test_smpl_0] = split_dataset(x_train_gr_smpl_0, y_test_smpl_0, new_random_index)
    # train_smpl_0 = np.asarray(train_smpl_0)
    # test_smpl_0 = np.asarray(test_smpl_0)
    #
    # [train_smpl_1, test_smpl_1] = split_dataset(x_train_gr_smpl_1, y_test_smpl_1, new_random_index)
    # [train_smpl_2, test_smpl_2] = split_dataset(x_train_gr_smpl_2, y_test_smpl_2, new_random_index)
    # [train_smpl_3, test_smpl_3] = split_dataset(x_train_gr_smpl_3, y_test_smpl_3, new_random_index)
    # [train_smpl_4, test_smpl_4] = split_dataset(x_train_gr_smpl_4, y_test_smpl_4, new_random_index)
    # [train_smpl_5, test_smpl_5] = split_dataset(x_train_gr_smpl_5, y_test_smpl_5, new_random_index)
    # [train_smpl_6, test_smpl_6] = split_dataset(x_train_gr_smpl_6, y_test_smpl_6, new_random_index)
    # [train_smpl_7, test_smpl_7] = split_dataset(x_train_gr_smpl_7, y_test_smpl_7, new_random_index)
    # [train_smpl_8, test_smpl_8] = split_dataset(x_train_gr_smpl_8, y_test_smpl_8, new_random_index)
    # [train_smpl_9, test_smpl_9] = split_dataset(x_train_gr_smpl_9, y_test_smpl_9, new_random_index)
    #
    #
    # train_smpl_1 = np.asarray(train_smpl_1)
    # train_smpl_2 = np.asarray(train_smpl_2)
    # train_smpl_3 = np.asarray(train_smpl_3)
    # train_smpl_4 = np.asarray(train_smpl_4)
    # train_smpl_5 = np.asarray(train_smpl_5)
    # train_smpl_6 = np.asarray(train_smpl_6)
    # train_smpl_7 = np.asarray(train_smpl_7)
    # train_smpl_8 = np.asarray(train_smpl_8)
    # train_smpl_9 = np.asarray(train_smpl_9)
    #
    # test_smpl_1 = np.asarray(test_smpl_1)
    # test_smpl_2 = np.asarray(test_smpl_2)
    # test_smpl_3 = np.asarray(test_smpl_3)
    # test_smpl_4 = np.asarray(test_smpl_4)
    # test_smpl_5 = np.asarray(test_smpl_5)
    # test_smpl_6 = np.asarray(test_smpl_6)
    # test_smpl_7 = np.asarray(test_smpl_7)
    # test_smpl_8 = np.asarray(test_smpl_8)
    # test_smpl_9 = np.asarray(test_smpl_9)
    #
    # get_csvfile(train_smpl_0, "x_train_9000_with_y_train_smpl_0.csv")
    # get_csvfile(train_smpl_1, "x_train_9000_with_y_train_smpl_1.csv")
    # get_csvfile(train_smpl_2, "x_train_9000_with_y_train_smpl_2.csv")
    # get_csvfile(train_smpl_3, "x_train_9000_with_y_train_smpl_3.csv")
    # get_csvfile(train_smpl_4, "x_train_9000_with_y_train_smpl_4.csv")
    # get_csvfile(train_smpl_5, "x_train_9000_with_y_train_smpl_5.csv")
    # get_csvfile(train_smpl_6, "x_train_9000_with_y_train_smpl_6.csv")
    # get_csvfile(train_smpl_7, "x_train_9000_with_y_train_smpl_7.csv")
    # get_csvfile(train_smpl_8, "x_train_9000_with_y_train_smpl_8.csv")
    # get_csvfile(train_smpl_9, "x_train_9000_with_y_train_smpl_9.csv")
    #
    # get_csvfile(test_smpl_0, "x_test_9000_with_y_test_smpl_0.csv")
    # get_csvfile(test_smpl_1, "x_test_9000_with_y_test_smpl_1.csv")
    # get_csvfile(test_smpl_2, "x_test_9000_with_y_test_smpl_2.csv")
    # get_csvfile(test_smpl_3, "x_test_9000_with_y_test_smpl_3.csv")
    # get_csvfile(test_smpl_4, "x_test_9000_with_y_test_smpl_4.csv")
    # get_csvfile(test_smpl_5, "x_test_9000_with_y_test_smpl_5.csv")
    # get_csvfile(test_smpl_6, "x_test_9000_with_y_test_smpl_6.csv")
    # get_csvfile(test_smpl_7, "x_test_9000_with_y_test_smpl_7.csv")
    # get_csvfile(test_smpl_8, "x_test_9000_with_y_test_smpl_8.csv")
    # get_csvfile(test_smpl_9, "x_test_9000_with_y_test_smpl_9.csv")


if __name__ == "__main__":
    main()