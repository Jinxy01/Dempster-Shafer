"""
@author: Tiago Roxo, UBI
@date: 2020
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
import csv
import numpy as np

from utils.config import *
from utils.aux_function import *
from torch.nn.functional import one_hot

def read_dataset(dataset_filepath):
    with open(dataset_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        X = []
        Y = []
        next(csv_reader) # to skip the header file
        for x,y,c in csv_reader:
            X.append([float(x), float(y)])
            Y.append(int(c))

    X = np.asarray(X).astype(float)
    Y = np.asarray(Y).astype(float)
    return X, Y

def split_test_train(X,Y):
    # Split between train and test (70%/30%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_PERCENTAGE) 
    return X_train, Y_train, X_test, Y_test

def generate_rules_dataset_A1():
    s_list = [lambda x,y: y > 0, lambda x,y: y <= 0,lambda x,y: x != 0]
    rule_set = start_weights(s_list) 
    return rule_set


def dataset_A1():
    dataset_filepath = os.path.join(DATASET_FOLDER, A1_DATASET_FILE)
    X, Y = read_dataset(dataset_filepath)
    X_train, Y_train, X_test, Y_test = split_test_train(X,Y)

    # Pre process
    Y_train = tensor(Y_train).to(torch.int64)
    Y_train = one_hot(Y_train, num_classes=NUM_CLASSES).float()

    return X_train, Y_train, X_test, Y_test