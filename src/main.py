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
from utils.dataset_functions import  *

def aid_test():
    dataset_filepath = os.path.join(DATASET_FOLDER, A1_DATASET_FILE)
    X, Y = read_dataset(dataset_filepath)
    X_train, Y_train, X_test, Y_test = split_test_train(X,Y)
    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    aid_test()
    