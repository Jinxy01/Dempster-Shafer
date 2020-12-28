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
from torch import tensor
from torch.nn.functional import one_hot
from torch.nn import MSELoss as MSE

from utils.config import *
from utils.dataset_functions import *
from src.model import *



def test_data():
    Y_train = tensor([1,0,1,0])
    X_train = [[0.2, 0.2], [0.3, -0.4], [0.3, 0.5], [-0.2, -0.9]]

    Y_train = one_hot(Y_train, num_classes=NUM_CLASSES).float()

    Y_test = [1,0,0]
    X_test = [[0.56, 0.1], [-0.3, -0.7], [0.4, -0.6]]

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    # X_train, Y_train, X_test, Y_test = dataset_A1()
    X_train, Y_train, X_test, Y_test = test_data()

    rule_set = generate_rules_dataset_A1()
    loss = MSE()

    rule_set, it_loss = training(X_train, Y_train, rule_set, loss)
    read_rules(rule_set)

    accuracy = inference(X_test, Y_test, rule_set)
    print(accuracy)