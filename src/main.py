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
    Y = tensor([1,0,1,0])
    X = [[0.2, 0.2], [0.3, -0.4], [0.3, 0.5], [-0.2, -0.9]]

    Y = one_hot(Y, num_classes=NUM_CLASSES).float()

    return X, Y


if __name__ == "__main__":
    # X_train, Y_train, X_test, Y_test = dataset_A1()
    X_train, Y_train = test_data()

    rule_set = generate_rules_dataset_A1()
    loss = MSE()

    training(X_train, Y_train, rule_set, loss)
    