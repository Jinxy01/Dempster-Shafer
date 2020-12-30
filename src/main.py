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
from utils.graph import *


def test_data():
    Y_train = tensor([1,0,1,0])
    X_train = [[0.2, 0.2], [0.3, -0.4], [0.3, 0.5], [-0.2, -0.9]]

    Y_train = one_hot(Y_train, num_classes=NUM_CLASSES).float()

    Y_test = [1,0,0]
    X_test = [[0.56, 0.1], [-0.3, -0.7], [0.4, -0.6]]

    return X_train, Y_train, X_test, Y_test

def evaluate_A1_dataset():
    # Variables
    graph_filepath = os.path.join(IMAGE_FOLDER, A1_LOSS_IMG)
    table_filepath = os.path.join(IMAGE_FOLDER, A1_RULE_TABLE)

    X_train, Y_train, X_test, Y_test = dataset_A1()
    #X_train, Y_train, X_test, Y_test = test_data()

    rule_set = generate_rules_dataset_A1()
    loss = MSE()

    # Training
    rule_set, it_loss = training(X_train, Y_train, rule_set, loss)

    # Inference
    accuracy, tot_correct_predicts, tot_predicts = inference(X_test, Y_test, rule_set)
    
    # Rules Table Drawing
    read_rules(rule_set)
    draw_rule_table(rule_set, table_filepath, accuracy, tot_correct_predicts, tot_predicts)

    # Loss drawing
    draw_loss(it_loss, graph_filepath)


def evaluate_breast_cancer_dataset():
    # Variables
    #graph_filepath = os.path.join(IMAGE_FOLDER, A1_LOSS_IMG)
    #table_filepath = os.path.join(IMAGE_FOLDER, A1_RULE_TABLE)

    X_train, Y_train, X_test, Y_test = dataset_breast_cancer()
    print(X_train[0])
    #X_train, Y_train, X_test, Y_test = test_data()

    # rule_set = generate_rules_dataset_A1()
    # loss = MSE()

    # # Training
    # rule_set, it_loss = training(X_train, Y_train, rule_set, loss)

    # # Inference
    # accuracy, tot_correct_predicts, tot_predicts = inference(X_test, Y_test, rule_set)
    
    # # Rules Table Drawing
    # read_rules(rule_set)
    # draw_rule_table(rule_set, table_filepath, accuracy, tot_correct_predicts, tot_predicts)

    # # Loss drawing
    # draw_loss(it_loss, graph_filepath)

if __name__ == "__main__":
    #evaluate_A1_dataset()
    evaluate_breast_cancer_dataset()
    