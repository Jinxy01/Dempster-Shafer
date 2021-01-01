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
from utils.bc_helper import *
from utils.common import *


def test_data():
    Y_train = tensor([1,0,1,0])
    #Y_train = [1,0,1,0]
    X_train = [[0.2, 0.8], [0.3, -0.4], [0.3, 0.5], [-0.2, -0.9]]

    Y_train = one_hot(Y_train, num_classes=NUM_CLASSES).float()

    Y_test = [1,0,0]
    X_test = [[0.56, 0.1], [-0.3, -0.7], [0.4, -0.6]]

    return X_train, Y_train, X_test, Y_test

def evaluate_A1_dataset(dataset_name):
    # Variables
    graph_filepath = os.path.join(IMAGE_FOLDER, A1_LOSS_IMG)
    table_filepath = os.path.join(IMAGE_FOLDER, A1_RULE_TABLE)

    X_train, Y_train, X_test, Y_test = dataset_A1()
    #X_train, Y_train, X_test, Y_test = test_data()

    rule_set, rule_presentation = generate_rules_dataset_A1(X_train, dataset_name)
    loss = MSE()

    # Training
    rule_set, it_loss = training(X_train, Y_train, rule_set, loss, dataset_name)

    # Inference
    accuracy, tot_correct_predicts, tot_predicts = model_evaluation(X_test, Y_test, rule_set, dataset_name)
    read_rules_A1(rule_set)
    
    # Rules Table Drawing
    draw_rule_table(rule_set, table_filepath, accuracy, tot_correct_predicts, tot_predicts, rule_presentation)

    # Loss drawing
    draw_loss(it_loss, graph_filepath)

def evaluate_breast_cancer_dataset(dataset_name):
    # Variables
    graph_filepath = os.path.join(IMAGE_FOLDER, BC_LOSS_IMG)
    table_filepath = os.path.join(IMAGE_FOLDER, BC_RULE_TABLE)

    X_train, Y_train, X_test, Y_test = dataset_breast_cancer()

    rule_set, rule_presentation = generate_rules_dataset_breast_cancer(X_train, dataset_name)

    loss = MSE()

    # Training
    rule_set, it_loss = training(X_train, Y_train, rule_set, loss, dataset_name)

    # Inference
    accuracy, tot_correct_predicts, tot_predicts = model_evaluation(X_test, Y_test, rule_set, dataset_name)
    
    # Rules Table Drawing
    read_rules_BC(rule_set)

    # Order rules by malignacy
    dict_rule_malig_sorted = order_rules_by_malign(rule_set, dataset_name)
    print(BC_RULE_PRESENTATION_TITLE)
    for k, v in dict_rule_malig_sorted.items():
        print(RULE_PRESENTATION_DISPLAY.format(k,v))

    print(RULE_TABLE_TITLE.format(accuracy, tot_correct_predicts, tot_predicts))

    #draw_rule_table(rule_set, table_filepath, accuracy, tot_correct_predicts, tot_predicts, rule_presentation)
    for i in range(len(rule_presentation)):
        print(RULE_PRESENTATION_DISPLAY.format(i+1, rule_presentation[i]))

    # Loss drawing
    draw_loss(it_loss, graph_filepath)

def evaluate_iris_dataset(dataset_name):
    # Variables
    graph_filepath = os.path.join(IMAGE_FOLDER, IRIS_LOSS_IMG)

    X_train, Y_train, X_test, Y_test = dataset_iris()

    rule_set, rule_presentation = generate_rules_dataset_iris(X_train, dataset_name)

    loss = MSE()

    # Training
    rule_set, it_loss = training(X_train, Y_train, rule_set, loss, dataset_name)

    # Inference
    accuracy, tot_correct_predicts, tot_predicts = model_evaluation(X_test, Y_test, rule_set, dataset_name)
    
    # Rules Table Drawing
    read_rules_iris(rule_set)

    # Exhibit accuracy
    print(RULE_TABLE_TITLE.format(accuracy, tot_correct_predicts, tot_predicts))

    #draw_rule_table(rule_set, table_filepath, accuracy, tot_correct_predicts, tot_predicts, rule_presentation)
    for i in range(len(rule_presentation)):
        print(RULE_PRESENTATION_DISPLAY.format(i+1, rule_presentation[i]))

    # Loss drawing
    draw_loss(it_loss, graph_filepath)

def evaluate_heart_disease_dataset(dataset_name):
    # Variables
    graph_filepath = os.path.join(IMAGE_FOLDER, HD_LOSS_IMG)

    X_train, Y_train, X_test, Y_test = dataset_heart_disease()

    rule_set, rule_presentation = generate_rules_dataset_heart_disease(X_train, dataset_name)

    loss = MSE()

    # Training
    rule_set, it_loss = training(X_train, Y_train, rule_set, loss, dataset_name)

    # Inference
    accuracy, tot_correct_predicts, tot_predicts = model_evaluation(X_test, Y_test, rule_set, dataset_name)
    
    # Rules Table Drawing
    read_rules_heart_disease(rule_set)

    # Order by malignacy
    dict_rule_malig_sorted = order_rules_by_malign(rule_set, dataset_name)
    print(BC_RULE_PRESENTATION_TITLE)
    for k, v in dict_rule_malig_sorted.items():
        print(RULE_PRESENTATION_DISPLAY.format(k,v))

    # Exhibit accuracy
    print(RULE_TABLE_TITLE.format(accuracy, tot_correct_predicts, tot_predicts))

    #draw_rule_table(rule_set, table_filepath, accuracy, tot_correct_predicts, tot_predicts, rule_presentation)
    for i in range(len(rule_presentation)):
        print(RULE_PRESENTATION_DISPLAY.format(i+1, rule_presentation[i]))

    # Loss drawing
    draw_loss(it_loss, graph_filepath)

def evaluate_wine_dataset(dataset_name):
    # Variables
    graph_filepath = os.path.join(IMAGE_FOLDER, WINE_LOSS_IMG)

    X_train, Y_train, X_test, Y_test = dataset_wine()

    rule_set, rule_presentation = generate_rules_dataset_wine(X_train, dataset_name)

    loss = MSE()

    # Training
    rule_set, it_loss = training(X_train, Y_train, rule_set, loss, dataset_name)

    # Inference
    accuracy, tot_correct_predicts, tot_predicts = model_evaluation(X_test, Y_test, rule_set, dataset_name)
    
    # Rules Table Drawing
    read_rules_wine(rule_set)

    # Exhibit accuracy
    print(RULE_TABLE_TITLE.format(accuracy, tot_correct_predicts, tot_predicts))

    #draw_rule_table(rule_set, table_filepath, accuracy, tot_correct_predicts, tot_predicts, rule_presentation)
    for i in range(len(rule_presentation)):
        print(RULE_PRESENTATION_DISPLAY.format(i+1, rule_presentation[i]))

    # Loss drawing
    draw_loss(it_loss, graph_filepath)

if __name__ == "__main__":
    #evaluate_A1_dataset("A1_Dataset")
    #evaluate_breast_cancer_dataset("BC_Dataset")
    #evaluate_iris_dataset("IRIS_Dataset")
    #evaluate_heart_disease_dataset("HD_Dataset")
    evaluate_wine_dataset("WINE_Dataset")
