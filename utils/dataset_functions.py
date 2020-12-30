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
import pandas as pd

from utils.config import *
from utils.aux_function import *
from torch.nn.functional import one_hot

import utils.a1_helper as a1

# ------------- A1 Dataset ---------------------

def read_dataset_A1(dataset_filepath):
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

def generate_rules_dataset_A1(X_train):
    # s_list = [
    #     lambda x,y: y > 0, 
    #     lambda x,y: y <= 0,
    #     lambda x,y: x != 0
    # ]
    [x_mean, y_mean] = np.mean(X_train, axis=0) # mean along columns
    [x_std,  y_std]  = np.std(X_train, axis=0, dtype=np.float64) # std along columns

    # s_list = [
    #     lambda x,y: x <= x_mean-x_std, 
    #     lambda x,y: x_mean-x_std < x and x <= x_mean,
    #     lambda x,y: x_mean < x and x <= x_mean+x_std,
    #     lambda x,y: x > x_mean+x_std, 
    #     lambda x,y: y <= y_mean-y_std, 
    #     lambda x,y: y_mean-y_std < y and y <= y_mean,
    #     lambda x,y: y_mean < y and y <= y_mean+y_std,
    #     lambda x,y: y > y_mean+y_std, 
    # ]

    rules_x = a1.generate_rule_A1_helper_x(x_mean, x_std)
    rules_y = a1.generate_rule_A1_helper_y(y_mean, y_std)
    s_list = rules_x + rules_y

    # Author rules
    # s_list = [
    #     lambda x,y: x <= -0.32, 
    #     lambda x,y: -0.32 < x and x <= 0.04,
    #     lambda x,y: 0.04 < x and x <= 0.41,
    #     lambda x,y: x > 0.41, 
    #     lambda x,y: y <= -0.34, 
    #     lambda x,y: -0.34 < y and y <= 0.04,
    #     lambda x,y: 0.04 < y and y <= 0.42,
    #     lambda x,y: y > 0.42, 
    # ]

    rule_set = start_weights(s_list) 

    # Aid in result presentation
    x_rules_presentation = presentation_rule_helper("x", x_mean, x_std)
    y_rules_presentation = presentation_rule_helper("y", y_mean, y_std)

    rule_presentation = x_rules_presentation + y_rules_presentation

    return rule_set, rule_presentation


def dataset_A1():
    dataset_filepath = os.path.join(DATASET_FOLDER, A1_DATASET_FILE)
    X, Y = read_dataset_A1(dataset_filepath)
    X_train, Y_train, X_test, Y_test = split_test_train(X,Y)

    # Pre process
    Y_train = tensor(Y_train).to(torch.int64)
    Y_train = one_hot(Y_train, num_classes=A1_NUM_CLASSES).float()

    return X_train, Y_train, X_test, Y_test

# ------------- Breast Cancer ---------------------

def preprocess_dataset_breast_cancer(dataset_filepath, processed_dataset_filepath):
    columns = ["scn", "ct", "ucsize", "ucshape", "ma", "secz", "bn", "bc", "nn", "m", "y"]
    df = pd.read_csv(dataset_filepath, usecols=columns, na_values='?')
    for column in columns:
        df[column] = df[column].fillna(value=df[column].mean())
    
    # Change classes to 0 and 1 (was 2 for benign and 4 for malignant)
    df.loc[df.y == 2, 'y'] = 0
    df.loc[df.y == 4, 'y'] = 1
    df.to_csv(processed_dataset_filepath, index=False)


def read_dataset_breast_cancer(dataset_filepath):

    with open(dataset_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        X = []
        Y = []

        next(csv_reader) # to skip the header file

           #  Attribute                     Domain
        # -- -----------------------------------------
        # 1. Sample code number            id number
        # 2. Clump Thickness               1 - 10
        # 3. Uniformity of Cell Size       1 - 10
        # 4. Uniformity of Cell Shape      1 - 10
        # 5. Marginal Adhesion             1 - 10
        # 6. Single Epithelial Cell Size   1 - 10
        # 7. Bare Nuclei                   1 - 10
        # 8. Bland Chromatin               1 - 10
        # 9. Normal Nucleoli               1 - 10
        # 10. Mitoses                      1 - 10
        # 11. Class:                       (2 for benign, 4 for malignant)


        for _, ct, ucsize, ucshape, ma, secz, bn, bc, nn, m, y in csv_reader:
            X.append([float(ct), float(ucsize), float(ucshape), float(ma), 
                float(secz), float(bn), float(bc), float(nn), float(m)])
            Y.append(int(y))

    X = np.asarray(X).astype(float)
    Y = np.asarray(Y).astype(float)
    return X, Y

def generate_rules_dataset_breast_cancer(X_train):

    [_, ct, ucsize, ucshape, ma, secz, bn, bc, nn, m, y] = np.mean(X_train, axis=0) # mean along columns
    [_, ct, ucsize, ucshape, ma, secz, bn, bc, nn, m, y] = np.std(X_train, axis=0, dtype=np.float64) # std along columns

    s_list = [
        lambda x,y: x <= x_mean-x_std, 
        lambda x,y: x_mean-x_std < x and x <= x_mean,
        lambda x,y: x_mean < x and x <= x_mean+x_std,
        lambda x,y: x > x_mean+x_std, 
        lambda x,y: y <= y_mean-y_std, 
        lambda x,y: y_mean-y_std < y and y <= y_mean,
        lambda x,y: y_mean < y and y <= y_mean+y_std,
        lambda x,y: y > y_mean+y_std, 
    ]

    rule_set = start_weights(s_list) 

    # Aid in result presentation
    x_rules_presentation = presentation_rule_helper("x", x_mean, x_std)
    y_rules_presentation = presentation_rule_helper("y", y_mean, y_std)

    rule_presentation = x_rules_presentation + y_rules_presentation

    return rule_set, rule_presentation


def dataset_breast_cancer():
    dataset_filepath           = os.path.join(DATASET_FOLDER, BC_DATASET_FILE)
    processed_dataset_filepath = os.path.join(DATASET_FOLDER, BC_PROCESSED_DATASET_FILE)

    preprocess_dataset_breast_cancer(dataset_filepath, processed_dataset_filepath)

    X, Y = read_dataset_breast_cancer(processed_dataset_filepath)
    X_train, Y_train, X_test, Y_test = split_test_train(X,Y)

    # Pre process
    Y_train = tensor(Y_train).to(torch.int64)
    Y_train = one_hot(Y_train, num_classes=BC_NUM_CLASSES).float()

    return X_train, Y_train, X_test, Y_test



# ------------ Common ------------------------

def split_test_train(X,Y):
    # Split between train and test (70%/30%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_PERCENTAGE) 
    return X_train, Y_train, X_test, Y_test


def presentation_rule_helper(element, mean, std):
    return [
        RULE_LTE.format(element, mean-std),
        RULE_BETWEEN.format(mean-std, element, mean),
        RULE_BETWEEN.format(mean, element, mean+std),
        RULE_GT.format(element, mean+std)
    ]