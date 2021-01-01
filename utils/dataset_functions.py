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

#from utils.config import *
from utils.common import *
from torch.nn.functional import one_hot

import utils.a1_helper as a1
import utils.bc_helper as bc

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

def generate_rules_dataset_A1(X_train, dataset_name):
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

    # Create rules
    #rules_x = a1.generate_rule_x(x_mean, x_std)
    #rules_y = a1.generate_rule_y(y_mean, y_std)
    rules_x = generate_rule(0, x_mean, x_std)
    rules_y = generate_rule(1, x_mean, x_std)
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

    rule_set = start_weights(s_list, dataset_name) 

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

def generate_rules_dataset_breast_cancer(X_train, dataset_name):

    [ct_mean, ucsize_mean, ucshape_mean, ma_mean, secz_mean, bn_mean, 
        bc_mean, nn_mean, m_mean] = np.mean(X_train, axis=0) # mean along columns
    [ct_std,  ucsize_std,  ucshape_std,  ma_std,  secz_std,  bn_std, 
        bc_std,  nn_std,  m_std] = np.std(X_train, axis=0, dtype=np.float64) # std along columns

    # Create rules
    rules_ct_1std      = bc.generate_rule_ct(ct_mean, ct_std)
    rules_ct_2std      = bc.generate_rule_ct(ct_mean-2*ct_std, ct_std)
    rules_ct_3std      = bc.generate_rule_ct(ct_mean+2*ct_std, ct_std)

    rules_ucsize_1std  = bc.generate_rule_ucsize(ucsize_mean, ucsize_std)
    rules_ucsize_2std  = bc.generate_rule_ucsize(ucsize_mean-2*ucsize_std, ucsize_std)
    rules_ucsize_3std  = bc.generate_rule_ucsize(ucsize_mean+2*ucsize_std, ucsize_std)

    rules_ucshape_1std = bc.generate_rule_ucshape(ucshape_mean, ucshape_std)
    rules_ucshape_2std = bc.generate_rule_ucshape(ucshape_mean-2*ucshape_std, ucshape_std)
    rules_ucshape_3std = bc.generate_rule_ucshape(ucshape_mean+2*ucshape_std, ucshape_std)

    rules_ma_1std      = bc.generate_rule_ma(ma_mean, ma_std)
    rules_ma_2std      = bc.generate_rule_ma(ma_mean-2*ma_std, ma_std)
    rules_ma_3std      = bc.generate_rule_ma(ma_mean+2*ma_std, ma_std)

    rules_secz_1std    = bc.generate_rule_secz(secz_mean, secz_std)
    rules_secz_2std    = bc.generate_rule_secz(secz_mean-2*secz_std, secz_std)
    rules_secz_3std    = bc.generate_rule_secz(secz_mean+2*secz_std, secz_std)

    rules_bn_1std      = bc.generate_rule_bn(bn_mean, bn_std)
    rules_bn_2std      = bc.generate_rule_bn(bn_mean-2*bn_std, bn_std)
    rules_bn_3std      = bc.generate_rule_bn(bn_mean+2*bn_std, bn_std)

    rules_bc_1std      = bc.generate_rule_bc(bc_mean, bc_std)
    rules_bc_2std      = bc.generate_rule_bc(bc_mean-2*bc_std, bc_std)
    rules_bc_3std      = bc.generate_rule_bc(bc_mean+2*bc_std, bc_std)

    rules_nn_1std      = bc.generate_rule_nn(nn_mean, nn_std)
    rules_nn_2std      = bc.generate_rule_nn(nn_mean-2*nn_std, nn_std)
    rules_nn_3std      = bc.generate_rule_nn(nn_mean+2*nn_std, nn_std)

    rules_m_1std       = bc.generate_rule_m(m_mean, m_std)
    rules_m_2std       = bc.generate_rule_m(m_mean-2*m_std, m_std)
    rules_m_3std       = bc.generate_rule_m(m_mean+2*m_std, m_std)

    s_list = []
    s_list += rules_ct_1std + rules_ct_2std + rules_ct_3std 
    s_list += rules_ucsize_1std + rules_ucsize_2std + rules_ucsize_3std
    s_list += rules_ucshape_1std + rules_ucshape_2std + rules_ucshape_3std 
    s_list += rules_ma_1std + rules_ma_2std + rules_ma_3std 
    s_list += rules_secz_1std + rules_secz_2std + rules_secz_3std  
    s_list += rules_bn_1std + rules_bn_2std + rules_bn_3std 
    s_list += rules_bc_1std + rules_bc_2std + rules_bc_3std  
    s_list += rules_nn_1std + rules_nn_2std + rules_nn_3std 
    s_list += rules_m_1std + rules_m_2std + rules_m_3std

    # # Create rules
    # rules_ct_1std      = bc.generate_rule_ct(ct_mean, ct_std)
    # #rules_ct_2std      = bc.generate_rule_ct(ct_mean, 2*ct_std)
    # #rules_ct_3std      = bc.generate_rule_ct(ct_mean, 3*ct_std)

    # rules_ucsize_1std  = bc.generate_rule_ucsize(ucsize_mean, ucsize_std)
    # #rules_ucsize_2std  = bc.generate_rule_ucsize(ucsize_mean, 2*ucsize_std)
    # #rules_ucsize_3std  = bc.generate_rule_ucsize(ucsize_mean, 3*ucsize_std)

    # rules_ucshape_1std = bc.generate_rule_ucshape(ucshape_mean, ucshape_std)
    # #rules_ucshape_2std = bc.generate_rule_ucshape(ucshape_mean, 2*ucshape_std)
    # #rules_ucshape_3std = bc.generate_rule_ucshape(ucshape_mean, 3*ucshape_std)

    # rules_ma_1std      = bc.generate_rule_ma(ma_mean, ma_std)
    # #rules_ma_2std      = bc.generate_rule_ma(ma_mean, 2*ma_std)
    # #rules_ma_3std      = bc.generate_rule_ma(ma_mean, 3*ma_std)

    # rules_secz_1std    = bc.generate_rule_secz(secz_mean, secz_std)
    # #rules_secz_2std    = bc.generate_rule_secz(secz_mean, 2*secz_std)
    # #rules_secz_3std    = bc.generate_rule_secz(secz_mean, 3*secz_std)

    # rules_bn_1std      = bc.generate_rule_bn(bn_mean, bn_std)
    # #rules_bn_2std      = bc.generate_rule_bn(bn_mean, 2*bn_std)
    # #rules_bn_3std      = bc.generate_rule_bn(bn_mean, 3*bn_std)

    # rules_bc_1std      = bc.generate_rule_bc(bc_mean, bc_std)
    # #rules_bc_2std      = bc.generate_rule_bc(bc_mean, 2*bc_std)
    # #rules_bc_3std      = bc.generate_rule_bc(bc_mean, 3*bc_std)

    # rules_nn_1std      = bc.generate_rule_nn(nn_mean, nn_std)
    # #rules_nn_2std      = bc.generate_rule_nn(nn_mean, 2*nn_std)
    # #rules_nn_3std      = bc.generate_rule_nn(nn_mean, 3*nn_std)

    # rules_m_1std       = bc.generate_rule_m(m_mean, m_std)
    # #rules_m_2std       = bc.generate_rule_m(m_mean, 2*m_std)
    # #rules_m_3std       = bc.generate_rule_m(m_mean, 3*m_std)

    # s_list = []
    # s_list += rules_ct_1std 
    # s_list += rules_ucsize_1std 
    # s_list += rules_ucshape_1std 
    # s_list += rules_ma_1std 
    # s_list += rules_secz_1std 
    # s_list += rules_bn_1std 
    # s_list += rules_bc_1std  
    # s_list += rules_nn_1std 
    # s_list += rules_m_1std

    rule_set = start_weights(s_list, dataset_name)

    # # Aid in result presentation
    # ct_rules_presentation = presentation_rule_helper("ct", ct_mean, ct_std)
    # ucsize_rules_presentation = presentation_rule_helper("ucsize", ucsize_mean, ucsize_std)
    # ucshape_rules_presentation = presentation_rule_helper("ucshape", ucshape_mean, ucshape_std)
    # ma_rules_presentation = presentation_rule_helper("ma", ma_mean, ma_std)
    # secz_rules_presentation = presentation_rule_helper("secz", secz_mean, secz_std)
    # bn_rules_presentation = presentation_rule_helper("bn", bn_mean, bn_std)
    # bc_rules_presentation = presentation_rule_helper("bc", bc_mean, bc_std)
    # nn_rules_presentation = presentation_rule_helper("nn", nn_mean, nn_std)
    # m_rules_presentation = presentation_rule_helper("m", m_mean, m_std)

    # rule_presentation  = ct_rules_presentation + ucsize_rules_presentation + ucshape_rules_presentation
    # rule_presentation += ma_rules_presentation + secz_rules_presentation + bn_rules_presentation
    # rule_presentation += bc_rules_presentation + nn_rules_presentation + m_rules_presentation

    # Aid in result presentation
    ct_1_rules_presentation = presentation_rule_helper("ct", ct_mean, ct_std)
    ct_2_rules_presentation = presentation_rule_helper("ct", ct_mean-2*ct_std, ct_std)
    ct_3_rules_presentation = presentation_rule_helper("ct", ct_mean+2*ct_std, ct_std)

    ucsize_1_rules_presentation = presentation_rule_helper("ucsize", ucsize_mean, ucsize_std)
    ucsize_2_rules_presentation = presentation_rule_helper("ucsize", ucsize_mean-2*ucsize_std, ucsize_std)
    ucsize_3_rules_presentation = presentation_rule_helper("ucsize", ucsize_mean+2*ucsize_std, ucsize_std)

    ucshape_1_rules_presentation = presentation_rule_helper("ucshape", ucshape_mean, ucshape_std)
    ucshape_2_rules_presentation = presentation_rule_helper("ucshape", ucshape_mean-2*ucshape_std, ucshape_std)
    ucshape_3_rules_presentation = presentation_rule_helper("ucshape", ucshape_mean+2*ucshape_std, ucshape_std)

    ma_rules_1_presentation = presentation_rule_helper("ma", ma_mean, ma_std)
    ma_rules_2_presentation = presentation_rule_helper("ma", ma_mean-2*ma_std, ma_std)
    ma_rules_3_presentation = presentation_rule_helper("ma", ma_mean+2*ma_std, ma_std)

    secz_rules_1_presentation = presentation_rule_helper("secz", secz_mean, secz_std)
    secz_rules_2_presentation = presentation_rule_helper("secz", secz_mean-2*secz_std, secz_std)
    secz_rules_3_presentation = presentation_rule_helper("secz", secz_mean+2*secz_std, secz_std)

    bn_rules_1_presentation = presentation_rule_helper("bn", bn_mean, bn_std)
    bn_rules_2_presentation = presentation_rule_helper("bn", bn_mean-2*bn_std, bn_std)
    bn_rules_3_presentation = presentation_rule_helper("bn", bn_mean+2*bn_std, bn_std)

    bc_rules_1_presentation = presentation_rule_helper("bc", bc_mean, bc_std)
    bc_rules_2_presentation = presentation_rule_helper("bc", bc_mean-2*bc_std, bc_std)
    bc_rules_3_presentation = presentation_rule_helper("bc", bc_mean+2*bc_std, bc_std)

    nn_rules_1_presentation = presentation_rule_helper("nn", nn_mean, nn_std)
    nn_rules_2_presentation = presentation_rule_helper("nn", nn_mean-2*nn_std, nn_std)
    nn_rules_3_presentation = presentation_rule_helper("nn", nn_mean+2*nn_std, nn_std)

    m_rules_1_presentation = presentation_rule_helper("m", m_mean, m_std)
    m_rules_2_presentation = presentation_rule_helper("m", m_mean-2*m_std, m_std)
    m_rules_3_presentation = presentation_rule_helper("m", m_mean+2*m_std, m_std)


    rule_presentation  = ct_1_rules_presentation + ct_2_rules_presentation + ct_3_rules_presentation
    rule_presentation += ucsize_1_rules_presentation + ucsize_2_rules_presentation + ucsize_3_rules_presentation
    rule_presentation += ucshape_1_rules_presentation + ucshape_2_rules_presentation + ucshape_3_rules_presentation
    rule_presentation += ma_rules_1_presentation + ma_rules_2_presentation + ma_rules_3_presentation
    rule_presentation += secz_rules_1_presentation + secz_rules_2_presentation + secz_rules_3_presentation
    rule_presentation += bn_rules_1_presentation + bn_rules_2_presentation + bn_rules_3_presentation
    rule_presentation += bc_rules_1_presentation + bc_rules_2_presentation + bc_rules_3_presentation
    rule_presentation += nn_rules_1_presentation + nn_rules_2_presentation + nn_rules_3_presentation
    rule_presentation += m_rules_1_presentation + m_rules_2_presentation + m_rules_3_presentation

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

# ------------- Iris ---------------------

def preprocess_dataset_iris(dataset_filepath, processed_dataset_filepath):
    columns = ["sl", "sw", "pl", "pw", "y"]
   
    # Change classes to 0, 1 and 2
    df = pd.read_csv(dataset_filepath, usecols=columns)
    df.loc[df.y == 'Iris-setosa', 'y']      = 0
    df.loc[df.y == 'Iris-versicolor', 'y'] = 1
    df.loc[df.y == 'Iris-virginica', 'y']   = 2
    df.to_csv(processed_dataset_filepath, index=False)


def read_dataset_iris(dataset_filepath):

    with open(dataset_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        X = []
        Y = []

        next(csv_reader) # to skip the header file

        #     Attribute                     Domain
        #     1. sepal length in cm
        #     2. sepal width in cm
        #     3. petal length in cm
        #     4. petal width in cm
        #     5. class: 
        #         -- Iris Setosa
        #         -- Iris Versicolour
        #         -- Iris Virginica

        for sl, sw, pl, pw, y in csv_reader:
            X.append([float(sl), float(sw), float(pl), float(pw)])
            Y.append(int(y))

    X = np.asarray(X).astype(float)
    Y = np.asarray(Y).astype(float)
    return X, Y

def generate_rules_dataset_iris(X_train, dataset_name):

    [sl_mean, sw_mean, pl_mean, pw_mean] = np.mean(X_train, axis=0) # mean along columns
    [sl_std,  sw_std,  pl_std,  pw_std]  = np.std(X_train, axis=0, dtype=np.float64) # std along columns

    # Create rules
    rules_sl = generate_rule(0, sl_mean, sl_std)
    rules_sw = generate_rule(1, sw_mean, sw_std)
    rules_pl = generate_rule(2, pl_mean, pl_std)
    rules_pw = generate_rule(3, pw_mean, pw_std)

    s_list = rules_sl + rules_sw + rules_pl + rules_pw

    rule_set = start_weights(s_list, dataset_name)

    # Aid in result presentation
    sl_rules_presentation = presentation_rule_helper("sl", sl_mean, sl_std)
    sw_rules_presentation = presentation_rule_helper("sw", sw_mean, sw_std)
    pl_rules_presentation = presentation_rule_helper("pl", pl_mean, pl_std)
    pw_rules_presentation = presentation_rule_helper("pw", pw_mean, pw_std)

    rule_presentation  = sl_rules_presentation + sw_rules_presentation
    rule_presentation += pl_rules_presentation + pw_rules_presentation

    return rule_set, rule_presentation


def dataset_iris():
    dataset_filepath           = os.path.join(DATASET_FOLDER, IRIS_DATASET_FILE)
    processed_dataset_filepath = os.path.join(DATASET_FOLDER, IRIS_PROCESSED_DATASET_FILE)

    preprocess_dataset_iris(dataset_filepath, processed_dataset_filepath)

    X, Y = read_dataset_iris(processed_dataset_filepath)
    X_train, Y_train, X_test, Y_test = split_test_train(X,Y)

    # Pre process
    Y_train = tensor(Y_train).to(torch.int64)
    Y_train = one_hot(Y_train, num_classes=IRIS_NUM_CLASSES).float()

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