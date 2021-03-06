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

def read_rules_A1(rule_set):
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        b   = dict_m[frozenset({'B'})].item()
        r   = dict_m[frozenset({'R'})].item()
        r_b = dict_m[frozenset({'B', 'R'})].item()
        print(A1_RULE_PRESENT.format(i+1,b,r,r_b))

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

    # # Create rules
    # rules_ct_1std      = bc.generate_rule_ct(ct_mean, ct_std)
    # rules_ct_2std      = bc.generate_rule_ct(ct_mean-2*ct_std, ct_std)
    # rules_ct_3std      = bc.generate_rule_ct(ct_mean+2*ct_std, ct_std)

    # rules_ucsize_1std  = bc.generate_rule_ucsize(ucsize_mean, ucsize_std)
    # rules_ucsize_2std  = bc.generate_rule_ucsize(ucsize_mean-2*ucsize_std, ucsize_std)
    # rules_ucsize_3std  = bc.generate_rule_ucsize(ucsize_mean+2*ucsize_std, ucsize_std)

    # rules_ucshape_1std = bc.generate_rule_ucshape(ucshape_mean, ucshape_std)
    # rules_ucshape_2std = bc.generate_rule_ucshape(ucshape_mean-2*ucshape_std, ucshape_std)
    # rules_ucshape_3std = bc.generate_rule_ucshape(ucshape_mean+2*ucshape_std, ucshape_std)

    # rules_ma_1std      = bc.generate_rule_ma(ma_mean, ma_std)
    # rules_ma_2std      = bc.generate_rule_ma(ma_mean-2*ma_std, ma_std)
    # rules_ma_3std      = bc.generate_rule_ma(ma_mean+2*ma_std, ma_std)

    # rules_secz_1std    = bc.generate_rule_secz(secz_mean, secz_std)
    # rules_secz_2std    = bc.generate_rule_secz(secz_mean-2*secz_std, secz_std)
    # rules_secz_3std    = bc.generate_rule_secz(secz_mean+2*secz_std, secz_std)

    # rules_bn_1std      = bc.generate_rule_bn(bn_mean, bn_std)
    # rules_bn_2std      = bc.generate_rule_bn(bn_mean-2*bn_std, bn_std)
    # rules_bn_3std      = bc.generate_rule_bn(bn_mean+2*bn_std, bn_std)

    # rules_bc_1std      = bc.generate_rule_bc(bc_mean, bc_std)
    # rules_bc_2std      = bc.generate_rule_bc(bc_mean-2*bc_std, bc_std)
    # rules_bc_3std      = bc.generate_rule_bc(bc_mean+2*bc_std, bc_std)

    # rules_nn_1std      = bc.generate_rule_nn(nn_mean, nn_std)
    # rules_nn_2std      = bc.generate_rule_nn(nn_mean-2*nn_std, nn_std)
    # rules_nn_3std      = bc.generate_rule_nn(nn_mean+2*nn_std, nn_std)

    # rules_m_1std       = bc.generate_rule_m(m_mean, m_std)
    # rules_m_2std       = bc.generate_rule_m(m_mean-2*m_std, m_std)
    # rules_m_3std       = bc.generate_rule_m(m_mean+2*m_std, m_std)

    # s_list = []
    # s_list += rules_ct_1std + rules_ct_2std + rules_ct_3std 
    # s_list += rules_ucsize_1std + rules_ucsize_2std + rules_ucsize_3std
    # s_list += rules_ucshape_1std + rules_ucshape_2std + rules_ucshape_3std 
    # s_list += rules_ma_1std + rules_ma_2std + rules_ma_3std 
    # s_list += rules_secz_1std + rules_secz_2std + rules_secz_3std  
    # s_list += rules_bn_1std + rules_bn_2std + rules_bn_3std 
    # s_list += rules_bc_1std + rules_bc_2std + rules_bc_3std  
    # s_list += rules_nn_1std + rules_nn_2std + rules_nn_3std 
    # s_list += rules_m_1std + rules_m_2std + rules_m_3std

    # Create rules
    rules_ct_1std      = bc.generate_rule_ct(ct_mean, ct_std)
    #rules_ct_2std      = bc.generate_rule_ct(ct_mean, 2*ct_std)
    #rules_ct_3std      = bc.generate_rule_ct(ct_mean, 3*ct_std)

    rules_ucsize_1std  = bc.generate_rule_ucsize(ucsize_mean, ucsize_std)
    #rules_ucsize_2std  = bc.generate_rule_ucsize(ucsize_mean, 2*ucsize_std)
    #rules_ucsize_3std  = bc.generate_rule_ucsize(ucsize_mean, 3*ucsize_std)

    rules_ucshape_1std = bc.generate_rule_ucshape(ucshape_mean, ucshape_std)
    #rules_ucshape_2std = bc.generate_rule_ucshape(ucshape_mean, 2*ucshape_std)
    #rules_ucshape_3std = bc.generate_rule_ucshape(ucshape_mean, 3*ucshape_std)

    rules_ma_1std      = bc.generate_rule_ma(ma_mean, ma_std)
    #rules_ma_2std      = bc.generate_rule_ma(ma_mean, 2*ma_std)
    #rules_ma_3std      = bc.generate_rule_ma(ma_mean, 3*ma_std)

    rules_secz_1std    = bc.generate_rule_secz(secz_mean, secz_std)
    #rules_secz_2std    = bc.generate_rule_secz(secz_mean, 2*secz_std)
    #rules_secz_3std    = bc.generate_rule_secz(secz_mean, 3*secz_std)

    rules_bn_1std      = bc.generate_rule_bn(bn_mean, bn_std)
    #rules_bn_2std      = bc.generate_rule_bn(bn_mean, 2*bn_std)
    #rules_bn_3std      = bc.generate_rule_bn(bn_mean, 3*bn_std)

    rules_bc_1std      = bc.generate_rule_bc(bc_mean, bc_std)
    #rules_bc_2std      = bc.generate_rule_bc(bc_mean, 2*bc_std)
    #rules_bc_3std      = bc.generate_rule_bc(bc_mean, 3*bc_std)

    rules_nn_1std      = bc.generate_rule_nn(nn_mean, nn_std)
    #rules_nn_2std      = bc.generate_rule_nn(nn_mean, 2*nn_std)
    #rules_nn_3std      = bc.generate_rule_nn(nn_mean, 3*nn_std)

    rules_m_1std       = bc.generate_rule_m(m_mean, m_std)
    #rules_m_2std       = bc.generate_rule_m(m_mean, 2*m_std)
    #rules_m_3std       = bc.generate_rule_m(m_mean, 3*m_std)

    s_list = []
    s_list += rules_ct_1std 
    s_list += rules_ucsize_1std 
    s_list += rules_ucshape_1std 
    s_list += rules_ma_1std 
    s_list += rules_secz_1std 
    s_list += rules_bn_1std 
    s_list += rules_bc_1std  
    s_list += rules_nn_1std 
    s_list += rules_m_1std

    rule_set = start_weights(s_list, dataset_name)

    # Aid in result presentation
    ct_rules_presentation = presentation_rule_helper("ct", ct_mean, ct_std)
    ucsize_rules_presentation = presentation_rule_helper("ucsize", ucsize_mean, ucsize_std)
    ucshape_rules_presentation = presentation_rule_helper("ucshape", ucshape_mean, ucshape_std)
    ma_rules_presentation = presentation_rule_helper("ma", ma_mean, ma_std)
    secz_rules_presentation = presentation_rule_helper("secz", secz_mean, secz_std)
    bn_rules_presentation = presentation_rule_helper("bn", bn_mean, bn_std)
    bc_rules_presentation = presentation_rule_helper("bc", bc_mean, bc_std)
    nn_rules_presentation = presentation_rule_helper("nn", nn_mean, nn_std)
    m_rules_presentation = presentation_rule_helper("m", m_mean, m_std)

    rule_presentation  = ct_rules_presentation + ucsize_rules_presentation + ucshape_rules_presentation
    rule_presentation += ma_rules_presentation + secz_rules_presentation + bn_rules_presentation
    rule_presentation += bc_rules_presentation + nn_rules_presentation + m_rules_presentation

    # # Aid in result presentation
    # ct_1_rules_presentation = presentation_rule_helper("ct", ct_mean, ct_std)
    # ct_2_rules_presentation = presentation_rule_helper("ct", ct_mean-2*ct_std, ct_std)
    # ct_3_rules_presentation = presentation_rule_helper("ct", ct_mean+2*ct_std, ct_std)

    # ucsize_1_rules_presentation = presentation_rule_helper("ucsize", ucsize_mean, ucsize_std)
    # ucsize_2_rules_presentation = presentation_rule_helper("ucsize", ucsize_mean-2*ucsize_std, ucsize_std)
    # ucsize_3_rules_presentation = presentation_rule_helper("ucsize", ucsize_mean+2*ucsize_std, ucsize_std)

    # ucshape_1_rules_presentation = presentation_rule_helper("ucshape", ucshape_mean, ucshape_std)
    # ucshape_2_rules_presentation = presentation_rule_helper("ucshape", ucshape_mean-2*ucshape_std, ucshape_std)
    # ucshape_3_rules_presentation = presentation_rule_helper("ucshape", ucshape_mean+2*ucshape_std, ucshape_std)

    # ma_rules_1_presentation = presentation_rule_helper("ma", ma_mean, ma_std)
    # ma_rules_2_presentation = presentation_rule_helper("ma", ma_mean-2*ma_std, ma_std)
    # ma_rules_3_presentation = presentation_rule_helper("ma", ma_mean+2*ma_std, ma_std)

    # secz_rules_1_presentation = presentation_rule_helper("secz", secz_mean, secz_std)
    # secz_rules_2_presentation = presentation_rule_helper("secz", secz_mean-2*secz_std, secz_std)
    # secz_rules_3_presentation = presentation_rule_helper("secz", secz_mean+2*secz_std, secz_std)

    # bn_rules_1_presentation = presentation_rule_helper("bn", bn_mean, bn_std)
    # bn_rules_2_presentation = presentation_rule_helper("bn", bn_mean-2*bn_std, bn_std)
    # bn_rules_3_presentation = presentation_rule_helper("bn", bn_mean+2*bn_std, bn_std)

    # bc_rules_1_presentation = presentation_rule_helper("bc", bc_mean, bc_std)
    # bc_rules_2_presentation = presentation_rule_helper("bc", bc_mean-2*bc_std, bc_std)
    # bc_rules_3_presentation = presentation_rule_helper("bc", bc_mean+2*bc_std, bc_std)

    # nn_rules_1_presentation = presentation_rule_helper("nn", nn_mean, nn_std)
    # nn_rules_2_presentation = presentation_rule_helper("nn", nn_mean-2*nn_std, nn_std)
    # nn_rules_3_presentation = presentation_rule_helper("nn", nn_mean+2*nn_std, nn_std)

    # m_rules_1_presentation = presentation_rule_helper("m", m_mean, m_std)
    # m_rules_2_presentation = presentation_rule_helper("m", m_mean-2*m_std, m_std)
    # m_rules_3_presentation = presentation_rule_helper("m", m_mean+2*m_std, m_std)


    # rule_presentation  = ct_1_rules_presentation + ct_2_rules_presentation + ct_3_rules_presentation
    # rule_presentation += ucsize_1_rules_presentation + ucsize_2_rules_presentation + ucsize_3_rules_presentation
    # rule_presentation += ucshape_1_rules_presentation + ucshape_2_rules_presentation + ucshape_3_rules_presentation
    # rule_presentation += ma_rules_1_presentation + ma_rules_2_presentation + ma_rules_3_presentation
    # rule_presentation += secz_rules_1_presentation + secz_rules_2_presentation + secz_rules_3_presentation
    # rule_presentation += bn_rules_1_presentation + bn_rules_2_presentation + bn_rules_3_presentation
    # rule_presentation += bc_rules_1_presentation + bc_rules_2_presentation + bc_rules_3_presentation
    # rule_presentation += nn_rules_1_presentation + nn_rules_2_presentation + nn_rules_3_presentation
    # rule_presentation += m_rules_1_presentation + m_rules_2_presentation + m_rules_3_presentation

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

def read_rules_BC(rule_set):
    for i in range(len(rule_set)):
        dict_m  = rule_set[i][0]
        b   = dict_m[frozenset({'B'})].item()
        m   = dict_m[frozenset({'M'})].item()
        b_m = dict_m[frozenset({'B','M'})].item()

        print(BC_RULE_PRESENT.format(i+1,b,m, b_m))
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

def read_rules_iris(rule_set):
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        s     = dict_m[frozenset({'S'})].item()
        c     = dict_m[frozenset({'C'})].item()
        v     = dict_m[frozenset({'V'})].item()
        s_c_v = dict_m[frozenset({'S', 'C', 'V'})].item()
        print(IRIS_RULE_PRESENT.format(i+1,s,c,v,s_c_v))

# ------------- Heart Disease ----------------

def preprocess_dataset_heart_disease(dataset_filepath, processed_dataset_filepath):
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", 
        "oldpeak", "slope", "ca", "thal", "y"]

    df = pd.read_csv(dataset_filepath, usecols=columns, na_values='?')
    for column in columns:
        df[column] = df[column].fillna(value=df[column].mean())

    # Change classes to 0 (Absence) and 1 (Present)
    # From heart-disease.names, of dataset:
    # Experiments with the Cleveland database have concentrated on simply
    #  attempting to distinguish presence (values 1,2,3,4) from absence (value
    #  0). 
    df.loc[df.y > 1, 'y'] = 1
    df.to_csv(processed_dataset_filepath, index=False)

def read_dataset_heart_disease(dataset_filepath):

    with open(dataset_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        X = []
        Y = []

        next(csv_reader) # to skip the header file

        # -- 1. #3  (age)       
        # -- 2. #4  (sex)       
        # -- 3. #9  (cp)        
        # -- 4. #10 (trestbps)  
        # -- 5. #12 (chol)      
        # -- 6. #16 (fbs)       
        # -- 7. #19 (restecg)   
        # -- 8. #32 (thalach)   
        # -- 9. #38 (exang)     
        # -- 10. #40 (oldpeak)   
        # -- 11. #41 (slope)     
        # -- 12. #44 (ca)        
        # -- 13. #51 (thal) 

        for age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,y in csv_reader:
            X.append([float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak),
                float(slope), float(ca), float(thal)])
            Y.append(int(y))

    X = np.asarray(X).astype(float)
    Y = np.asarray(Y).astype(float)
    return X, Y

def generate_rules_dataset_heart_disease(X_train, dataset_name):

    [age_mean, sex_mean, cp_mean, trestbps_mean, chol_mean,
    fbs_mean, restecg_mean, thalach_mean, exang_mean, oldpeak_mean,
    slope_mean, ca_mean, thal_mean] = np.mean(X_train, axis=0) # mean along columns

    [age_std, sex_std, cp_std, trestbps_std, chol_std,
    fbs_std, restecg_std, thalach_std, exang_std, oldpeak_std,
    slope_std, ca_std, thal_std] = np.std(X_train, axis=0, dtype=np.float64) # std along columns

    # Create rules
    rules_age      = generate_rule(0, age_mean, age_std)
    rules_sex      = generate_rule(1, sex_mean, sex_std)
    rules_cp       = generate_rule(2, cp_mean, cp_std)
    rules_trestbps = generate_rule(3, trestbps_mean, trestbps_std)
    rules_chol     = generate_rule(4, chol_mean, chol_std)
    rules_fbs      = generate_rule(5, fbs_mean, fbs_std)
    rules_restecg  = generate_rule(6, restecg_mean, restecg_std)
    rules_thalach  = generate_rule(7, thalach_mean, thalach_std)
    rules_exang    = generate_rule(8, exang_mean, exang_std)
    rules_oldpeak  = generate_rule(9, oldpeak_mean, oldpeak_std)
    rules_slope    = generate_rule(10, slope_mean, slope_std)
    rules_ca       = generate_rule(11, ca_mean, ca_std)
    rules_thal     = generate_rule(12, thal_mean, thal_std)

    s_list  = rules_age + rules_sex + rules_cp + rules_trestbps + rules_chol
    s_list += rules_fbs + rules_restecg + rules_thalach + rules_exang
    s_list += rules_oldpeak + rules_slope + rules_ca + rules_thal

    rule_set = start_weights(s_list, dataset_name)

    # Aid in result presentation
    age_rules_presentation      = presentation_rule_helper("age", age_mean, age_std)
    sex_rules_presentation      = presentation_rule_helper("sex", sex_mean, sex_std)
    cp_rules_presentation       = presentation_rule_helper("cp", cp_mean, cp_std)
    trestbps_rules_presentation = presentation_rule_helper("trestbps", trestbps_mean, trestbps_std)
    chol_rules_presentation     = presentation_rule_helper("chol", chol_mean, chol_std)
    fbs_rules_presentation      = presentation_rule_helper("fbs", fbs_mean, fbs_std)
    restecg_rules_presentation  = presentation_rule_helper("restecg", restecg_mean, restecg_std)
    thalach_rules_presentation  = presentation_rule_helper("thalach", thalach_mean, thalach_std)
    exang_rules_presentation    = presentation_rule_helper("exang", exang_mean, exang_std)
    oldpeak_rules_presentation  = presentation_rule_helper("oldpeak", oldpeak_mean, oldpeak_std)
    slope_rules_presentation    = presentation_rule_helper("slope", slope_mean, slope_std)
    ca_rules_presentation       = presentation_rule_helper("ca", ca_mean, ca_std)
    thal_rules_presentation     = presentation_rule_helper("thal", thal_mean, thal_std)

    rule_presentation  = age_rules_presentation + sex_rules_presentation + cp_rules_presentation
    rule_presentation += trestbps_rules_presentation + chol_rules_presentation
    rule_presentation += fbs_rules_presentation + restecg_rules_presentation + thalach_rules_presentation
    rule_presentation += exang_rules_presentation + oldpeak_rules_presentation + slope_rules_presentation
    rule_presentation += ca_rules_presentation + thal_rules_presentation

    return rule_set, rule_presentation

def dataset_heart_disease():
    dataset_filepath           = os.path.join(DATASET_FOLDER, HD_DATASET_FILE)
    processed_dataset_filepath = os.path.join(DATASET_FOLDER, HD_PROCESSED_DATASET_FILE)

    preprocess_dataset_heart_disease(dataset_filepath, processed_dataset_filepath)

    X, Y = read_dataset_heart_disease(processed_dataset_filepath)
    X_train, Y_train, X_test, Y_test = split_test_train(X,Y)

    # Pre process
    Y_train = tensor(Y_train).to(torch.int64)
    Y_train = one_hot(Y_train, num_classes=HD_NUM_CLASSES).float()

    return X_train, Y_train, X_test, Y_test

def read_rules_heart_disease(rule_set):
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        a   = dict_m[frozenset({'A'})].item()
        p   = dict_m[frozenset({'P'})].item()
        a_p = dict_m[frozenset({'A', 'P'})].item()
        print(HD_RULE_PRESENT.format(i+1,a,p,a_p))

# ------------- Wine ----------------

def preprocess_dataset_wine(dataset_filepath, processed_dataset_filepath):
    columns = ["y","ma","ash","al","mg","ph","fl","nph","pr","cl","hue","od","prol"]

    df = pd.read_csv(dataset_filepath, usecols=columns)
    # Change classes to 0,1,2 from 1,2,3
    df.loc[df.y == 1, 'y'] = 0
    df.loc[df.y == 2, 'y'] = 1
    df.loc[df.y == 3, 'y'] = 2
    df.to_csv(processed_dataset_filepath, index=False)

def read_dataset_wine(dataset_filepath):

    with open(dataset_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        X = []
        Y = []

        next(csv_reader) # to skip the header file

        # 1) Alcohol (y)
        # 2) Malic acid (ma)
        # 3) Ash (ash)
        # 4) Alcalinity of ash (al) 
        # 5) Magnesium (mg)
        # 6) Total phenols (ph)
        # 7) Flavanoids (fl)
        # 8) Nonflavanoid phenols (nph)
        # 9) Proanthocyanins (pr)
        # 10)Color intensity (cl)
        # 11)Hue (hue)
        # 12)OD280/OD315 of diluted wines (od)
        # 13)Proline (prol)           

        for y,ma,ash,al,mg,ph,fl,nph,pr,cl,hue,od,prol in csv_reader:
            X.append([float(ma), float(ash), float(al), float(mg), float(ph), float(fl), 
                float(nph), float(pr), float(cl), float(hue), float(od), float(prol)])
            Y.append(int(y))

    X = np.asarray(X).astype(float)
    Y = np.asarray(Y).astype(float)
    return X, Y

def generate_rules_dataset_wine(X_train, dataset_name):

    [ma_mean, ash_mean, al_mean, mg_mean, ph_mean, fl_mean, nph_mean,
    pr_mean, cl_mean, hue_mean, od_mean, prol_mean] = np.mean(X_train, axis=0) # mean along columns

    [ma_std, ash_std, al_std, mg_std, ph_std, fl_std, nph_std,
    pr_std, cl_std, hue_std, od_std, prol_std] = np.std(X_train, axis=0, dtype=np.float64) # std along columns

    # Create rules
    rules_ma   = generate_rule(0, ma_mean, ma_std)
    rules_ash  = generate_rule(1, ash_mean, ash_std)
    rules_al   = generate_rule(2, al_mean, al_std)
    rules_mg   = generate_rule(3, mg_mean, mg_std)
    rules_ph   = generate_rule(4, ph_mean, ph_std)
    rules_fl   = generate_rule(5, fl_mean, fl_std)
    rules_nph  = generate_rule(6, nph_mean, nph_std)
    rules_pr   = generate_rule(7, pr_mean, pr_std)
    rules_cl   = generate_rule(8, cl_mean, cl_std)
    rules_hue  = generate_rule(9, hue_mean, hue_std)
    rules_od   = generate_rule(10, od_mean, od_std)
    rules_prol = generate_rule(11, prol_mean, prol_std)


    s_list  = rules_ma + rules_ash + rules_al + rules_mg + rules_ph
    s_list += rules_fl + rules_nph + rules_pr + rules_cl
    s_list += rules_hue + rules_od + rules_prol

    rule_set = start_weights(s_list, dataset_name)

    # Aid in result presentation
    ma_rules_presentation   = presentation_rule_helper("ma", ma_mean, ma_std)
    ash_rules_presentation  = presentation_rule_helper("ash", ash_mean, ash_std)
    al_rules_presentation   = presentation_rule_helper("al", al_mean, al_std)
    mg_rules_presentation   = presentation_rule_helper("mg", mg_mean, mg_std)
    ph_rules_presentation   = presentation_rule_helper("ph", ph_mean, ph_std)
    fl_rules_presentation   = presentation_rule_helper("fl", fl_mean, fl_std)
    nph_rules_presentation  = presentation_rule_helper("nph", nph_mean, nph_std)
    pr_rules_presentation   = presentation_rule_helper("pr", pr_mean, pr_std)
    cl_rules_presentation   = presentation_rule_helper("cl", cl_mean, cl_std)
    hue_rules_presentation  = presentation_rule_helper("hue", hue_mean, hue_std)
    od_rules_presentation   = presentation_rule_helper("od", od_mean, od_std)
    prol_rules_presentation = presentation_rule_helper("prol", prol_mean, prol_std)

    rule_presentation  = ma_rules_presentation + ash_rules_presentation + al_rules_presentation
    rule_presentation += mg_rules_presentation + ph_rules_presentation + fl_rules_presentation
    rule_presentation += nph_rules_presentation + pr_rules_presentation + cl_rules_presentation
    rule_presentation += hue_rules_presentation + od_rules_presentation + prol_rules_presentation

    return rule_set, rule_presentation

def dataset_wine():
    dataset_filepath           = os.path.join(DATASET_FOLDER, WINE_DATASET_FILE)
    processed_dataset_filepath = os.path.join(DATASET_FOLDER, WINE_PROCESSED_DATASET_FILE)

    preprocess_dataset_wine(dataset_filepath, processed_dataset_filepath)

    X, Y = read_dataset_wine(processed_dataset_filepath)
    X_train, Y_train, X_test, Y_test = split_test_train(X,Y)

    # Pre process
    Y_train = tensor(Y_train).to(torch.int64)
    Y_train = one_hot(Y_train, num_classes=WINE_NUM_CLASSES).float()

    return X_train, Y_train, X_test, Y_test

def read_rules_wine(rule_set):
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        a     = dict_m[frozenset({'A'})].item()
        b     = dict_m[frozenset({'B'})].item()
        c     = dict_m[frozenset({'C'})].item()
        a_b_c = dict_m[frozenset({'A', 'B', 'C'})].item()
        print(WINE_RULE_PRESENT.format(i+1,a,b,c,a_b_c))

# ------------- Digits ----------------

def preprocess_dataset_digits(dataset_filepath, processed_dataset_filepath):

    # Write csv with only class 0 and 1
    df = pd.read_csv(dataset_filepath)

    # Consider class 0 and 1 only
    list_0_1 = []
    for line in df.to_numpy():
        if line[64] < 2: # Class
            list_0_1.append(line)

    df_0_1 = pd.DataFrame(list_0_1)
    df_0_1.to_csv(processed_dataset_filepath, index=False)

def read_dataset_digits(dataset_filepath):

    with open(dataset_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        X = []
        Y = []

        next(csv_reader) # to skip the header file
        
        # 64 px are the attributes 
        for line in csv_reader:
            att = line[:-1]
            att_float = [float(e) for e in att]
            X.append(att_float)
            Y.append(int(line[-1]))

    X = np.asarray(X).astype(float)
    Y = np.asarray(Y).astype(float)
    return X, Y

def generate_rules_dataset_digits(X_train, dataset_name):

    # Attributes are name after pixel: 0, 1, ..., 64
    att_mean = np.mean(X_train, axis=0) # mean along columns
    att_std  = np.std(X_train, axis=0, dtype=np.float64)

    s_list = []
    # Create rules
    for i in range(len(att_mean)):
        s_list += generate_rule(i, att_mean[i], att_std[i])

    rule_set = start_weights(s_list, dataset_name)

    # Aid in result presentation
    rule_presentation = []
    for i in range(len(att_mean)):
        rule_presentation += presentation_rule_helper(str(i), att_mean[i], att_std[i])

    return rule_set, rule_presentation

def dataset_digits():
    dataset_filepath           = os.path.join(DATASET_FOLDER, DIG_DATASET_FILE)
    processed_dataset_filepath = os.path.join(DATASET_FOLDER, DIG_PROCESSED_DATASET_FILE)

    preprocess_dataset_digits(dataset_filepath, processed_dataset_filepath)

    X, Y = read_dataset_digits(processed_dataset_filepath)
    X_train, Y_train, X_test, Y_test = split_test_train(X,Y)

    # Pre process
    Y_train = tensor(Y_train).to(torch.int64)
    Y_train = one_hot(Y_train, num_classes=DIG_NUM_CLASSES).float()

    return X_train, Y_train, X_test, Y_test

def read_rules_digits(rule_set):
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        _0     = dict_m[frozenset({'0'})].item()
        _1     = dict_m[frozenset({'1'})].item()
        # _2     = dict_m[frozenset({'2'})].item()
        # _3     = dict_m[frozenset({'3'})].item()
        # _4     = dict_m[frozenset({'4'})].item()
        # _5     = dict_m[frozenset({'5'})].item()
        # _6     = dict_m[frozenset({'6'})].item()
        # _7     = dict_m[frozenset({'7'})].item()
        # _8     = dict_m[frozenset({'8'})].item()
        # _9     = dict_m[frozenset({'9'})].item()
        # unc    = dict_m[frozenset({'0','1','2','3','4','5','6','7','8','9'})].item()
        # print(DIG_RULE_PRESENT.format(i+1,_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,unc))
        _0_1    = dict_m[frozenset({'0','1'})].item()
        print(DIG_RULE_PRESENT.format(i+1,_0,_1,_0_1))

# ------------ Common ------------------------

def split_test_train(X,Y):
    # Split between train and test (70%/30%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_PERCENTAGE) 
    return X_train, Y_train, X_test, Y_test
