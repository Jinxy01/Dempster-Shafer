import re
from itertools import islice

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph import *
from utils.config import *

# ------------------------
def digits(fileprefix):
    dict_rule = {i : 0 for i in range(1,257)}
    file_format = "{}_{}.txt"
    num_files = 3

    # 3 files. Get the mean of rules
    for i in range(num_files):
        
        filepath = file_format.format(fileprefix,i)
        filepath = os.path.join(DATA_FOLDER, filepath)
        f = open(filepath, "r")
        for line in f:
            rule = re.search('Rule (.+?):', line).group(1)
            value = re.findall(r'\d+\.\d+', line)[0]
            dict_rule[int(rule)] += float(value)

    dict_rule = dict(sorted(dict_rule.items()))
    # Mean
    dict_rule = {k: v /num_files for k, v in dict_rule.items()}

    factor      = 4 # 4 rules per pixel
    num_pix     = 64
    matrix_side = 8
    f.close()

    dict_px = {x : 0 for x in range(1,num_pix+1)}
    for i in range(num_pix):
        max_ = -1
        for j in range(1,factor+1):
            max_ = max(max_, dict_rule[i*factor+j])
            #dict_px[i+1] += dict_rule[i*factor+j] 
        dict_px[i+1] = max_
        #dict_px[i+1] = dict_px[i+1]/factor # mean

    # Group into matrix 8x8
    matrix = []
    for i in range(matrix_side):
        line = []
        for j in range(1,matrix_side+1):
            line.append(dict_px[i*matrix_side+j])
        matrix.append(line)

    return matrix

def process_digits():
    list_files = ["class0", "class1"]
    #list_files = ["class0_300", "class1_300"]
    #list_files = ["class0_20", "class1_20"]
    #list_files = ["class0_100", "class1_100"]
    #list_files = ["class0_50", "class1_50"]
    #list_files = ["class0_np", "class1_np"]
    for fileprefix in list_files:
        matrix = digits(fileprefix)
        print(matrix)
        draw_digits(matrix)

# ------------------------
def heart_disease_presence(fileprefix):
    dict_rule = {i : 0 for i in range(1,53)}
    file_format = "{}_{}.txt"
    num_files = 3

    # 3 files. Get the mean of rules
    for i in range(num_files):
        
        filepath = file_format.format(fileprefix,i)
        filepath = os.path.join(DATA_FOLDER, filepath)
        f = open(filepath, "r")
        for line in f:
            rule = re.search('Rule (.+?):', line).group(1)
            value = re.findall(r'\d+\.\d+', line)[0]
            dict_rule[int(rule)] += float(value)

    # Mean
    dict_rule = {k: v /num_files for k, v in dict_rule.items()}
    dict_rule = dict(sorted(dict_rule.items(), key=lambda item: item[1], reverse=True))
    print(dict_rule)
    return dict_rule

def process_heart_disease():
    malign_file = "hd_p"
    dict_rule = heart_disease_presence(malign_file)
    return

# ------------------------
def breast_cancer(fileprefix):
    dict_rule = {i : 0 for i in range(1,53)}
    file_format = "{}_{}.txt"
    num_files = 3
    top_n = 18 

    # 3 files. Get the mean of rules
    for i in range(num_files):
        
        filepath = file_format.format(fileprefix,i)
        filepath = os.path.join(DATA_FOLDER, filepath)
        f = open(filepath, "r")
        for line in f:
            rule = re.search('Rule (.+?):', line).group(1)
            value = re.findall(r'\d+\.\d+', line)[0]
            dict_rule[int(rule)] += float(value)

    # Mean
    dict_rule = {k: v /num_files for k, v in dict_rule.items()}
    dict_rule = dict(sorted(dict_rule.items(), key=lambda item: item[1], reverse=True))
    #dict_top_n_rules = dict(islice(dict_rule.items(), top_n)) 
    print("\nFile prefix:", fileprefix)
    for k, v in dict_rule.items():
        print(RULE_PRESENTATION_DISPLAY.format(k, v))

def process_breast_cancer():
    malign_file = "bc_m"
    bening_file = "bc_b"
    breast_cancer(malign_file)
    breast_cancer(bening_file)


# ------------------------

if __name__ == "__main__":
    process_digits()
    #process_heart_disease()
    #process_breast_cancer()
    