import torch
from itertools import chain, combinations
from torch.nn.functional import one_hot

def get_powerset(set_elements):
    # Powerset: set + empty set + subsets of given set
    list_elements = list(set_elements)
    list_powerset = list(chain.from_iterable(combinations(list_elements, e) 
        for e in range(1, len(list_elements)+1))) # start at 1 to ignore empty set
    # Transform into a list of sets. 
    # We can use set() but then we will get "TypeError: unhashable type: 'set'" when adding as key to dictionary
    # So we use frozenset()
    list_sets_powerset = [frozenset(e) for e in list_powerset] # allow to be added to dictionary
    return list_sets_powerset

#------------------ Dataset Generic -----------------
DATASET_FOLDER  = "dataset"
BREAK_IT        = "\nBreaking at {} iteration\n"
# TRAIN_PERCENTAGE = 0.7
TEST_PERCENTAGE   = 0.3
EPSILON         = 0.0001
NUM_CLASSES     = 2
NUM_EPOCHS      = 750
BATCH_SIZE      = 16

RULE_LTE     = "{} <= {:.3f}"
RULE_BETWEEN = "{:.3f} < {} <= {:.3f}"
RULE_GT      = "{} > {:.3f}"

RULE_PRESENTATION_DISPLAY = "Rule {}: {}"

CLASS_0_ONE_HOT = one_hot(torch.tensor(0), num_classes=NUM_CLASSES).float()
CLASS_1_ONE_HOT = one_hot(torch.tensor(1), num_classes=NUM_CLASSES).float()
#CLASS_2_ONE_HOT = one_hot(torch.tensor(2), num_classes=NUM_CLASSES).float()

#------------------ A1 Dataset ----------------- 
NUM_ELEMENTS      = 500
X_                = "x"
Y_                = "y"
CLASS_            = "class"
A1_DATASET_FILE   = "A1.csv" 
A1_NUM_CLASSES    = 2
A1_LOSS_IMG       = "A1_Loss"
A1_RULE_TABLE     = "A1_Rule_Table"
RULE_TABLE_TITLE  = "\nAccuracy = {:.2f}% ({}/{})\n"
A1_TABLE_HEADER   = ["Rule", "Blue", "Red", "Uncertainty"]
A1_RULE_PRESENT   = "Rule {}: B = {}, R = {}, Uncertainty = {}"

A1_COMPLETE_SET = frozenset({'R','B'})
A1_POWERSET     = get_powerset({'R'}.union({'B'}))

#------------------ Breast Cancer Dataset ------------------
BC_NUM_CLASSES            = 2
BC_DATASET_FILE           = "breast_cancer_uci.csv" 
BC_PROCESSED_DATASET_FILE = "breast_cancer_uci_processed.csv" 
BC_LOSS_IMG   = "BC_Loss"
BC_RULE_TABLE = "BC_Rule_Table"

BC_COMPLETE_SET = frozenset({'B','M'})
BC_POWERSET     = get_powerset({'B'}.union({'M'}))

BC_RULE_PRESENT              = "Rule {}: B = {}, M = {}, Uncertainty = {}"
BC_RULE_PRESENTATION_TITLE   = "\nRules ordered by malignacy:"

#---------------------- -Iris Dataset ----------------------
IRIS_NUM_CLASSES            = 3
IRIS_DATASET_FILE           = "iris.csv" 
IRIS_PROCESSED_DATASET_FILE = "iris_processed.csv" 
IRIS_LOSS_IMG               = "IRIS_Loss"

IRIS_COMPLETE_SET = frozenset({'S','C','V'})
IRIS_POWERSET     = get_powerset(({'S'}.union({'C'})).union({'V'}))

IRIS_RULE_PRESENT = "Rule {}: S = {}, C = {}, V = {}, Uncertainty = {}"

#---------------------- Heart Disease Dataset ----------------------
HD_NUM_CLASSES            = 2
HD_DATASET_FILE           = "heart_disease_cleveland.csv" 
HD_PROCESSED_DATASET_FILE = "heart_disease_processed.csv" 
HD_LOSS_IMG               = "HD_Loss"

HD_COMPLETE_SET = frozenset({'A','P'})
HD_POWERSET     = get_powerset({'A'}.union({'P'}))

HD_RULE_PRESENT = "Rule {}: A = {}, P = {}, Uncertainty = {}"

#---------------------- Wine Dataset ----------------------
WINE_NUM_CLASSES            = 3
WINE_DATASET_FILE           = "wine.csv" 
WINE_PROCESSED_DATASET_FILE = "wine_processed.csv" 
WINE_LOSS_IMG               = "WINE_Loss"

WINE_COMPLETE_SET = frozenset({'A','B', 'C'})
WINE_POWERSET     = get_powerset(({'A'}.union({'B'})).union({'C'}))

WINE_RULE_PRESENT = "Rule {}: A = {}, B = {}, C = {}, Uncertainty = {}"

#-----------------------------------------------------------

# Image
IMAGE_FOLDER = "imgs"
TITLE_LOSS   = "Training Loss"
Y_AXIS       = "Mean Square Error (MSE)"
X_AXIS       = "Epochs"

# Variables
DTYPE = torch.float
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
EMPTY_SET    = set()

# Table
NUM_FORMAT = '{:.3f}'

