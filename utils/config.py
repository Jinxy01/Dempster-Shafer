import torch
from dempster_shaffer import get_powerset

# Dataset generic
DATASET_FOLDER  = "dataset"
X_              = "x"
Y_              = "y"
CLASS_          = "class"
BREAK_IT        = "Breaking at {} iteration"
EPSILON         = 0.0001
NUM_CLASSES     = 2 

#--- A1 Dataset ---
NUM_ELEMENTS      = 500
A1_DATASET_FILE   = "A1.csv" 
# TRAIN_PERCENTAGE = 0.7
TEST_PERCENTAGE   = 0.3
A1_NUM_CLASSES    = 2
A1_LOSS_IMG       = "A1_Loss"
A1_RULE_TABLE     = "A1_Rule_Table"
RULE_TABLE_TITLE  = "Accuracy = {:.2f}% ({}/{})"
A1_TABLE_HEADER   = ["Rule", "Blue", "Red", "Uncertainty"]

RULE_LTE     = "{} <= {:.3f}"
RULE_BETWEEN = "{:.3f} < {} <= {:.3f}"
RULE_GT      = "{} > {:.3f}"

#--- Breast Cancer Dataset ---
BC_NUM_CLASSES            = 2
BC_DATASET_FILE           = "breast_cancer_uci.csv" 
BC_PROCESSED_DATASET_FILE = "breast_cancer_uci_processed.csv" 
# BC_READ_RULES = "Rule {:>3}: ct = {:.3f}, ucsize = {:.3f}, ucshape = {:.3f}, ma = {:.3f}, secz = {:.3f}, bn = {:.3f}, bc = {:.3f}, nn = {:.3f}, m = {:.3f}, uncertainty = {:.3f}"
BC_LOSS_IMG   = "BC_Loss"
BC_RULE_TABLE = "BC_Rule_Table"

_att_fronzenset = {
    "A" : "ct",
    "B" : "ucsize",
    "C" : "ucshape",
    "D" : "ma",
    "E" : "secz",
    "F" : "bn",
    "G" : "bc",
    "H" : "nn",
    "I" : "m"
}


# Image
IMAGE_FOLDER = "imgs"
TITLE_LOSS   = "Training Loss"
Y_AXIS       = "Mean Square Error (MSE)"
X_AXIS       = "Epochs"

# Variables
DTYPE = torch.float
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMPTY_SET    = set()
#COMPLETE_SET = frozenset({'R','B'})
#POWERSET = get_powerset({'R'}.union({'B'}))

COMPLETE_SET = frozenset({'B','M'})
POWERSET = get_powerset({'B'}.union({'M'}))

# Table
NUM_FORMAT = '{:.3f}'