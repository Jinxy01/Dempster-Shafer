import torch
from dempster_shaffer import get_powerset

# Dataset generic
DATASET_FOLDER  = "dataset"
X_              = "x"
Y_              = "y"
CLASS_          = "class"

#--- A1 Dataset ---
NUM_ELEMENTS     = 500
A1_DATASET_FILE  = "A1.csv" 
# TRAIN_PERCENTAGE = 0.7
TEST_PERCENTAGE  = 0.3
NUM_CLASSES      = 2
EPSILON          = 0.0001

# Variables
DTYPE = torch.float
DEVICE = torch.device("cpu")
EMPTY_SET    = set()
COMPLETE_SET = frozenset({'R','B'})
POWERSET = get_powerset({'R'}.union({'B'}))