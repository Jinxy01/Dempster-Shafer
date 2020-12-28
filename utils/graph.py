import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import *

def draw_loss(it_loss, filepath):

    for i in range(len(it_loss)):
        plt.plot(i, it_loss[i].detach().numpy(), marker='o', color='red', linestyle='-', linewidth=2, markersize=1)     

    plt.grid(b=True) # Add grid
    plt.title(TITLE_LOSS)
    plt.xlabel(X_AXIS)
    plt.ylabel(Y_AXIS)

    plt.savefig(filepath)
    plt.show()