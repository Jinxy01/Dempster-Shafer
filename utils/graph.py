import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import *

def draw_loss(it_loss, filepath):

    for i in range(len(it_loss)):
        #plt.plot(i, it_loss[i].detach().numpy(), marker='o', color='red', linestyle='-', linewidth=2, markersize=1)
        plt.plot(i, it_loss[i], marker='o', color='red', linestyle='-', linewidth=2, markersize=1)     

    plt.grid(b=True) # Add grid
    plt.title(TITLE_LOSS)
    plt.xlabel(X_AXIS)
    plt.ylabel(Y_AXIS)

    plt.savefig(filepath)
    plt.show()


def draw_digits(matrix):
    # Make a 8x8 grid
    nrows, ncols = 8,8
    image = np.zeros(nrows*ncols)

    # Set every other cell to a random number (this would be your data)
    image[::2] = np.random.random(nrows*ncols //2)

    # Reshape things into a 9x9 grid.
    image = image.reshape((nrows, ncols))
    image = matrix

    row_labels = range(nrows)
    col_labels = range(ncols)
    plt.matshow(image, cmap='viridis_r')
    plt.tick_params('x', bottom=True, top=False, labelbottom=True, labeltop=False)
    plt.xticks(range(ncols), col_labels)
    plt.colorbar()
    #plt.clim(-0.2, 0.2)
    plt.title("Pixel contribution for class X")
    plt.yticks(range(nrows), row_labels)

    plt.savefig("digit")
    plt.show()

def draw_rule_table(rule_set, filepath, accuracy, tot_correct_predicts, tot_predict, rule_presentation):

    table_data=[A1_TABLE_HEADER]

    fig = plt.figure(dpi=120)
    fig.suptitle(RULE_TABLE_TITLE.format(accuracy, tot_correct_predicts, tot_predict), y=0.9)
    ax = fig.add_subplot(1,1,1)
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        r, b, u = dict_m[frozenset({'R'})].item(), dict_m[frozenset({'B'})].item(), dict_m[frozenset({'R', 'B'})].item() 
        table_data.append([rule_presentation[i], NUM_FORMAT.format(b), NUM_FORMAT.format(r), NUM_FORMAT.format(u)])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.set_fontsize(20)
    # table.scale(1,4)
    table.scale(1,2)
    ax.axis('off')
    plt.savefig(filepath)
    plt.show()
