"""
@author: Tiago Roxo, UBI
@date: 2020
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split

def read_dataset(dataset_filepath):
    with open(dataset_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        X = []
        Y = []
        next(csv_reader) # to skip the header file
        # crop_land | grazing_land | forest_land | fishing_ground | built_up_land | carbon(Y) | total
        for row in csv_reader:
            # X.append([row[0], row[1], row[2], row[3], row[4], row[6], 1.0])
            X.append([row[0], row[1], row[2], row[3], row[4], 1.0])
            Y.append(float("{0:.3f}".format(float(row[5])))) # Carbon

    X = np.asarray(X).astype(float)
    Y = np.asarray(Y).astype(float)

if __name__ == "__main__":
    dataset_filepath = os.path.join(DATASET_FOLDER, A1_DATASET_FILE)
    X, Y = read_dataset(dataset_filepath)