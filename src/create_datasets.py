from random import uniform
from sklearn.model_selection import train_test_split
import csv
import os

def create_dataset_A1():
    num_element = 500
    list_elements = [(uniform(-1,1),uniform(-1,1)) for _ in range(num_element)]

    # Classes: 0 = blue, 1 = red
    # Criteria: if y < 0 then blue else red
    list_elements_with_class = []
    for x,y in list_elements:
        list_elements_with_class.append((x,y, 0 if y < 0 else 1))
    return list_elements_with_class

    return list_elements_with_class

def save_dataset_file(list_elements, dataset_filename):
    dataset_folder   = "dataset"

    dataset_filepath = os.path.join(dataset_folder, dataset_filename)

    with open(dataset_filepath, mode='w') as dataset_file:
        dataset_writer = csv.writer(dataset_file, delimiter=',')
        for x,y,c in list_elements:
            dataset_writer.writerow([x,y,c])
    
    dataset_file.close()

if __name__ == "__main__":
    dataset_filename = "A1.csv"

    list_elements = create_dataset_A1()
    save_dataset_file(list_elements, dataset_filename)
    
    