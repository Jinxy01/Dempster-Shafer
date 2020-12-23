"""
@author: Tiago Roxo, UBI
@date: 2020
"""
import torch


if __name__ == "__main__":
    s = torch.nn.functional.one_hot(tensor, num_classes=2)