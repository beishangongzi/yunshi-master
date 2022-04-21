import os

import numpy as np


def is_val(idx):
    return idx % 8 == 7


def save(rows, file='file.txt'):
    file = open(file, 'w');
    for r in rows:
        file.write(r)
        file.write('\n')


def depart(dir='/marine-farm-seg2/gt'):
    train, val = [], []
    files = os.listdir(dir)
    for i, file in enumerate(files):
        if not is_val(i):
            train.append(file)
        else:
            val.append(file)

    print(len(train))
    print(len(val))
    save(train, 'train.txt')
    save(val, 'val.txt')


depart()
