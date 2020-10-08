# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    x = np.random.permutation(x)
    nb = int(x.shape[0] * ratio)
    train_x = x[:nb]
    train_y = y[:nb]
    test_x = x[nb:]
    test_y = y[nb:]
    return train_x, train_y, test_x, test_y
