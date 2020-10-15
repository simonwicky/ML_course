# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    gram = np.linalg.inv(tx.transpose() @ tx)
    weight = gram @ tx.transpose() @ y
    return weight
