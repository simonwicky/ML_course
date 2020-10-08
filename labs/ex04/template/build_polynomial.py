# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    return (x[:,np.newaxis] @ np.ones([1,degree+1])) ** np.array(range(degree + 1))


