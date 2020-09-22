# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y[:,np.newaxis] - tx @ w[:,np.newaxis]
    loss = (e.T @ e) / tx.shape[0]
    return loss.item()