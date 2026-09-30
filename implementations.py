import numpy as np


def least_squares(y, tx):

    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    txw = tx @ w
    return w, 1/(2*len(y))*(y- txw).T @ (y- txw)




def ridge_regression(y, tx, lambda_):

    N=len(y)
    w = np.linalg.solve(tx.T @ tx + (lambda_ * (2*N)) * np.identity(tx.shape[1]), tx.T @ y)
    txw = tx @ w
    return w, 1/(2*N)*(y- txw).T @ (y- txw)