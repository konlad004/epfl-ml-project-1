import numpy as np


def least_squares(y : np.ndarray, tx : np.ndarray) -> tuple[np.ndarray, float]:     

    '''
    Computes the Least Squares solution.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (N,)
    tx : np.ndarray
        Features array of shape (N, D)

    Returns
    ---------
    w : np.ndarray
        Optimal weights for the least squares algorithm with shape (D,)
    loss : float
        MSE loss for the input data with the optimal weights.
    
    '''
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    txw = tx @ w
    loss = 1/(2*len(y))*(y- txw).T @ (y - txw)
    
    return w, loss




def ridge_regression(y : np.ndarray, tx : np.ndarray, lambda_ : float) -> tuple[np.ndarray, float]:

    '''
    Computes the Ridge regression solution.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (N,)
    tx : np.ndarray
        Features array of shape (N, D)
    lambda_ : float
        Regularisation parameter.


    Returns
    ---------
    w : np.ndarray
        Optimal weights for the ridge regression algorithm with shape (D,)
    loss : float
        MSE loss for the input data with the optimal weights, not including the regularisation loss.

    '''

    N=len(y)
    w = np.linalg.solve(tx.T @ tx + (lambda_ * (2*N)) * np.identity(tx.shape[1]), tx.T @ y)
    txw = tx @ w
    loss = 1/(2*N)*(y- txw).T @ (y- txw)

    return w, loss