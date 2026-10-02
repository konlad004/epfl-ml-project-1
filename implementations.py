import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.

    Parameters
    ----------
    y: np.ndarray
        Target values of shape (N,).
    tx: np.ndarray
        Feature matrix of shape (N, D).
    initial_w: np.ndarray
        Initial weight vector of shape (D,).
    max_iters: int
        Number of gradient descent iterations.
    gamma: float
        Learning rate.

    Returns
    ----------
    w: np.ndarray
        Final weight vector of shape (D,).
    loss: float
        Mean squared error loss.
    """
    w = initial_w.copy()
    N = len(y)

    for _ in range(max_iters):
        e = y - np.dot(tx, w)
        gradient = -1/N * np.dot(tx.T, e)
        w = w - gamma * gradient

    e = y - np.dot(tx, w)
    loss = (1/(2*N)) * np.sum(e**2)

    return (w, loss)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent with batch size = 1.

    Parameters
    ----------
    y: np.ndarray
        Target values of shape (N,).
    tx: np.ndarray
        Feature matrix of shape (N, D).
    initial_w: np.ndarray
        Initial weight vector of shape (D,).
    max_iters: int
        Number of gradient descent iterations.
    gamma: float
        Learning rate.

    Returns
    ----------
    w: np.ndarray
        Final weight vector of shape (D,).
    loss: float
        Mean squared error loss.

    """
    w = initial_w.copy()
    N = len(y)

    for _ in range(max_iters):
        n = np.random.randint(N)
        x_b = tx[n]
        y_b = y[n]
        e = y_b - np.dot(x_b, w)
        gradient = -1 * e * x_b
        w = w - gamma * gradient

    e = y - np.dot(tx, w)
    loss = (1/(2*N)) * np.sum(e**2)
    
    return (w, loss)
