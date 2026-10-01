import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
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
