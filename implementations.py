import numpy as np


def mean_squared_error_gd(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> tuple[np.ndarray, float]:
    """
    Compute linear regression using gradient descent.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (N,).
    tx : np.ndarray
        Feature matrix of shape (N, D).
    initial_w : np.ndarray
        Initial weight vector of shape (D,).
    max_iters : int
        Number of gradient descent iterations.
    gamma : float
        Learning rate.

    Returns
    -------
    w : np.ndarray
        Final weight vector of shape (D,).
    loss : float
        Mean squared error loss.
    """
    w = initial_w.copy()
    N = len(y)

    for _ in range(max_iters):
        e = y - np.dot(tx, w)
        gradient = -1 / N * np.dot(tx.T, e)
        w = w - gamma * gradient

    e = y - np.dot(tx, w)
    loss = (1 / (2 * N)) * np.sum(e**2)

    return (w, loss)


def mean_squared_error_sgd(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> tuple[np.ndarray, float]:
    """
    Compute linear regression using stochastic gradient descent with batch size = 1.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (N,).
    tx : np.ndarray
        Feature matrix of shape (N, D).
    initial_w : np.ndarray
        Initial weight vector of shape (D,).
    max_iters : int
        Number of gradient descent iterations.
    gamma : float
        Learning rate.

    Returns
    -------
    w : np.ndarray
        Final weight vector of shape (D,).
    loss : float
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
    loss = (1 / (2 * N)) * np.sum(e**2)

    return (w, loss)


def least_squares(y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute the Least Squares solution.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (N,)
    tx : np.ndarray
        Features array of shape (N, D)

    Returns
    -------
    w : np.ndarray
        Optimal weights for the least squares algorithm with shape (D,)
    loss : float
        MSE loss for the input data with the optimal weights.

    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    txw = tx @ w
    loss = 1 / (2 * len(y)) * (y - txw).T @ (y - txw)

    return w, loss


def ridge_regression(
    y: np.ndarray, tx: np.ndarray, lambda_: float
) -> tuple[np.ndarray, float]:
    """
    Compute the Ridge regression solution.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (N,)
    tx : np.ndarray
        Features array of shape (N, D)
    lambda_ : float
        Regularisation parameter.


    Returns
    -------
    w : np.ndarray
        Optimal weights for the ridge regression algorithm with shape (D,)
    loss : float
        MSE loss for the input data with the optimal weights, not including the regularisation loss.

    """
    N = len(y)
    w = np.linalg.solve(
        tx.T @ tx + (lambda_ * (2 * N)) * np.identity(tx.shape[1]), tx.T @ y
    )
    txw = tx @ w
    loss = 1 / (2 * N) * (y - txw).T @ (y - txw)

    return (w, loss)


def sigmoid(x: np.ndarray):
    """
    Apply the sigmoid function.

    Parameters
    ----------
    x : np.ndarray
        Input array of arbitrary shape.

    Returns
    -------
    np.ndarray
        Array with the same shape as `x`, containing values in the interval (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))


def compute_loss_bce(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    Compute the binary cross-entropy loss.

    Parameters
    ----------
    y : np.ndarray
        Target labels of shape (N,), with values in {0, 1}.
    tx : np.ndarray
        Feature matrix of shape (N, D), where N is the number of
        samples and D is the number of features.
    w : np.ndarray
        Weight vector of shape (D,).

    Returns
    -------
    loss : float
        The average binary cross-entropy loss over all samples.
    """
    p = sigmoid(tx @ w)
    loss = np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return loss


def compute_gradient_bce(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the binary cross-entropy loss.

    Parameters
    ----------
    y : np.ndarray
        Target labels of shape (N,), with values in {0, 1}.
    tx : np.ndarray
        Feature matrix of shape (N, D), where N is the number of
        samples and D is the number of features.
    w : np.ndarray
        Weight vector of shape (D,).

    Returns
    -------
    gradient : np.ndarray
        Gradient of the binary cross-entropy loss with respect to `w`,
        with shape (D,).
    """
    p = sigmoid(tx @ w)
    gradient = tx.T @ (p - y) / y.shape[0]

    return gradient


def compute_gradient_reg_bce(
    y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_: float
) -> np.ndarray:
    """
    Compute the gradient of the L2 regularized binary cross-entropy loss.

    Parameters
    ----------
    y : np.ndarray
        Target labels of shape (N,), with values in {0, 1}.
    tx : np.ndarray
        Feature matrix of shape (N, D), where N is the number of
        samples and D is the number of features.
    w : np.ndarray
        Weight vector of shape (D,).
    lambda_ : float
        L2 regularization parameter.

    Returns
    -------
    gradient : np.ndarray
        Gradient of the binary cross-entropy loss with respect to `w`,
        with shape (D,).
    """
    gradient = compute_gradient_bce(y, tx, w) + 2 * lambda_ * w

    return gradient


def logistic_regression(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> tuple[np.ndarray, float]:
    """
    Perform logistic regression using gradient descent.

    Parameters
    ----------
    y : np.ndarray
        Target labels of shape (N,), with values in {0, 1}.
    tx : np.ndarray
        Feature matrix of shape (N, D), where N is the number of
        samples and D is the number of features.
    initial_w : np.ndarray
        Initial weight vector of shape (D,).
    max_iters : int
        Number of gradient descent iterations.
    gamma : float
        Step size used for the gradient descent updates.

    Returns
    -------
    w : np.ndarray
        Final weight vector after `max_iters` gradient descent steps,
        with shape (D,).
    loss : float
        Binary cross-entropy loss evaluated at the final weight vector.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        gradient = compute_gradient_bce(y, tx, w)
        w = w - gamma * gradient

    loss = compute_loss_bce(y, tx, w)

    return w, loss


def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> tuple[np.ndarray, float]:
    """
    Perform L2 regularized logistic regression using gradient descent.

    Parameters
    ----------
    y : np.ndarray
        Target labels of shape (N,), with values in {0, 1}.
    tx : np.ndarray
        Feature matrix of shape (N, D), where N is the number of
        samples and D is the number of features.
    lambda_ : float
        L2 regularization parameter.
    initial_w : np.ndarray
        Initial weight vector of shape (D,).
    max_iters : int
        Number of gradient descent iterations.
    gamma : float
        Step size used for the gradient descent updates.

    Returns
    -------
    w : np.ndarray
        Final weight vector after `max_iters` gradient descent steps,
        with shape (D,).
    loss : float
        Binary cross-entropy loss evaluated at the final weight vector.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        gradient = compute_gradient_reg_bce(y, tx, w, lambda_)
        w = w - gamma * gradient

    loss = compute_loss_bce(y, tx, w)

    return w, loss
