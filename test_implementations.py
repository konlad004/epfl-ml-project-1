import pytest

from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    sigmoid,
    compute_loss_bce,
    logistic_regression,
    reg_logistic_regression,
)

import numpy as np

# GD


def test_gd_single_step():
    y = np.array([0.1, 0.3, 0.5])
    tx = np.array([[2.3, 1.2], [1.0, 0.1], [1.4, 2.3]])
    w0 = np.array([0.5, 0.7])
    gamma = 0.1

    grad = -tx.T @ (y - tx @ w0) / len(y)
    expected_w = w0 - gamma * grad

    temp_e = y - tx @ expected_w
    expected_mse_loss = temp_e @ temp_e / (2 * len(y))

    w, loss = mean_squared_error_gd(y, tx, w0, 1, gamma)

    np.testing.assert_allclose(w, expected_w)
    np.testing.assert_allclose(loss, expected_mse_loss)


def test_gd_perfect_start():
    tx = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 5.0]])
    w_true = np.array([0.5, 2.0])
    y = tx @ w_true

    w, loss = mean_squared_error_gd(y, tx, w_true.copy(), 10, 0.1)
    np.testing.assert_allclose(w, w_true)
    assert loss == pytest.approx(0.0, abs=1e-12)

# SGD


def test_sgd_perfect_start():
    tx = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 5.0]])
    w_true = np.array([0.5, 2.0])
    y = tx @ w_true

    w, loss = mean_squared_error_sgd(y, tx, w_true.copy(), 20, 0.1)
    np.testing.assert_allclose(w, w_true)
    assert loss == pytest.approx(0.0, abs=1e-12)


# least squares


def test_least_squares_exact():
    for tx, w_true in zip(
        [
            np.array([[5.0, 0.0], [3.0, 1.0], [1.0, 6.0], [2.0, 3.0]]),
            np.array(
                [[2.0, 6.0, 1.0], [17.0, 4.0, 5.0], [9.0, 4.0, 8.0], [6.0, 2.0, 11.0]]
            ),
        ],
        [np.array([2.0, -1.0]), np.array([4.0, 2.0, -6.0])],
    ):

        y = tx @ w_true

        w, loss = least_squares(y, tx)

        np.testing.assert_allclose(w, w_true, atol=1e-8)
        assert loss == pytest.approx(0.0, abs=1e-10)


# ridge


def test_ridge_equals_least_squares():
    y = np.array([0.1, 0.3, 0.5])
    tx = np.array([[2.3, 1.2], [1.0, 0.1], [1.4, 2.3]])

    w_ridge, loss_ridge = ridge_regression(y, tx, 0.0)
    w_ls, loss_ls = least_squares(y, tx)

    np.testing.assert_allclose(w_ridge, w_ls, atol=1e-8)
    np.testing.assert_allclose(loss_ridge, loss_ls)


def test_ridge_lambda_one():
    # values from the main tests
    y = np.array([0.1, 0.2])
    tx =  np.array([[2.3, 3.2], [1., 0.1]])

    w, loss = ridge_regression(y, tx, 1.0)

    np.testing.assert_allclose(w, [0.03947092, 0.00319628], rtol=1e-4)
    np.testing.assert_allclose(loss, 0.0064170226, rtol=1e-4)
    assert w.shape == (2,)
    assert np.ndim(loss) == 0

# logistic


def test_logistic_single_step():
    y = np.array([1.0, 0.0, 0.0])
    tx = np.array([[2.3, 1.2], [1.0, 0.1], [1.4, 2.3]])
    w0 = np.array([0.5, 0.7])
    gamma = 0.7

    grad = tx.T @ (sigmoid(tx @ w0) - y) / len(y)
    expected_w = w0 - gamma * grad

    w, loss = logistic_regression(y, tx, w0, 1, gamma)

    np.testing.assert_allclose(w, expected_w)
    np.testing.assert_allclose(loss, compute_loss_bce(y, tx, expected_w))


# logistic regularised


def test_logistic_reg_single_step():
    y = np.array([1.0, 0.0, 1.0])
    tx = np.array([[2.3, 1.2], [1.0, 0.1], [1.4, 2.3]])
    w0 = np.array([0.5, 0.7])
    gamma = 0.7
    lambda_ = 0.3

    grad = tx.T @ (sigmoid(tx @ w0) - y) / len(y) + 2 * lambda_ * w0
    expected_w = w0 - gamma * grad

    w, loss = reg_logistic_regression(y, tx, lambda_, w0, 1, gamma)

    np.testing.assert_allclose(w, expected_w)
    np.testing.assert_allclose(loss, compute_loss_bce(y, tx, expected_w))


def test_reg_logistic_official_values():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    tx = np.array([[0.1, 0.4], [0.7, 0.2], [0.0, 0.1], [1.2, 0.7]])
    w0 = np.array([0.2, 0.5])
    
    w, loss = reg_logistic_regression(y, tx, 0.5, w0, 3, 0.1)

    np.testing.assert_allclose(w, [0.145983, 0.363036], rtol=1e-4)
    np.testing.assert_allclose(loss, 0.68429, rtol=1e-4)

    assert w.shape == (2,)
    assert np.ndim(loss) == 0