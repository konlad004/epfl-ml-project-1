import pytest

from implementations import mean_squared_error_gd, mean_squared_error_sgd, least_squares, ridge_regression, sigmoid, compute_gradient_reg_bce,  compute_loss_bce, logistic_regression, reg_logistic_regression

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
    expected_mse_loss = temp_e @ temp_e / (2*len(y))

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

@pytest.mark.parametrize("y, tx, initial_w, gamma, loss, w", [
    #randomly generated data with treating ML ex 2 solution as ground truth (probably does more harm than good due to precision)
   (np.array([np.float64(0.099), np.float64(0.273), np.float64(0.526), np.float64(0.443), np.float64(0.075), np.float64(0.18), np.float64(0.634), np.float64(0.957), np.float64(0.279), np.float64(0.825), np.float64(0.661), np.float64(0.153), np.float64(0.475), np.float64(0.509), np.float64(0.482), np.float64(0.244), np.float64(0.709), np.float64(0.042), np.float64(0.234), np.float64(0.962)]), np.array([[np.float64(0.993), np.float64(0.396)], [np.float64(0.017), np.float64(0.115)], [np.float64(0.314), np.float64(0.686)], [np.float64(0.695), np.float64(0.538)], [np.float64(0.613), np.float64(0.027)], [np.float64(0.159), np.float64(0.83)], [np.float64(0.514), np.float64(0.229)], [np.float64(0.952), np.float64(0.099)], [np.float64(0.91), np.float64(0.894)], [np.float64(0.341), np.float64(0.348)], [np.float64(0.805), np.float64(0.545)], [np.float64(0.614), np.float64(0.442)], [np.float64(0.305), np.float64(0.753)], [np.float64(0.915), np.float64(0.739)], [np.float64(0.039), np.float64(0.404)], [np.float64(0.681), np.float64(0.022)], [np.float64(0.134), np.float64(0.701)], [np.float64(0.079), np.float64(0.223)], [np.float64(0.659), np.float64(0.827)], [np.float64(0.873), np.float64(0.915)]]), np.array([np.float64(0.602), np.float64(0.058)]), 0.7, 0.044663032082412756, np.array([np.float64(0.3535951853539826), np.float64(0.40994447399087924)])),
    (np.array([np.float64(0.824), np.float64(0.113), np.float64(0.968), np.float64(0.153), np.float64(0.761), np.float64(0.884), np.float64(0.762), np.float64(0.416), np.float64(0.577), np.float64(0.146)]), np.array([[np.float64(0.028), np.float64(0.33), np.float64(0.09)], [np.float64(0.229), np.float64(0.512), np.float64(0.455)], [np.float64(0.296), np.float64(0.326), np.float64(0.356)], [np.float64(0.849), np.float64(0.877), np.float64(0.205)], [np.float64(0.942), np.float64(0.881), np.float64(0.789)], [np.float64(0.416), np.float64(0.815), np.float64(0.421)], [np.float64(0.21), np.float64(0.509), np.float64(0.71)], [np.float64(0.323), np.float64(0.219), np.float64(0.124)], [np.float64(0.987), np.float64(0.8), np.float64(0.263)], [np.float64(0.761), np.float64(0.272), np.float64(0.92)]]), np.array([np.float64(0.094), np.float64(0.444), np.float64(0.492)]), 0.7, 0.05743311077579155, np.array([np.float64(-0.6076329243475169), np.float64(1.1047600811832758), np.float64(0.43587213134516245)])),
    (np.array([np.float64(0.621), np.float64(0.841), np.float64(0.855), np.float64(0.418), np.float64(0.994)]), np.array([[np.float64(0.305), np.float64(0.857), np.float64(0.149), np.float64(0.288)], [np.float64(0.067), np.float64(0.262), np.float64(0.963), np.float64(0.051)], [np.float64(0.278), np.float64(0.51), np.float64(0.78), np.float64(0.211)], [np.float64(0.828), np.float64(0.748), np.float64(0.727), np.float64(0.312)], [np.float64(0.385), np.float64(0.52), np.float64(0.227), np.float64(0.53)]]), np.array([np.float64(0.978), np.float64(0.613), np.float64(0.32), np.float64(0.501)]), 0.5, 1.1209817367740587e-06, np.array([np.float64(-1.2594019767662268), np.float64(0.3169737028473679), np.float64(0.7588392665778424), np.float64(2.1529686168674735)]))
    ])

def test_gd_example_data(y, tx, initial_w, gamma, loss, w):

    w_gen, loss_gen = mean_squared_error_gd(y, tx, initial_w, 1000, gamma)

    np.testing.assert_allclose(w_gen, w, rtol = 1e-3)
    assert loss_gen == pytest.approx(loss, abs=1e-3)


#SGD

def test_sgd_perfect_start():
    tx = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 5.0]])
    w_true = np.array([0.5, 2.0])
    y = tx @ w_true

    w, loss = mean_squared_error_sgd(y, tx, w_true.copy(), 20, 0.1)
    np.testing.assert_allclose(w, w_true)
    assert loss == pytest.approx(0.0, abs=1e-12)


# least squares


def test_least_squares_exact():
    for tx, w_true in zip([np.array([[5.0, 0.0], [3.0, 1.0], [1.0, 6.0], [2.0, 3.0]]),
    np.array([[2.0, 6.0, 1.0], [17.0, 4.0, 5.0], [9.0, 4.0, 8.0], [6.0, 2.0, 11.0]])], [np.array([2.0, -1.0]), np.array([4.0,2.0, -6.0])]):

  
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

