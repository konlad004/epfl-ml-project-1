import numpy as np
import argparse
import os
from pathlib import Path
from argparse import Namespace
from helpers import load_csv_data, create_csv_submission
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression
)


METHODS = [
    'mean_squared_error_gd',
    'mean_squared_error_sgd',
    'least_squares',
    'ridge_regression',
    'logistic_regression',
    'reg_logistic_regression'
]


def train_model(method: str, y: np.ndarray, x: np.ndarray, max_iters: int, gamma: float, lambda_: float):
    if method =='mean_squared_error_gd':
        return mean_squared_error_gd(y, x, np.zeros(x.shape[1]), max_iters, gamma)
    if method =='mean_squared_error_sgd':
        return mean_squared_error_sgd(y, x, np.zeros(x.shape[1]), max_iters, gamma)
    if method =='least_squares':
        return least_squares(y, x)
    if method =='ridge_regression':
        return ridge_regression(y, x, lambda_)
    if method =='logistic_regression':
        # Here we need to convert -1 labels to 0 since BCE loss expects the values to be either 0 or 1.
        return logistic_regression((y_train == 1).astype(np.int64), x, np.zeros(x.shape[1]), max_iters, gamma)
    if method =='reg_logistic_regression':
        # Same conversion as in logistic regression.
        return reg_logistic_regression((y_train == 1).astype(np.int64), x, lambda_, np.zeros(x.shape[1]), max_iters, gamma)
    raise ValueError(f'Method {method} not implemented!')


def predict(y_pred: float, threshold: float) -> np.ndarray:
    return np.where(y_pred > threshold, 1, -1)


def parse_args() -> Namespace:
    # TODO: Finetune the hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, default=7555)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=1e-6)
    parser.add_argument('--lambda_', type=float, default=1e-2)
    parser.add_argument('--pred_threshold', type=float, default=0.5) 
    parser.add_argument('--data_dir', type=Path, default=Path(__file__).parent / 'Data' / 'dataset')
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).parent / 'Data' / 'results')
    parser.add_argument('--output_name', type=str, default='submission.csv')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(args.data_dir)
    
    # TODO: Proper preprocessing of the data!
    train_means = np.nanmean(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), train_means, x_train)
    x_test = np.where(np.isnan(x_test), train_means, x_test)
    
    weights, loss = train_model(args.method, y_train, x_train, args.max_iters, args.gamma, args.lambda_)
    y_pred = predict(x_test @ weights, args.pred_threshold)
    
    create_csv_submission(test_ids, y_pred, args.output_dir / args.output_name)


