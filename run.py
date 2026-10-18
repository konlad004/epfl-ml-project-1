import numpy as np
import argparse
import os
from pathlib import Path
from argparse import Namespace
from helpers import load_csv_data, create_csv_submission
import json
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression
)
from datetime import datetime, timezone

# This is to keep track of submission/configuration file pairs
file_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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
        return logistic_regression((y == 1).astype(np.int64), x, np.zeros(x.shape[1]), max_iters, gamma)
    if method =='reg_logistic_regression':
        # Same conversion as in logistic regression.
        return reg_logistic_regression((y == 1).astype(np.int64), x, lambda_, np.zeros(x.shape[1]), max_iters, gamma)
    raise ValueError(f'Method {method} not implemented!')


def predict(y_pred: float, threshold: float) -> np.ndarray:
    return np.where(y_pred > threshold, 1, -1)


def preprocess(x_train, x_test):
    # Replace nans with respective means of training set columns
    means = np.nanmean(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), means, x_train)
    x_test = np.where(np.isnan(x_test), means, x_test)

    # Remove constant columns
    std = np.std(x_train, axis=0)
    x_train = x_train[:, std > 1e-12]
    x_test = x_test[:, std > 1e-12]

    # Standardize
    means = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - means) / std
    x_test = (x_test - means) / std

    # Add intercept
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_test = np.c_[np.ones(len(x_test)), x_test]

    return x_train, x_test


def create_metadata_file(args, loss, path):
    metadata = {
        **{k: v for k, v in args.__dict__.items() if k not in ("data_dir", "output_dir")}, 
        'train_loss': float(loss)
    }
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=4)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, default=7555)
    parser.add_argument('--max_iters', type=int, default=1_000)
    parser.add_argument('--gamma', type=float, default=1e-2)
    parser.add_argument('--lambda_', type=float, default=1e-2)
    parser.add_argument('--pred_threshold', type=float, default=0) 
    parser.add_argument('--data_dir', type=Path, default=Path(__file__).parent / 'Data' / 'dataset')
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).parent / 'Data' / 'results')
    parser.add_argument('--output_name', type=str, default=f'submission_{file_suffix}.csv')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(args.data_dir)
    
    # TODO: Proper preprocessing of the data!
    x_train, x_test = preprocess(x_train, x_test)
    
    weights, loss = train_model(args.method, y_train, x_train, args.max_iters, args.gamma, args.lambda_)
    y_pred = predict(x_test @ weights, args.pred_threshold)

    submission_path = args.output_dir / args.output_name
    create_csv_submission(test_ids, y_pred, submission_path)
    create_metadata_file(args, loss, submission_path)
