import numpy as np
import argparse
import os
from pathlib import Path
from argparse import Namespace
from itertools import product
from datetime import datetime, timezone
from helpers import load_csv_data
from feature_engineering import preprocess, load_feature_names
from run import train_model, predict
import pandas as pd  # added only for csv export

file_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


METHODS = [
    "mean_squared_error_gd",
    "mean_squared_error_sgd",
    "least_squares",
    "ridge_regression",
    "logistic_regression",
    "reg_logistic_regression",
]


def split_data(
    y: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the labeled data into stratified training and validation subsets.

    Returns
    -------
    train_idx, val_idx : np.ndarray
        Row indices for the two subsets.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for label in (-1, 1):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        n_samples = max(1, min(len(indices) - 1, round(len(indices) * val_fraction)))
        val_idx.extend(indices[:n_samples])
        train_idx.extend(indices[n_samples:])

    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return train_idx, val_idx


def compute_metrics(y: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics on the validation set.

    Returns
    -------
    Dictionary containing validation F1 score, Accuracy score, Precision score and Sensitivity score
    """
    tp = np.sum((y == 1) & (y_pred == 1))
    fp = np.sum((y == -1) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == -1))

    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )

    return {
        "val_f1": float(f1),
        "val_accuracy": float(np.mean(y == y_pred)),
        "val_precision": float(precision),
        "val_sensitivity": float(sensitivity),
    }


def get_configurations(args: Namespace) -> list[tuple]:
    """
    Build the parameter combinations needed for each selected method.
    Unused hyperparameters are set to None in the results file.
    """
    configurations = []

    for method in args.methods:
        if method == "least_squares":
            configurations.append((method, None, None, None))
        elif method == "ridge_regression":
            for lambda_ in args.lambdas:
                configurations.append((method, None, None, lambda_))
        elif method == "mean_squared_error_sgd":
            for max_iters, gamma in product(args.max_iters, args.gammas):
                configurations.append((method, max_iters, gamma, None))
        elif method in ("mean_squared_error_gd", "logistic_regression"):
            for max_iters, gamma in product(args.max_iters, args.gammas):
                configurations.append((method, max_iters, gamma, None))
        elif method == "reg_logistic_regression":
            for max_iters, gamma, lambda_ in product(
                args.max_iters, args.gammas, args.lambdas
            ):
                configurations.append((method, max_iters, gamma, lambda_))

    return configurations


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=METHODS,
        default=["ridge_regression", "reg_logistic_regression"],
    )
    parser.add_argument("--seed", type=int, default=7555)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--max_iters", type=int, nargs="+", default=[1000])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=[0.0001, 0.001, 0.01, 0.1]
    )
    parser.add_argument(
        "--pred_thresholds",
        type=float,
        nargs="+",
        default=np.round(np.arange(-2, 1.01, 0.05), 2).tolist(),
    )
    parser.add_argument(
        "--data_dir", type=Path, default=Path(__file__).parent / "Data" / "dataset"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path(__file__).parent / "Data" / "results"
    )
    parser.add_argument(
        "--output_name", type=str, default=f"validation_{file_suffix}.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    x_train, _, y_train, _, _ = load_csv_data(args.data_dir)

    names = load_feature_names(args.data_dir)
    train_idx, val_idx = split_data(y_train, args.val_fraction, args.seed)
    x_fit, x_val, _ = preprocess(x_train[train_idx], x_train[val_idx], names)
    y_fit, y_val = y_train[train_idx], y_train[val_idx]
    del x_train

    configurations = get_configurations(args)
    output_path = args.output_dir / args.output_name

    rows = []
    for i, (method, max_iters, gamma, lambda_) in enumerate(configurations, start=1):
        np.random.seed(args.seed)
        weights, loss = train_model(
            method,
            y_fit,
            x_fit,
            max_iters if max_iters is not None else 1000,
            gamma if gamma is not None else 0.01,
            lambda_ if lambda_ is not None else 0.01,
        )
        scores = x_val @ weights

        for threshold in args.pred_thresholds:
            y_pred = predict(scores, threshold)
            rows.append(
                {
                    "method": method,
                    "lambda_": lambda_,
                    "gamma": gamma,
                    "max_iters": max_iters,
                    "train_loss": float(loss),
                    "train_size": len(y_fit),
                    "val_size": len(y_val),
                    "val_fraction": args.val_fraction,
                    "seed": args.seed,
                    "pred_threshold": threshold,
                    **compute_metrics(y_val, y_pred),
                }
            )

        pd.DataFrame(rows).sort_values(by="val_f1", ascending=False).to_csv(
            output_path, index=False
        )
