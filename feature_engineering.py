import numpy as np

from helpers import load_csv_data

DATA_PATH = "Data"
OUTPUT_PATH = "processed_data.npz"

METADATA_COLUMNS = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "QSTVER",
    "QSTLANG",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_DUALUSE",
    "_DUALCOR",
    "_LLCPWT",
]

MAX_CATEGORIES = 10


def load_feature_names(data_path: str) -> list[str]:
    """
    Read column names of training data.

    Parameters
    ----------
    data_path : str
        Folder containing training data.

    Returns
    -------
    list[str]
        Names of feature columns.
    """
    with open(f"{data_path}/x_train.csv") as f:
        return f.readline().strip().split(",")[1:]


def replace_codes(x: np.ndarray, column_max: np.ndarray) -> np.ndarray:
    """
    Replace codes for answers like "I don't know", "refused" and "none".

    Parameters
    ----------
    x : np.ndarray
        Data of shape (N, D).
    column_max : np.ndarray
        Largest value of each column in the training set.

    Returns
    -------
    x : np.ndarray
        Data with the codes replaced.
    """
    for j in range(x.shape[1]):
        col = x[:, j]

        if column_max[j] == 9:
            col[(col == 7) | (col == 9)] = np.nan

        elif column_max[j] in (98, 99):
            col[(col == 77) | (col == 99)] = np.nan
            col[col == 88] = 0

        elif column_max[j] == 999:
            col[(col == 777) | (col == 999)] = np.nan
            col[col == 888] = 0

    return x


def standardize(x: np.ndarray, mean=None, std=None):
    """
    Standardize each column to zero mean and unit variance.

    Parameters
    ----------
    x : np.ndarray
        Data of shape (N, D).
    mean, std : np.ndarray, optional
        Statistics of the training set.

    Returns
    -------
    x : np.ndarray
        Standardized data of shape (N, D).
    mean, std : np.ndarray
        Statistics of shape (D,).
    """
    if mean is None:
        mean = np.array([x[:, k].mean(dtype=np.float64) for k in range(x.shape[1])])
        std = np.array([x[:, k].std(dtype=np.float64) for k in range(x.shape[1])])

        std[std == 0] = 1.0

    x -= mean
    x /= std

    return x, mean, std


def preprocess(x_train: np.ndarray, x_test: np.ndarray, names: list[str]):
    """
    Turn the raw answers into feature matrices for training.

    Parameters
    ----------
    x_train : np.ndarray
        Raw training data of shape (N x_train, D).
    x_test : np.ndarray
        Raw test data of shape (N x_test, D).
    names : list[str]
        Column names of the raw data.

    Returns
    -------
    tx_train : np.ndarray
        Training features of shape (N x_train, F + 1).
        The first column is the bias.
    tx_test : np.ndarray
        Test features of shape (N_test, F + 1).
    feature_names : list[str]
        Names of the F feature columns.
    """
    # Drop metadata
    keep = [j for j, name in enumerate(names) if name not in METADATA_COLUMNS]
    x_train, x_test = x_train[:, keep], x_test[:, keep]
    names = [names[j] for j in keep]

    # Replace codes
    column_max = np.array(
        [np.nanmax(col) if np.any(~np.isnan(col)) else 0 for col in x_train.T]
    )
    x_train = replace_codes(x_train, column_max)
    x_test = replace_codes(x_test, column_max)

    # Building columns
    train_columns, test_columns, feature_names = [], [], []

    for j, name in enumerate(names):
        col_train, col_test = x_train[:, j], x_test[:, j]
        values = np.unique(col_train[~np.isnan(col_train)])

        if 2 < len(values) <= MAX_CATEGORIES:
            # for categorical answers, the first value is left to not to sum to 1
            for value in values[1:]:
                train_columns.append(col_train == value)
                test_columns.append(col_test == value)

                feature_names.append(f"{name}={value:g}")

        elif len(values) >= 2:
            # for numerical or boolean answers
            # skipping single value columns
            median = np.median(col_train[~np.isnan(col_train)])

            col_train = np.where(np.isnan(col_train), median, col_train)
            col_test = np.where(np.isnan(col_test), median, col_test)

            train_columns.append(col_train.astype(np.float32))
            test_columns.append(col_test.astype(np.float32))

            feature_names.append(name)

        # a missing answer says something about the previous answers
        missing_fraction = np.isnan(x_train[:, j]).mean()
        if 0 < missing_fraction < 1:
            train_columns.append(np.isnan(x_train[:, j]))
            test_columns.append(np.isnan(x_test[:, j]))

            feature_names.append(f"{name}_missing")

    x_train = np.column_stack(train_columns)
    x_test = np.column_stack(test_columns)

    # Standardization with training statistics and bias
    x_train, mean, std = standardize(x_train)
    x_test, _, _ = standardize(x_test, mean, std)

    tx_train = np.c_[np.ones(len(x_train), dtype=np.float32), x_train]
    tx_test = np.c_[np.ones(len(x_test), dtype=np.float32), x_test]

    return tx_train, tx_test, feature_names


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)
    names = load_feature_names(DATA_PATH)

    tx_train, tx_test, feature_names = preprocess(x_train, x_test, names)

    # logistic regression needs labels in {0, 1} instead of {-1, 1}
    y_train = (y_train == 1).astype(np.float32)

    print("Raw features", x_train.shape[1], "processed: ", len(feature_names))
    print("tx_train: ", tx_train.shape, "tx_test: ", tx_test.shape)

    np.savez_compressed(
        OUTPUT_PATH,
        tx_train=tx_train,
        tx_test=tx_test,
        y_train=y_train,
        train_ids=train_ids,
        test_ids=test_ids,
        feature_names=np.array(feature_names),
    )

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
