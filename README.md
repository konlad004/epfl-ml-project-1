# epfl-ml-project-1
EPFL CS-433 Machine Learning Project 1 (Fall 2026). 

The aim of the project is predicting coronary heart disease from the BRFSS 2015 health survey with linear models implemented from scratch in Numpy.

## Setup

Requires Python 3.13.

```bash
pip install -r requirements.txt
```

Put `x_train.csv`, `y_train.csv` and `x_test.csv` files in `Data/dataset/` directory.

Optional tools used only by `submit.py` are listed in `requirements.txt` file.

## Reproducing our best submission
The following command writes the submission CSV and a JSON file with the parameters to `Data/results/`.

```bash
python run.py --method reg_logistic_regression --lambda_ hyperparameter_value --gamma hyperparameter_value --max_iters hyperparameter_value --pred_threshold hyperparameter_value
```

## Files

| File | Description |
|---|---|
| `implementations.py` | The 6 required methods |
| `feature_engineering.py` | Data cleaning and feature processing |
| `run.py` | Trains a model and writes the AIcrowd submission |
| `helpers.py` | CSV loading and submission writing from the course sources |
| `test_implementations.py` | Unit tests for `implementations.py` |
| `submit.py` | Upload script for submissions via the AIcrowd API (It is optional and requires `aicrowd-cli` and `python-dotenv` additionally) |


## Pipeline
1. Preprocessing `feature_engineering.preprocess`. 
All statistics are computed on the training set only:
    - drop interview metadata
    - replace survey codes, (7, 9, 77, 99 for NaN and 88 for 0)
    - one-hot encoding for categorical values
    - fill missing values with the median and add a missing indicator column
    - standardize and add a bias column
2. Model - regularized logistic regression with gradient descent
3. (To add) Model selection - k-fold cross-validation over lambda, then the decision threshold that maximizes F1 on validation 


## Results

| Model | Validation F1 | AIcrowd F1 |
|---|---|---|
| baseline median and standardization | real value | real value |
| Regularized logistic regression and feature engineering | real value | real value |

## Contributors
- Konstantinos LADAKIS
- Remigiusz SEK 
- Oskar WACHEK