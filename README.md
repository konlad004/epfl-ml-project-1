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
python run.py --method reg_logistic_regression --lambda_ 0.0001 --gamma 0.5 --max_iters 1000 --pred_threshold -1.35
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
| `validation.py` | Trains and evaluates different model configurations on a stratified 80/20 train-validation set |


## Pipeline
1. Preprocessing `feature_engineering.preprocess`. 
All statistics are computed on the training set only:
    - drop interview metadata
    - replace survey codes, (7, 9, 77, 99 for NaN and 88 for 0)
    - one-hot encoding for categorical values
    - fill missing values with the median and add a missing indicator column
    - standardize and add a bias column
2. Model - regularized logistic regression with gradient descent
3. Model selection - stratified 80/20 validation set over lambda, then the decision threshold that maximizes F1 on validation 


## Results

| Model | Validation F1 | AIcrowd F1 |
|---|---|---|
| Regularized logistic regression baseline median and standardization | -- | 0.412 |
| Regularized logistic regression and feature engineering (best overall validation configuration)| **0.43998** | 0.451 |
| Regularized logistic regression and feature engineering (manual configuration)| 0.4394 | **0.452** |
| Ridge regression and feature engineering (validation configuration)| 0.429 | 0.439 |


The configuration achieving the highest local validation F1 did not achieve the highest AIcrowd F1. A configuration ranked third during validation produced a higher leaderboard score. Both configurations are reported to compare validation performance with performance on the competition test set.


## Contributors
- Konstantinos LADAKIS
- Remigiusz SEK 
- Oskar WACHEK
