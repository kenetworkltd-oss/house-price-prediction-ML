# House Price Prediction with Machine Learning

An end-to-end machine learning project for predicting residential house prices using the Kaggle House Prices dataset.

## Overview

This project applies supervised machine learning to predict `SalePrice` from residential property features. The workflow covers data cleaning, exploratory data analysis, feature preprocessing, model training, hyperparameter tuning, evaluation, and deployment preparation.

## Project Workflow

1. Data inspection
2. Data cleaning and missing-value handling
3. Exploratory data analysis
4. Feature and target separation
5. Train/test split
6. Categorical encoding and preprocessing
7. Model training and evaluation
8. Hyperparameter tuning
9. Final model selection
10. Prediction on new data
11. Model and preprocessing pipeline serialization

## Models

The following regression models were evaluated:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

Random Forest and Gradient Boosting were further tuned using selected hyperparameters.

## Model Performance

Evaluation was performed on the held-out test set using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R².

| Model                   |      RMSE |     R² |
| ----------------------- | --------: | -----: |
| Linear Regression       | 31,336.80 | 0.8720 |
| Random Forest           | 29,616.23 | 0.8856 |
| Tuned Random Forest     | 28,991.16 | 0.8904 |
| Gradient Boosting       | 28,794.58 | 0.8919 |
| Tuned Gradient Boosting | 26,229.43 | 0.9103 |

The tuned Gradient Boosting Regressor was selected as the final model based on the evaluation results.

## Preprocessing

Categorical features were transformed using `OneHotEncoder` within a `ColumnTransformer`, while numerical features were passed through unchanged.

The fitted preprocessing pipeline was saved separately to ensure that new data receives the same transformations used during model training.

## Deployment

The trained model and preprocessing pipeline were serialized using Joblib:

```text
house_price_model.pkl
house_price_preprocessor.pkl
```

These files can be loaded by a backend service such as FastAPI to accept house features and return predicted prices.

## Repository Structure

```text
House_Price _ML/
├── House_Price_Prediction.ipynb
├── data_description.txt
├── house_price_model.pkl
├── house_price_preprocessor.pkl
├── submission.csv
└── .gitignore
```

The original Kaggle `train.csv` and `test.csv` datasets are excluded from the repository.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook

## Dataset

The project uses the Kaggle House Prices: Advanced Regression Techniques dataset.

Dataset source: Kaggle House Prices competition.

## Future Improvements

* Cross-validation for more robust model evaluation
* Further hyperparameter optimization
* Feature engineering
* Model explainability
* FastAPI prediction endpoint
* Web-based prediction interface
