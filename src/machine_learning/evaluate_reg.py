import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
                            r2_score, explained_variance_score, \
                            median_absolute_error


def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    """
    Evaluates the performance of a regression model on both training and test
    datasets.

    Parameters:
    X_train (DataFrame or ndarray): Training feature set.
    y_train (Series or ndarray): Target variable for training set.
    X_test (DataFrame or ndarray): Test feature set.
    y_test (Series or ndarray): Target variable for test set.
    pipeline (sklearn Pipeline): Trained pipeline model.

    Returns:
    None
    """
    st.write("Model Evaluation \n")
    st.info("* Train Set")
    regression_evaluation(X_train, y_train, pipeline)
    st.info("* Test Set")
    regression_evaluation(X_test, y_test, pipeline)


def regression_evaluation(X, y, pipeline):
    """
    Computes and displays various regression evaluation metrics for a given
    dataset.

    Parameters:
    X (DataFrame or ndarray): Feature set.
    y (Series, DataFrame, or ndarray): Target variable.
    pipeline (sklearn Pipeline): Trained pipeline model.

    Returns:
    None
    """
    # Generate predictions using the pipeline model
    prediction = pipeline.predict(X)

    # If target variable is a DataFrame, extract the values
    if isinstance(y, pd.DataFrame):
        y = y['SalePrice'].values

    # Compute evaluation metrics
    mae = mean_absolute_error(y, prediction)  # Mean Absolute Error
    r2 = r2_score(y, prediction)  # R-squared score
    mse = mean_squared_error(y, prediction)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y - prediction) / y)) * 100
    evs = explained_variance_score(y, prediction)  # Explained Variance Score
    medae = median_absolute_error(y, prediction)  # Median Absolute Error

    # Displaying all metrics
    st.write('Mean Absolute Error:', round(mae, 3))
    st.write('R2 Score:', round(r2, 3))
    st.write('Mean Squared Error:', round(mse, 3))
    st.write('Root Mean Squared Error:', round(rmse, 3))
    st.write('Mean Absolute Percentage Error:', round(mape, 3))
    st.write('Explained Variance Score:', round(evs, 3))
    st.write('Median Absolute Error:', round(medae, 3))
    st.write("\n")
