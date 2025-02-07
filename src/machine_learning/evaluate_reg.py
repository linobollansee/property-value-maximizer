import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    st.write("Model Evaluation \n")
    st.info("* Train Set")
    regression_evaluation(X_train, y_train, pipeline)
    st.info("* Test Set")
    regression_evaluation(X_test, y_test, pipeline)


def regression_evaluation(X, y, pipeline):
    prediction = pipeline.predict(X)
    if isinstance(y, pd.DataFrame):
        y = y['SalePrice'].values
    mae = mean_absolute_error(y, prediction)
    r2 = r2_score(y, prediction)
    mse = mean_squared_error(y, prediction)
    rmse = np.sqrt(mse)
    st.write('Mean Absolute Error:', round(mae, 3))
    st.write('R2 Score:', round(r2, 3))
    st.write('Mean Squared Error:', round(mse, 3))
    st.write('Root Mean Squared Error:', round(rmse, 3))
    st.write("\n")
