import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from src.data_management import load_ames_data, load_pkl_file
from src.machine_learning.evaluate_reg import regression_performance


def load_pkl_file(file_path):
    """
    Loads a pickled file (model) using joblib from the given file path.

    Parameters:
    file_path (str): The path to the pickled file (model) to be loaded.

    Returns:
    model: The loaded machine learning model.
    """
    try:
        # Attempt to load the pickled file using joblib
        model = joblib.load(file_path)
        return model
    except Exception as e:
        # If an error occurs during loading, print the error and return None
        print(f"Error loading model from {file_path}: {e}")
        return None


def ml_price_prediction_body():
    """
    Displays the machine learning pipeline, feature importance, and regression
    performance plots.

    This method visualizes the trained machine learning pipeline used for
    predicting house prices and evaluates its performance on both the training
    and test sets.
    """
    # Define the version of the pipeline being used
    version = 'v1'

    # Load the regression pipeline (model)
    price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_price/{version}/regression_pipeline.pkl"
    )

    # Load the feature importance plot image
    price_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_price/{version}/features_importance.png"
    )

    # Load the training and test datasets (features and targets)
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/X_train.csv"
    )
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/X_test.csv"
    )
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/y_train.csv"
    )
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/y_test.csv"
    )

    # Display the title for the ML pipeline section
    st.write("## ML Pipeline: Predict House Price")

    # Show an introductory message about the ML pipeline
    st.info(
        "* To answer the BR2, a Regressor model was trained and the pipeline "
        "tuned aiming for at least 0.75 accuracy in predicting the sales "
        "price of a property with a set of attributes.\n"
        "* The pipeline performance for the best model on the train and test "
        "set is R2 = 0.869 and R2 = 0.847 respectively.\n"
        "* We present the pipeline steps, best features list along with "
        "feature importance plot, pipeline performance and regression "
        "performance report below."
    )
    st.write("---")

    # Display the ML pipeline code
    st.write("* ML pipeline to predict sales prices of houses ")
    st.code(price_pipe)  # Display the code of the pipeline
    st.write("---")

    # Show the features the model was trained on and their importance
    st.write("* The features the model was trained and their importance")
    st.write(X_train.columns.to_list())  # List of features used for training
    st.image(price_feat_importance)  # Display the feature importance plot
    st.write("---")

    # Display information about the pipeline performance goals
    st.write("### Pipeline Performance")
    st.write("##### Performance goal of the predictions:")
    st.write(
        "* We agreed with the client an R2 score of at least 0.75 on the "
        "train set as well as on the test set."
    )
    st.write(
        "* Our ML pipeline performance shows that our model performance "
        "metrics have been successfully satisfied."
    )

    # Call the regression performance evaluation function
    regression_performance(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        pipeline=price_pipe
    )

    # Display the regression performance plots
    st.write("### Regression Performance Plots")
    st.write(
        "* The regression performance plots below indicate that the model "
        "with the best features can predict sale prices well. For houses "
        "with higher prices, the model does, however, look to be less "
        "dependable."
    )

    # Display the regression performance plot
    image_path = 'docs/plots/regression_performance.png'
    st.image(image_path, caption="Regression Performance",
             use_container_width=True)
