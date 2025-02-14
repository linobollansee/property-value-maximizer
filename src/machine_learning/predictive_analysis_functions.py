import streamlit as st


def predict_price(X_live, house_features, price_pipeline):
    """
    Predicts the price of a house based on live input data.

    Parameters:
    X_live (DataFrame): The live input data containing house features.
    house_features (list): List of features relevant for price prediction.
    price_pipeline (Pipeline): Pre-trained machine learning pipeline for price
    prediction.

    Returns:
    ndarray: Predicted price(s) for the input house data.
    """
    # Select relevant features from the live input data
    X_live_price = X_live.filter(house_features)

    # Make a prediction using the pre-trained pipeline
    price_prediction = price_pipeline.predict(X_live_price)

    return price_prediction


def predict_inherited_house_price(X_inherited, house_features, price_pipeline):
    """
    Predicts the price of an inherited house based on its features.

    Parameters:
    X_inherited (DataFrame): Data containing features of the inherited house.
    house_features (list): List of features relevant for price prediction.
    price_pipeline (Pipeline): Pre-trained machine learning pipeline for price
    prediction.

    Returns:
    float: Predicted price of the inherited house.
    """
    # Select relevant features from the inherited house data
    X_inherited_price = X_inherited.filter(house_features)

    # Make a price prediction using the pre-trained pipeline
    price_prediction_inherited = price_pipeline.predict(X_inherited_price)

    # Extract the predicted price (assuming a single value output)
    this_price = price_prediction_inherited[0]

    return this_price
