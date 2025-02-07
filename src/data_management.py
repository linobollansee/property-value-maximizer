import joblib
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_ames_data():
    df = pd.read_csv("outputs/datasets/collection/HousePricesRecords.csv")
    return df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)


@st.cache_data
def load_inherited_houses_data():
    df = pd.read_csv("outputs/datasets/collection/InheritedHouses.csv")
    return df
