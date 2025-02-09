import streamlit as st
from feature_engine.discretisation import ArbitraryDiscretiser
import numpy as np
import pandas as pd
import plotly.express as px
import ppscore as pps
from src.data_management import load_ames_data


def correlation_analysis_body():
    # load data
    df = load_ames_data()

    vars_to_study = [
        '1stFlrSF', 'GarageArea', 'GarageYrBlt', 'GrLivArea',
        'KitchenQual_Ex', 'KitchenQual_TA', 'OverallQual',
        'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd'
    ]

    st.write("## Correlation Analysis")
    st.write("The client is interested in analyzing how different house "
             "features impact sale prices. Therefore, they are looking for "
             "data visualizations that highlight the connection between "
             "these features and the sale price.")

    # inspect data
    if st.checkbox("🏠 Inspect house data from the dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns")
        st.write(df)

    st.write("---")

    # Correlation Study Summary
    st.write("In this analysis, we explored the relationships between various "
             "factors and house sale prices. The primary goal was to identify "
             "key variables that influence pricing trends. After conducting a "
             "thorough correlation analysis, the variables most closely "
             "linked to the sale price are: ")
    st.write(f"**{vars_to_study}**")

    st.info(
        "The correlation analysis and plot interpretations lead to the "
        "following key conclusions: \n\n"
        "* Larger homes are more valuable: Homes with larger square footage "
        "and additional features show a strong correlation with higher "
        "market values. This was the most prominent finding in the analysis. "
        "Further investigation confirmed that overall quality is the most "
        "significant factor when determining feature importance. \n\n"
        "* Better condition and higher-quality features drive up value: Homes "
        "in better condition and with higher-quality materials tend to be "
        "more valuable, reinforcing the first hypothesis. \n\n"
        "* New or recently renovated homes hold higher value: Homes that are "
        "newly built or have undergone recent renovations typically have a "
        "higher value, confirming the second hypothesis."
    )

    df_eda = df.filter(vars_to_study + ['SalePrice'])
    target_var = 'SalePrice'
    st.write("#### Data Visualizations")

    # Plot to display the distribution of the target variable
    if st.checkbox("⚖️ Distribution of Target Variable"):
        plot_target_hist(df_eda, target_var)

    # Show heatmaps
    if st.checkbox("🔥 Show Correlation and PPS Heatmaps"):
        df_corr_pearson, df_corr_spearman, pps_matrix = (
            CalculateCorrAndPPS(df_eda))
        DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix,
                          CorrThreshold=0.4, PPS_Threshold=0.2)

    # functions that will lead to a display of plots of the variables

    def scatter_plot_for_eda(df, col, target_var):
        fig = px.scatter(df, x=col, y=target_var,
                         title=f"Scatter Plot of {col} vs {target_var}")
        st.plotly_chart(fig)

    def plot_categorical(df, col, target_var):
        fig = px.histogram(df, x=col, color=target_var,
                           title=f"Distribution of {col} vs {target_var}",
                           barmode='stack')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

    def variables_plots(df_eda):
        target_var = 'SalePrice'
        for col in df_eda.drop([target_var], axis=1).columns.to_list():
            if df_eda[col].dtype == 'object':
                plot_categorical(df_eda, col, target_var)
            else:
                scatter_plot_for_eda(df_eda, col, target_var)

    # Plots per variable
    if st.checkbox("🔍 Variables Plots - Visual Analysis"):
        variables_plots(df_eda)


def plot_target_hist(df, target_var):
    """
    Function for the histogram plot of the target variable
    """
    fig = px.histogram(df, x=target_var, marginal="box", nbins=50,
                       title=f"Distribution of {target_var}")
    st.plotly_chart(fig)


def heatmap_corr(df, threshold, figsize=(20, 12), font_annot=8):
    """
    Function to create heatmap using correlations.
    """
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mask[abs(df) < threshold] = True

        fig = px.imshow(df, title="Correlation Heatmap",
                        color_continuous_scale='viridis',
                        labels={'x': 'Features', 'y': 'Features'},
                        text_auto=True)
        st.plotly_chart(fig)


def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=8):
    """
    Function to create heatmap with pps.
    """
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=bool)
        mask[abs(df) < threshold] = True

        fig = px.imshow(df, title="PPS Heatmap",
                        color_continuous_scale='viridis',
                        labels={'x': 'Features', 'y': 'Features'},
                        text_auto=True)
        st.plotly_chart(fig)


def CalculateCorrAndPPS(df):
    """
    Function for calculation of correlations and pps.
    """
    df_corr_spearman = df.corr(method="spearman")
    df_corr_spearman.name = 'corr_spearman'
    df_corr_pearson = df.corr(method="pearson")
    df_corr_pearson.name = 'corr_pearson'

    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
        columns='x', index='y', values='ppscore')

    pps_score_stats = (pps_matrix_raw.query("ppscore < 1").filter(['ppscore'])
                       .describe().T)
    print(pps_score_stats.round(3))

    return df_corr_pearson, df_corr_spearman, pps_matrix


def DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix,
                      CorrThreshold, PPS_Threshold,
                      figsize=(20, 12), font_annot=8):
    """
    Function to display the correlations and pps.
    """
    heatmap_corr(df=df_corr_spearman, threshold=CorrThreshold, figsize=figsize,
                 font_annot=font_annot)
    heatmap_corr(df=df_corr_pearson, threshold=CorrThreshold, figsize=figsize,
                 font_annot=font_annot)
    heatmap_pps(df=pps_matrix, threshold=PPS_Threshold, figsize=figsize,
                font_annot=font_annot)
