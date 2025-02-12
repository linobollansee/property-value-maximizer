import streamlit as st
from feature_engine.discretisation import ArbitraryDiscretiser
import numpy as np
import pandas as pd
import plotly.express as px
import ppscore as pps
from src.data_management import load_ames_data


def correlation_analysis_body():
    # Load the Ames housing dataset
    df = load_ames_data()

    # List of variables to study in the analysis
    vars_to_study = [
        '1stFlrSF', 'GarageArea', 'GarageYrBlt', 'GrLivArea',
        'KitchenQual_Ex', 'KitchenQual_TA', 'OverallQual',
        'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd'
    ]

    # Displaying initial text and explanation
    st.write("## Correlation Analysis")
    st.write("The client is interested in analyzing how different house "
             "features impact sale prices. Therefore, they are looking for "
             "data visualizations that highlight the connection between "
             "these features and the sale price.")

    # Optional: Display the raw dataset
    if st.checkbox("🏠 Inspect house data from the dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns")
        st.write(df)

    st.write("---")

    # Summary of the correlation study and key variables
    st.write("In this analysis, we explored the relationships between various "
             "factors and house sale prices. The primary goal was to identify "
             "key variables that influence pricing trends. After conducting a "
             "thorough correlation analysis, the variables most closely "
             "linked to the sale price are: ")
    st.write(f"**{vars_to_study}**")

    # Key conclusions based on the correlation study
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

    # Subset of data for analysis (target variable and selected features)
    df_eda = df.filter(vars_to_study + ['SalePrice'])
    target_var = 'SalePrice'
    st.write("#### Data Visualizations")

    # Display histogram of the target variable's distribution
    if st.checkbox("⚖️ Distribution of Target Variable"):
        plot_target_hist(df_eda, target_var)

    # Display correlation and PPS (Predictive Power Score) heatmaps
    if st.checkbox("🔥 Show Correlation and PPS Heatmaps"):
        df_corr_pearson, df_corr_spearman, pps_matrix = (
            CalculateCorrAndPPS(df_eda))
        DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix,
                          CorrThreshold=0.4, PPS_Threshold=0.2)

    # Function to display scatter plots or categorical distribution plots
    def scatter_plot_for_eda(df, col, target_var):
        """
        Function to create a scatter plot between a feature and the target
        variable.
        """
        fig = px.scatter(df, x=col, y=target_var,
                         title=f"Scatter Plot of {col} vs {target_var}",
                         trendline="ols", trendline_color_override="red")
        st.plotly_chart(fig)

    def plot_categorical(df, col, target_var):
        """
        Function to create a stacked histogram for categorical variables vs
        target.
        """
        fig = px.histogram(df, x=col, color=target_var,
                           title=f"Distribution of {col} vs {target_var}",
                           barmode='stack')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

    def variables_plots(df_eda):
        """
        Function to plot either scatter plots or categorical plots for all
        selected variables.
        """
        target_var = 'SalePrice'
        # Iterate over all variables and plot according to type
        for col in df_eda.drop([target_var], axis=1).columns.to_list():
            if df_eda[col].dtype == 'object':
                plot_categorical(df_eda, col, target_var)
            else:
                scatter_plot_for_eda(df_eda, col, target_var)

    # Display visual analysis for each selected variable
    if st.checkbox("🔍 Variables Plots - Visual Analysis"):
        variables_plots(df_eda)


def plot_target_hist(df, target_var):
    """
    Function to plot a histogram for the target variable.

    Args:
        df: DataFrame containing the data.
        target_var: Name of the target variable to plot.
    """
    fig = px.histogram(df, x=target_var, marginal="box", nbins=50,
                       title=f"Distribution of {target_var}")
    st.plotly_chart(fig)


def heatmap_corr(df, threshold, figsize=(20, 12), font_annot=8):
    """
    Function to generate a correlation heatmap for numerical features in the
    DataFrame.
    Values below the specified threshold are hidden in the plot.

    Args:
        df: DataFrame containing the correlation matrix.
        threshold: Correlation value below which cells will be hidden.
        figsize: Optional tuple to set the size of the heatmap.
        font_annot: Optional font size for annotations in the heatmap.
    """
    if len(df.columns) > 1:
        # Create a mask to hide upper triangle and correlations below the
        # threshold
        mask = np.zeros_like(df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mask[abs(df) < threshold] = True

        # Apply mask to the correlation matrix
        df_masked = df.mask(mask)

        # Plot heatmap using Plotly
        fig = px.imshow(
            df_masked,
            title="Correlation Heatmap",
            color_continuous_scale='viridis',
            labels={'x': 'Features', 'y': 'Features'},
            text_auto=True
        )
        st.plotly_chart(fig)


def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=8):
    """
    Function to generate a Predictive Power Score (PPS) heatmap.
    Values below the specified threshold are hidden in the plot.

    Args:
        df: DataFrame containing the PPS matrix.
        threshold: PPS value below which cells will be hidden.
        figsize: Optional tuple to set the size of the heatmap.
        font_annot: Optional font size for annotations in the heatmap.
    """
    if len(df.columns) > 1:
        # Create a mask to hide PPS below the threshold
        mask = np.zeros_like(df, dtype=bool)
        mask[abs(df) < threshold] = True

        # Apply mask to the PPS matrix
        df_masked = df.mask(mask)

        # Plot heatmap using Plotly
        fig = px.imshow(
            df_masked,
            title="PPS Heatmap",
            color_continuous_scale='viridis',
            labels={'x': 'Features', 'y': 'Features'},
            text_auto=True
        )
        st.plotly_chart(fig)


def CalculateCorrAndPPS(df):
    """
    Function to calculate both the Pearson and Spearman correlations as well as
    the Predictive Power Score (PPS) matrix for the dataset.

    Args:
        df: DataFrame containing the features to analyze.

    Returns:
        df_corr_pearson: Pearson correlation matrix.
        df_corr_spearman: Spearman correlation matrix.
        pps_matrix: Matrix of PPS values between the features.
    """
    # Calculate correlation matrices
    df_corr_spearman = df.corr(method="spearman")
    df_corr_spearman.name = 'corr_spearman'
    df_corr_pearson = df.corr(method="pearson")
    df_corr_pearson.name = 'corr_pearson'

    # Calculate Predictive Power Score (PPS)
    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
        columns='x', index='y', values='ppscore')

    # Display PPS statistics for values below 1
    pps_score_stats = (pps_matrix_raw.query("ppscore < 1").filter(['ppscore'])
                       .describe().T)
    print(pps_score_stats.round(3))

    return df_corr_pearson, df_corr_spearman, pps_matrix


def DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix,
                      CorrThreshold, PPS_Threshold,
                      figsize=(20, 12), font_annot=8):
    """
    Function to display both the correlation heatmap and the PPS heatmap.

    Args:
        df_corr_pearson: Pearson correlation matrix.
        df_corr_spearman: Spearman correlation matrix.
        pps_matrix: Matrix of PPS values between the features.
        CorrThreshold: Threshold to mask correlations in the heatmap.
        PPS_Threshold: Threshold to mask PPS values in the heatmap.
        figsize: Optional tuple for setting the size of the heatmaps.
        font_annot: Optional font size for annotations in the heatmap.
    """
    # Display Spearman and Pearson correlation heatmaps
    heatmap_corr(df=df_corr_spearman, threshold=CorrThreshold, figsize=figsize,
                 font_annot=font_annot)
    heatmap_corr(df=df_corr_pearson, threshold=CorrThreshold, figsize=figsize,
                 font_annot=font_annot)
    # Display PPS heatmap
    heatmap_pps(df=pps_matrix, threshold=PPS_Threshold, figsize=figsize,
                font_annot=font_annot)
