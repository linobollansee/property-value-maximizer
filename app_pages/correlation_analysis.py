import numpy as np
import ppscore as pps
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from feature_engine.discretisation import ArbitraryDiscretiser

from src.data_management import load_ames_data


def correlation_analysis_body():
    df = load_ames_data()
    vars_to_study = ["1stFlrSF", "GarageArea", "GarageYrBlt", "GrLivArea",
                     "KitchenQual_Ex", "KitchenQual_TA", "OverallQual",
                     "TotalBsmtSF", "YearBuilt", "YearRemodAdd"]

    st.write("## Correlation Analysis")
    st.write(
        "The client is interested in analyzing how different house features "
        "impact sale prices. Therefore, they are looking for data "
        "visualizations that highlight the connection between these features "
        "and the sale price."
    )

    if st.checkbox("📊 Display the housing data"):
        st.write(
            f"* The dataset contains {df.shape[0]} rows and has "
            f"{df.shape[1]} columns"
        )
        st.write(df)

    st.write("---")
    st.write(
        "In this analysis, we explored the relationships between various "
        "factors and house sale prices. The primary goal was to identify key "
        "variables that influence pricing trends. After conducting a thorough "
        "correlation analysis, the variables most closely linked to the "
        "sale price are: "
        f"**{vars_to_study}**"
    )

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

    df_eda = df.filter(vars_to_study + ["SalePrice"])
    target_var = "SalePrice"
    st.write("#### Interactive Data visualizations")

    if st.checkbox("⚖️ Distribution of target variable"):
        plot_target_hist(df_eda, target_var)

    if st.checkbox("🔥 Show Correlation and PPS Heatmaps"):
        df_corr_pearson, df_corr_spearman, pps_matrix = (
            CalculateCorrAndPPS(df_eda)
        )
        DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix,
                          CorrThreshold=0.4, PPS_Threshold=0.2)


def plot_target_hist(df, target_var):
    "Interactive histogram plot of the target variable"
    fig = px.histogram(df, x=target_var, nbins=30,
                       title=f"Distribution of {target_var}", marginal="box")
    st.plotly_chart(fig)


def heatmap_corr_interactive(df, threshold, figsize=(20, 12), font_annot=8):
    "Interactive heatmap using correlations."
    if len(df.columns) > 1:
        fig = px.imshow(df, text_auto=True, color_continuous_scale="Viridis",
                        title="Correlation Heatmap")
        st.plotly_chart(fig)


def heatmap_pps_interactive(df, threshold, figsize=(20, 12), font_annot=8):
    "Interactive heatmap with PPS scores."
    if len(df.columns) > 1:
        fig = px.imshow(df, text_auto=True, color_continuous_scale="Jet",
                        title="PPS Heatmap")
        st.plotly_chart(fig)


def CalculateCorrAndPPS(df):
    "Function for calculation of correlations and pps."
    df_corr_spearman = df.corr(method="spearman")
    df_corr_spearman.name = "corr_spearman"
    df_corr_pearson = df.corr(method="pearson")
    df_corr_pearson.name = "corr_pearson"
    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(["x", "y", "ppscore"]).pivot(
        columns="x", index="y", values="ppscore")
    pps_score_stats = (
        pps_matrix_raw.query("ppscore < 1").filter(["ppscore"]).describe().T)
    print(pps_score_stats.round(3))
    return df_corr_pearson, df_corr_spearman, pps_matrix


def DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix,
                      CorrThreshold, PPS_Threshold, figsize=(20, 12),
                      font_annot=8):
    "Display the correlations and PPS in interactive heatmaps."
    heatmap_corr_interactive(df=df_corr_spearman, threshold=CorrThreshold,
                             figsize=figsize, font_annot=font_annot)
    heatmap_corr_interactive(df=df_corr_pearson, threshold=CorrThreshold,
                             figsize=figsize, font_annot=font_annot)
    heatmap_pps_interactive(df=pps_matrix, threshold=PPS_Threshold,
                            figsize=figsize, font_annot=font_annot)
