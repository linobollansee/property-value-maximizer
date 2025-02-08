import streamlit as st

from app_pages.multipage import MultiPage
from app_pages.page_summary import page_summary_body
from app_pages.correlation_analysis import correlation_analysis_body
from app_pages.sales_price_prediction import sales_price_prediction_body
from app_pages.project_hypothesis import project_hypothesis_body
from app_pages.ml_price_prediction import ml_price_prediction_body

# Initialize MultiPage application
app = MultiPage(app_name="Property-Value-Maximizer")

st.markdown(
    """
    <style>
        /* This applies to the whole Streamlit app container */
        .stApp {
            background-image: url("assets/images/background.png");
            background-repeat: repeat;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Add pages to the application
app.add_page("👁️ Project Overview", page_summary_body)
app.add_page("📈 Correlation Analysis", correlation_analysis_body)
app.add_page("🔮 Sale Price Prediction", sales_price_prediction_body)
app.add_page("🔬 Hypothesis Validation", project_hypothesis_body)
app.add_page("🤖 Machine Learning Model", ml_price_prediction_body)

# Run the application
app.run()