import streamlit as st

from app_pages.multipage import MultiPage
from app_pages.summary_page import summary_page_body
from app_pages.correlation_analysis_page import correlation_analysis_page
from app_pages.sales_price_prediction_page import sales_price_prediction_page
from app_pages.project_hypothesis_page import project_hypothesis_page_body
from app_pages.ml_price_prediction import ml_price_prediction_page

# Set Streamlit page configuration
st.set_page_config(page_title="Property-Value-Maximizer", page_icon="💰")

# Initialize MultiPage application
app = MultiPage(app_name="Property-Value-Maximizer")

# Add pages to the application
app.add_page("👁️ Project Overview", summary_page_body)
app.add_page("📈 Correlation Analysis", correlation_analysis_page)
app.add_page("🔮 Sale Price Prediction", sales_price_prediction_page)
app.add_page("🔬 Hypothesis Validation", project_hypothesis_page_body)
app.add_page("🤖 Machine Learning Model", ml_price_prediction_page)

# Run the application
app.run()