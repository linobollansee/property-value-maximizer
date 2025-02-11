import streamlit as st


def page_summary_body():
    """
    Displays the contents of the project summary page in the Streamlit app.
    This includes an overview of the project, dataset details, business
    requirements, and additional resources like the README file.
    """
    # Displaying the project overview section
    st.write("## Project Overview")

    # Display a general project summary with key information
    st.info(
        "📌 **Project Summary**\n\n"
        "After inheriting four houses in Ames, Iowa, our client enlisted our "
        "expertise to help secure the highest possible 💲 sale prices. "
        "To accomplish this, we designed a 🤖 Machine Learning model paired "
        "with 📈 regression algorithms to provide precise pricing insights and "
        "maximize the 🏠 property's market value."
    )

    # Displaying the dataset section
    st.write("### Project Dataset")

    # Information about the dataset source and key attributes
    st.info(
        "🌐 **Data Source**\n\n"
        "The dataset, originating from 📂 Kaggle, contains 1460 house price "
        "records detailing housing sales in Ames, Iowa. "
        "Each entry features 24 attributes that describe various aspects of "
        "the 🏠 homes, including 📏 floor area, 🏗️ basement details, and the "
        "presence of a 🚗 garage."
    )

    # Displaying the business requirements section
    st.write("### Business Requirements")

    # Information about the project's business goals and requirements
    st.success(
        "🎯 **Project Aims**\n\n"
        "This project is driven by two business requirements:\n"
        "* **Business Requirement 1**: 📊 Explore and analyze how various "
        "house characteristics impact the sale price, with the help of "
        "visualizations to illustrate these connections.\n"
        "* **Business Requirement 2**: 📈 Build a forecasting model to predict "
        "the sale prices of the inherited properties and other homes in Ames, "
        "Iowa."
    )

    # Additional information and external links related to the project
    st.write("### Additional Information")

    # Provide a link to the README file for further details
    st.write(
        "📄 To learn more about this project, visit the "
        "[README file](https://github.com/linobollansee/"
        "property-value-maximizer/blob/main/README.md)."
    )
