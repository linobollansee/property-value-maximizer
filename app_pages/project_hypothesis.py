import streamlit as st


def project_hypothesis_body():
    """
    Displays the hypotheses and their validation results in a Streamlit app.
    This section communicates the project's key assumptions regarding property
    size, quality, and condition in relation to sale price.
    """
    # Title for the section of the Streamlit app
    st.write("## Project Hypotheses and Validation")

    # Display the first hypothesis about the relationship between property size
    # and sale price
    st.success(
        "🔍**First Hypothesis : The Relationship Between Property Size "
        "and Sale Price**\n\n"
        "Our initial hypothesis posits that larger properties tend to "
        "command higher absolute sale prices. This assumption is based "
        "on the general principle that more spacious properties offer "
        "greater utility, which can be an attractive factor for "
        "potential buyers.\n\n"
        "* ✅**Hypothesis Confirmation:** After conducting a thorough "
        "correlation analysis, we observed a positive and moderate "
        "correlation between property size-related features and sale "
        "price. This suggests that, as expected, larger homes tend to "
        "sell for higher prices, reinforcing our initial assumption.\n\n"
    )

    # Display the second hypothesis about the impact of overall quality on sale
    # price
    st.success(
        "🔍**Second Hypothesis: The Impact of Overall Quality on Sale "
        "Price**\n\n"
        "We hypothesize that the overall quality of a house plays a "
        "significant role in determining its sale price. Properties "
        "with higher quality ratings—indicative of superior materials, "
        "craftsmanship, and overall design—are expected to fetch "
        "higher market values due to increased buyer preference and "
        "perceived long-term value.\n\n"
        "* ✅**Hypothesis Confirmation:** Our analysis confirms a strong "
        "correlation between a property's overall quality rating and "
        "its sale price. The data suggests that higher quality homes "
        "consistently achieve higher market valuations, supporting our "
        "hypothesis that construction quality is a key factor in "
        "pricing dynamics.\n\n"
    )

    # Display the third hypothesis about the influence of property condition on
    # market value
    st.success(
        "🔍**Third Hypothesis: The Influence of Property Condition on "
        "Market Value**\n\n"
        "We further hypothesize that a property's overall condition "
        "will significantly impact its sale price. Well-maintained "
        "homes, particularly those with recent renovations or newer "
        "construction dates, are expected to be more desirable to "
        "buyers and, as a result, command higher sale prices.\n\n"
        "* ✅**Hypothesis Confirmation:** The findings of our correlation "
        "analysis support this hypothesis. Specifically, we identified "
        "a positive and moderate correlation between sale price and "
        "factors such as the property's construction year and last "
        "recorded remodel year. These results indicate that newer and "
        "recently updated homes tend to sell at higher prices, "
        "affirming the importance of property condition in market "
        "valuation.\n\n"
    )
