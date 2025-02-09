# Property-Value-Maximizer

Welcome to the Property-Value-Maximizer project! This initiative aims to apply **Machine Learning** and **regression algorithms** to accurately predict house prices in Ames, Iowa. Our client has inherited four properties and seeks to maximize their **market value** before selling. By analyzing key housing features and building a powerful predictive model, we strive to provide **data-driven insights** that lead to optimal pricing strategies.

![Responsive-view-multi-device-readme](docs/readme-images/responsive-view-multi-device-readme.png)

The project is accessible at the following URL: <https://property-value-maximizer.onrender.com>

# Table of Contents
- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Agile Methodology](#agile-methodology)
- [Hypotheses and Validation](#hypotheses-and-validation)
- [Mapping Business Requirements to Data Visualizations and ML Tasks](#mapping-business-requirements-to-data-visualizations-and-ml-tasks)

## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

## Business Requirements

Our client has inherited four properties from her late great-grandfather, located in Ames, Iowa, USA. While she has a strong understanding of property prices in her home country, she is concerned that relying on her existing knowledge of the Iowan market may result in inaccurate appraisals. Factors that make a house desirable and valuable in her country may differ from those in Ames, Iowa.

The client has provided a public dataset containing house prices for the Ames area and has requested our assistance in maximizing the sale price for her inherited properties. Our goal is to predict the sale price of these four homes based on their respective attributes.

The business requirements are as follows:

- BR1 - The client wants to understand how various house attributes correlate with the sale price in Ames, Iowa. She expects data visualizations that illustrate the relationships between these variables and the sale price.

- BR2 - The client is looking to predict the sale price for her four inherited houses, as well as for any other property in Ames, Iowa.

To meet these business requirements, Epics and User Stories have been defined. These user stories have been further broken down into manageable tasks, allowing for an agile approach to implementation.

## Agile Methodology

### Epics
- **Data Collection and Information Gathering Epic**
- **Data Visualization, Cleaning, and Preparation Epic**
- **Model Training, Optimization, and Validation Epic**
- **Dashboard Planning, Design, and Development Epic**
- **Dashboard Deployment and Release Epic**

### User Stories
- **Data Collection and Information Gathering Epic**

  - User Story: As a developer, I need to install all required dependencies and packages so that I can effectively utilize the necessary tools for project implementation.
  
    - Business Requirement Addressed: BR 1 & 2

  - User Story: As a developer, I need to import relevant data into a Jupyter Notebook so that I can conduct a thorough analysis of the dataset.
  
    - Business Requirement Addressed: BR 1 & 2

- **Data Visualization, Cleaning, and Preparation Epic**

  - User Story: As a developer, I want to implement a robust data cleaning process so that I can ensure the dataset is accurate, reliable, and of high quality.
  
    - Business Requirement Addressed: BR 1 & 2

- **Model Training, Optimization, and Validation Epic**

  - User Story: As a developer, I want to evaluate the performance of the predictive model so that I can ensure the reliability and accuracy of its predictions.
  
    - Business Requirement Addressed: BR 1 & 2

  - User Story: As a developer, I want to test individual data points against the model’s predictions so that I can determine the target variable based on my provided features.
  
    - Business Requirement Addressed: BR 1 & 2

- **Dashboard Planning, Design, and Development Epic**

  - User Story: As a client, I want to access the Streamlit landing page so that I can quickly gain an overview of the project.
  
    - Business Requirement Addressed: BR 1 & 2

  - User Story: As a client, I want to view data visualizations that illustrate the relationship between the target variable and its key features so that I can gain deeper insights from the data.
  
    - Business Requirement Addressed: BR 1 & 2

  - User Story: As a client, I want to view a correlation analysis page on Streamlit so that I can understand the relationships between various features and the target variable.
  
    - Business Requirement Addressed: BR 1 & 2

  - User Story: As a client, I want to identify the key attributes of a house that have the strongest correlation with its potential sale price so that I can make data-driven pricing decisions. The sale price prediction should be based on the set of features with the highest predictive power.
  
    - Business Requirement Addressed: BR 1 & 2

  - User Story: As a client, I want interactive input fields that allow me to enter custom data so that I can generate personalized predictions for the target variable.
  
    - Business Requirement Addressed: BR 1 & 2

  - User Story: As a client, I want the most accurate possible prediction of the sale prices for the inherited properties so that I can maximize the financial returns from selling the four houses.
  
    - Business Requirement Addressed: BR 1 & 2

  - User Story: As a developer, I need to create a dashboard to effectively visualize and communicate the results of the model's predictions.
  
    - Business Requirement Addressed: BR 1 & 2

- **Dashboard Deployment and Release Epic**

  - User Story: As a developer, I want to initiate the deployment process of my application on Render at an early stage so that I can conduct end-to-end manual deployment testing from the outset.
  
    - Business Requirement Addressed: BR 1 & 2

## Hypotheses and Validation

- First Hypothesis: The Relationship Between Property Size and Sale Price
  - Our first hypothesis posits that the size of a property has a direct and positive influence on its sale price. This assumption is grounded in the widely accepted notion that larger properties tend to offer more space and functionality, which in turn, makes them more attractive to potential buyers. The increased square footage of a property typically allows for additional rooms, larger living areas, and greater customization options, all of which are desirable attributes in a real estate market. Consequently, it is expected that properties with greater size will command higher sale prices due to their enhanced utility and appeal.
    - Hypothesis Confirmation: Following a rigorous correlation analysis of the dataset, we observed a positive and moderate correlation between the size-related features of the properties and their sale prices. This finding validates our hypothesis, as it indicates that larger properties indeed tend to sell for higher prices. The data clearly supports the notion that, all other factors being equal, the size of a property plays a significant role in determining its market value, confirming our initial assumption.

- Second Hypothesis: The Impact of Overall Quality on Sale Price
  - Our second hypothesis focuses on the role of a property's overall quality in influencing its sale price. We hypothesize that properties with higher quality ratings, which reflect superior materials, craftsmanship, and design, will be priced higher in the market. Buyers are likely to place a premium on well-constructed homes that offer longevity, comfort, and aesthetic appeal, which in turn boosts their market value. As such, homes with higher quality ratings should be more desirable and consequently demand higher prices.
    - Hypothesis Confirmation: After analyzing the data, we confirmed that there is a strong correlation between a property's overall quality rating and its sale price. Homes that received higher quality ratings were consistently priced higher in the market, reinforcing the idea that construction quality plays a pivotal role in determining a property’s value. This analysis supports our hypothesis that factors such as the quality of materials, craftsmanship, and overall design are crucial in shaping buyer perceptions and influencing the final sale price.

- Third Hypothesis: The Influence of Property Condition on Market Value
  - For our third hypothesis, we investigate how a property's condition affects its sale price. We hypothesize that homes in excellent condition, particularly those that have undergone recent renovations or are newly built, will be more desirable to buyers and therefore will command higher sale prices. The condition of a property often reflects its upkeep and can signal to buyers the level of maintenance and care invested in the home. Properties in better condition are generally perceived as more move-in ready, which makes them more attractive to prospective buyers looking for immediate comfort without the need for costly repairs or improvements.
    - Hypothesis Confirmation: Our analysis supports this hypothesis by revealing a positive and moderate correlation between sale price and key factors such as the property's construction year and the year of its last remodel. The data suggests that newer homes and those with recent upgrades tend to sell at higher prices, highlighting the importance of property condition in the pricing process. The findings confirm that well-maintained homes or those with modern features are more likely to achieve higher sale prices, underscoring the influence of condition on market value.

## Mapping Business Requirements to Data Visualizations and ML Tasks

- **Business Requirement 1: Data Visualization & Correlation Analysis**
  - Conduct a correlation study using Pearson and Spearman correlation coefficients to assess the relationship between house attributes and the target variable, house price.
  - Evaluate the significance of these correlations.
  - Visualize key variables against house prices to gain insights into their impact.
  - This analysis is documented in the following notebook: <https://github.com/linobollansee/property-value-maximizer/blob/main/jupyter_notebooks/03%20-%20CorrelationStudy.ipynb>

- **Business Requirement 2: Regression Analysis for Price Prediction**
  - Since house price is a continuous variable, a regression analysis is performed to build a predictive model.
  - If regression models do not meet performance expectations, classification-based approaches may be explored.
  - The goal is to predict house prices using key features: `OverallQual`, `GrLivArea`, `GarageArea`, `YearBuilt`, `TotalBsmtSF`
  - This analysis is detailed in the following notebook: <https://github.com/linobollansee/property-value-maximizer/blob/main/jupyter_notebooks/05%20-%20MLModelEvaluation.ipynb>
  
