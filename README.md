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
- [Business Case Understanding](#business-case-understanding)
- [Cross-industry standard process for data mining](#cross-industry-standard-process-for-data-mining)
- [Data Preprocessing](#data-preprocessing)
  - [Data Cleaning Pipeline](#data-cleaning-pipeline)
  - [Feature Engineering](#feature-engineering)
    - [Categorical encoding](#categorical-encoding)
    - [Numerical Transformations](#numerical-transformations)
- [Dashboard Features](#dashboard-features)
- [Bugs and Fixes](#bugs-and-fixes)
- [Project Testing](#project-testing)
- [Deployment](#deployment)
- [Python Packages](#python-packages)
- [Credits](#credits)
  - [Code](#code)
  - [Media](#media)
  - [Content](#content)
- [Acknowledgements](#acknowledgements)

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
  
## Business Case Understanding

- Client’s Business Requirements

  - The client requires a comprehensive analysis to understand the correlation between various house attributes and their corresponding sale prices. They expect visualizations that clearly display the features most strongly correlated with the sale price, allowing for more informed decision-making.

  - In addition to this, the client has inherited four houses and seeks a predictive model to estimate their sale prices. The client also intends to extend the model to predict sale prices for other properties in Ames, Iowa.

- Traditional Data Analysis Feasibility:

  - The client could use traditional analysis methods to approximate the sale prices of the inherited houses by comparing them with similar properties in the dataset. However, this approach is prone to errors due to its subjective nature and reliance on general assumptions. Such estimates are not precise and may lead to inaccuracies in decision-making.

- Dashboard vs. API:

  - The client specifically requires a dashboard. This dashboard will provide an interactive way to visualize data, explore correlations, and predict sale prices in real-time. An API is not necessary for the client’s use case.

- Success Criteria for the Client:

  - A successful outcome for the client is defined by delivering an in-depth analysis that reveals the key attributes most strongly correlated with house sale prices. This will guide the client in maximizing the sale prices of their inherited properties.

- Ethical and Privacy Considerations:

  - The dataset used for this project is publicly available, eliminating any ethical or privacy concerns associated with its use.

- Agile Implementation: EPICS and User Stories:

  - EPICS have been defined for clear guidance, and user stories are tracked through GitHub issues on a Kanban board, which facilitates agile project management. This ensures effective tracking and smooth collaboration throughout the project lifecycle.

  - The project board can be found here: [GitHub Project Board]

- Given that the task involves predicting a continuous numeric outcome, a regression model is the most appropriate for this scenario. This model will leverage the relationships between house features and sale price to make accurate predictions.

- Project Inputs and Expected Outputs:

  - Inputs: The house attributes derived from the publicly available dataset.
  - Outputs: The predicted sale price for each house, expressed in USD as a continuous numeric value.
  - The model will be used to predict the sale price of each of the four inherited houses based on their respective attributes. Additionally, a combined prediction of the total sale price for all four houses will be generated.
  - A user interacting with the dashboard will be able to input the attributes of any house (excluding the inherited ones) through input widgets and receive an estimated sale price instantly.

- Definition of Success:

  - Success will be achieved if the model achieves an R² score of at least 0.75 on both the training and test datasets. This threshold will indicate that the model is sufficiently accurate and reliable.

- Client Benefits:

  - By utilizing this predictive model, the client will be able to optimize the sale prices for their inherited properties. The model will provide accurate, data-driven price predictions, empowering the client to make informed decisions that maximize the sale value of each house.

## Cross-industry standard process for data mining

This project applies the CRISP-DM (CRoss Industry Standard Process for Data Mining) methodology.

|Phase|Explanation|
|---|---|
|**Business Understanding**|This phase focuses on understanding the project objectives and requirements from a business perspective. The goal is to define the problem, set objectives, and determine the data mining goals to achieve business success.|
|**Data Understanding**|In this phase, the focus is on collecting initial data and understanding its quality, content, and structure. It involves exploratory data analysis to uncover insights, patterns, and potential issues.|
|**Data Preparation**|This phase involves cleaning and transforming raw data into a suitable format for modeling. It includes tasks like dealing with missing data, outlier detection, and feature engineering.|
|**Modeling**|In this phase, various data mining techniques (such as classification, regression, clustering, etc.) are applied to the prepared data to create models. It is often an iterative process where models are trained, tested, and refined.|
|**Evaluation**|After the model has been built, this phase evaluates its performance based on predefined criteria. The model is assessed to ensure it meets business goals and objectives before it is deployed.|
|**Deployment**|The final phase focuses on implementing the data mining solution into the business environment. This includes integrating the model into production systems, delivering results, and monitoring its impact on business processes.|

## Data Preprocessing

### Data Cleaning Pipeline

A data cleaning pipeline was developed to handle missing values. Various imputation methods were applied based on the statistical properties of the variables.

- Mean Imputation for Normally Distributed Continuous Variables
  - For continuous features such as `LotFrontage` and `BedroomAbvGr`, missing values were imputed using the mean. This approach is suitable for variables that follow an approximately normal distribution without significant outliers, as it maintains the overall data distribution without skewing the central tendency.

- Median Imputation for Skewed Continuous Variables
  - Variables exhibiting right-skewed distributions, such as `2ndFlrSF` and `MasVnrArea`, were imputed using the median. Since the median is less sensitive to extreme values, it provides a more robust imputation strategy for skewed data, preventing artificial distortion of the dataset.

- Categorical Variable Imputation with 'None'
  - Categorical features like `GarageFinish`, `BsmtFinType1`, and `BsmtExposure` were missing primarily because these attributes did not apply to certain properties (e.g., a house without a basement). To preserve this structural information, missing values were imputed with "None" rather than the mode, ensuring that the absence of a feature is explicitly represented rather than inferred as a common category.

- Feature Removal Due to High Missingness
  - Features such as `EnclosedPorch`, `GarageYrBlt`, and `WoodDeckSF` contained a substantial proportion of missing values. Rather than imputing them with limited available observations which could introduce bias, these features were removed from the dataset. Their exclusion was justified based on their potential lack of predictive power and the risk of introducing noise into the model.

For imputation rationale, refer to the detailed analysis in the following notebook: <https://github.com/linobollansee/property-value-maximizer/blob/main/jupyter_notebooks/02%20-%20DataCleaning.ipynb>

### Feature Engineering

#### Categorical encoding
Categorical encoding was applied to convert ordinal categories into numerical values, preserving both the order and hierarchy of the categories. This allowed the regression analysis to account for their relative rankings. However, during the data cleaning process, most ordinal categories were removed.  

#### Numerical Transformations

|**Feature**|**Assessment**|**Applied Transformation**|
|----|----|----|
|TotalBsmtSF| Mean imputation proved to be the most effective method for handling missing values.|MeanMedianImputer|
|GrLivArea| A logarithmic transformation was the best approach to achieve normalization.|LogTransformer|
|TotalBsmtSF| Power transformation yielded the most effective normalization.|PowerTransformer|
|TotalBsmtSF, GarageArea| Outliers were best handled using Winsorization with the IQR method.|Winsorizer|
|TotalBsmtSF, GrLivArea, GarageArea|Standard scaling provided the most effective way to normalize feature ranges.|StandardScaler|

## Dashboard Features

## Bugs and Fixes

ModuleNotFoundError: No module named 'pkg_resources'

![pkg-resources-bug](docs/readme-images/pkg-resources-bug-readme.png)

This bug was fixed by adding setuptools==75.8.0 to requirements.txt

## Project Testing

## Deployment

1. Log in to Render.com using Github.
2. Click on the New button, select Web Service.
3. At Source Code, select Git Providor. Select your repository name. Click Connect.
4. Enter a unique name for your web service.
5. Select the Python3 language.
6. Select the main branch.
7. Select the Frankfurt (EU Central) Region.
8. Set the Build Command: `pip install -r requirements.txt && ./setup.sh`
9. Set the Start Command: `streamlit run app.py`
10. Set Instance Type: Free
11. Set the Environment Variables: `Key: PORT` `Value: 8501` and `Key: PYTHON_VERSION` `Value: 3.12.1`
12. Click Deploy Web Service

## Python Packages

- Data Processing & Feature Engineering
  - feature-engine: A library for feature engineering in machine learning pipelines, offering transformations like encoding, imputation, and scaling.
  - pandas: A fundamental library for data manipulation and analysis using DataFrames and Series.
  - numpy: Provides support for numerical operations, arrays, and mathematical functions.

- Data Visualization
  - matplotlib: A widely used library for static, animated, and interactive visualizations.
  - seaborn: Built on top of Matplotlib, it simplifies statistical data visualization.
  - plotly: Enables interactive plots, dashboards, and web-based visualizations.

- Machine Learning & Model Evaluation
  - scikit-learn: A popular ML library offering tools for classification, regression, clustering, and preprocessing.
  - xgboost: An optimized gradient boosting framework widely used for structured data ML tasks.

- Data Profiling & Exploratory Analysis
  - ppscore: Calculates predictive power scores to determine relationships between variables.
  - ydata-profiling: Generates detailed EDA reports, summarizing data characteristics, correlations, and missing values.

- Web Applications & Image Processing
  - streamlit: A framework for building interactive ML and data science web apps with minimal code.

- Others
  - setuptools: A package development and distribution tool, ensuring dependencies are managed properly.

## Credits

### Code

A significantly large portion of the code used in this project was directly sourced from the Code Institute. This includes:

- Setup and Data Collection
  - Code to change working directory.
  - Code to create directories.
  - Code to download data from Kaggle.
  - Code to extract zip files.
  - Code to import CSV files.
- Exploratory Data Analysis (EDA) and Data Cleaning
  - Code to display DataFrame (df) summaries.
  - Code to count null values.
  - Code to count duplicates.
  - Code to drop variables from a DataFrame (df).
  - Code to subset columns or rows.
  - Code to generate an EDA report.
  - Code to visualize data cleaning effect.
  - Code to plot numerical and categorical variables.
  - Code to generate a heatmap.
  - Code to generate a histogram.
- Data Preprocessing
  - Code to apply mean imputation.
  - Code to apply median imputation.
  - Code to apply categorical imputation.
  - Code to OneHotEncode.
  - Code to apply ordinal encoding on categorical variables.
  - Code to apply a winsoriser transformation.
  - Code to apply a power transformation.
  - Code to apply a log transformation.
  - Code to apply feature scaling using standardization.
  - Code to check for feature engineering for numerical and categorical variables.
  - Code to identify highly correlated features.
  - Code to calculate correlation coefficients.
- Data Splitting and Feature Selection
  - Code to split train and test set.
  - Code to identify the most important features by the best regression model.
  - Code to extract the best regressor from search.
  - Code to extract the best hyperparameter.
  - Code to check the best model.
- Modeling and Hyperparameter Tuning
  - Code to perform hyperparameter optimization.
  - Code to summarizing the results of the grid searches.
  - Code to fit a machine learning pipeline.
- Model Evaluation and Saving
  - Code to evaluate regression performance on train set and test set.
  - Code to save a machine learning model to a pickle file.

### Media

The Unicode icons used in this project were generated with the assistance of ChatGPT, an AI language model developed by OpenAI. These icons were selected and formatted based on UX to enhance clarity and visual communication.

### Content

## Acknowledgements

I would like to acknowledge my mentor, Mo Shami, for his support throughout the project. His suggestion to explore the repositories of students doing the same project and run these repositories locally with the `streamlit run app.py` terminal command when the Render or Heroku deployments were unavailable was especially helpful.