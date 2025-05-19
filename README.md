# Customer Churn Prediction

## Description
This project analyzes retail customer data to predict customer churn using machine learning techniques. The dataset used is the [Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) from the UCI Machine Learning Repository.

The goal is to create a predictive model that identifies which customers are likely to churn based on their purchase behavior and return activity.

## Features Engineered
- Recency: Days since last purchase
- Frequency: Number of distinct purchases
- Monetary: Total spending
- AvgQuantity: Average quantity per purchase
- AvgUnitPrice: Average price per item
- UniqueItems: Number of unique items purchased
- NumReturns: Number of return transactions
- ReturnedQty: Quantity of items returned

## Model
A Random Forest classification model is used to predict churn, with feature importance analysis and evaluation using ROC/AUC metrics.

## Usage
- The main script loads and processes data
- Creates customer-level features
- Trains a Random Forest model
- Evaluates model performance on a test set

## Dashboard (Coming Soon)
A dashboard to visualize churn patterns and model insights will be developed to provide an interactive user experience.

## Installation & Dependencies
This project requires the following R packages:

- ggplot2
- tidyverse
- readxl
- lubridate
- randomForest
- pROC

You can install them using:

```r
install.packages(c("ggplot2", "tidyverse", "readxl", "lubridate", "randomForest", "pROC"))
