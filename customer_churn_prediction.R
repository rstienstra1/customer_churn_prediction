# Name: Rachel Stienstra
# Date: 2025-05-19
# Script Title: Customer Churn Prediction
# Description: This script analyzes historical retail transaction data to 
#              predict customer churn using engineered features and a Random 
#              Forest classification model. Churn is defined as a customer not 
#              making a purchase in the last 90 days.
#              The dataset is sourced from: https://archive.ics.uci.edu/dataset/352/online+retail
# Output: 
# Dependencies: [Required packages/libraries]




# Set a seed for reproducibility ------------------------------------------
set.seed(123)




# Load libraries ----------------------------------------------------------
library(ggplot2)       # For data visualization
library(tidyverse)     # For data manipulation (includes dplyr, ggplot2, etc.)
library(readxl)        # For reading Excel files
library(lubridate)     # For handling dates and times
library(randomForest)  # For building the Random Forest model
library(pROC)          # For computing and plotting ROC curves




# Load and clean retail dataset -------------------------------------------

# Read in data and filter out rows with missing CustomerID
retail_data <- read_excel("./data/retail.xlsx") %>%
  filter(!is.na(CustomerID))




# Define function to label churn ------------------------------------------

# Churn = no purchases within the last 90 days from snapshot date
compute_churn_labels <- function(data, churn_days = 90) {
  snapshot_date <- max(data$InvoiceDate)  # Use latest invoice as snapshot
  
  last_purchase <- data %>%
    group_by(CustomerID) %>%
    summarise(LastPurchase = max(InvoiceDate), .groups = "drop")
  
  churned <- last_purchase %>%
    mutate(
      DaysSinceLastPurchase = as.numeric(snapshot_date - LastPurchase, units = "days"),
      Churned = DaysSinceLastPurchase > churn_days
    )
  
  left_join(data, churned, by = "CustomerID")
}

# Apply churn labeling function
data_churn <- compute_churn_labels(retail_data)




# Filter valid  purchases -------------------------------------------------

# Exclude cancelled invoices (those starting with "C") and negative quantities
valid_orders <- retail_data %>%
  filter(!str_starts(InvoiceNo, "C") & Quantity > 0)




# Generate customer-level features ----------------------------------------

customer_features <- valid_orders %>%
  group_by(CustomerID) %>%
  summarise(
    Recency = as.numeric(max(data_churn$InvoiceDate) - max(InvoiceDate), units = "days"),  # Time since last purchase
    Frequency = n_distinct(InvoiceNo),                    # Number of distinct purchases
    Monetary = sum(Quantity * UnitPrice, na.rm = TRUE),   # Total spending
    AvgQuantity = mean(Quantity, na.rm = TRUE),           # Avg quantity per order
    AvgUnitPrice = mean(UnitPrice, na.rm = TRUE),         # Avg price per unit
    UniqueItems = n_distinct(StockCode),                  # Number of different items bought
    .groups = "drop"
  ) %>%
  # Merge with churn labels
  left_join(
    data_churn %>%
      distinct(CustomerID, DaysSinceLastPurchase, Churned),
    by = "CustomerID"
  )




# Add return-related features ---------------------------------------------

returns <- retail_data %>%
  filter(Quantity < 0) %>%  # Returned items have negative quantities
  group_by(CustomerID) %>%
  summarise(
    NumReturns = n(),                    # Number of return transactions
    ReturnedQty = sum(abs(Quantity)),    # Total number of units returned
    .groups = "drop"
  )

# Join return data and fill missing values with 0
customer_features <- customer_features %>%
  left_join(returns, by = "CustomerID") %>%
  replace_na(list(NumReturns = 0, ReturnedQty = 0))




# Split into training and test sets ---------------------------------------

# Randomly split 70% of the data for training
split <- sample(1:nrow(customer_features), 0.7 * nrow(customer_features))
train <- customer_features[split, ]
test <- customer_features[-split, ]

# Convert Churned to factor for classification
train$Churned <- as.factor(train$Churned)
test$Churned <- as.factor(test$Churned)




# Build Random Forest model -----------------------------------------------

rf_model <- randomForest(
  Churned ~ Frequency + Monetary + AvgQuantity + AvgUnitPrice + UniqueItems + NumReturns + ReturnedQty,
  data = train,
  importance = TRUE,       # Enables calculation of feature importance
  ntree = 500,             # Number of trees in the forest
  mtry = 3,                # Number of variables to try at each split
  na.action = na.roughfix  # Impute missing values with median/mode
)

# Print model summary
print(rf_model)




# Evaluate variable importance --------------------------------------------

# Plot importance based on MeanDecreaseGini
varImpPlot(rf_model, type = 2, main = "Random Forest Variable Importance")

# Show raw importance values
importance_values <- importance(rf_model)
print(importance_values)




# Predict churn on test set -----------------------------------------------

rf_pred <- predict(rf_model, newdata = test, type = "response")

# Create and print confusion matrix
conf_matrix <- table(Predicted = rf_pred, Actual = test$Churned)
print(conf_matrix)




# Evaluate model with ROC curve -------------------------------------------

# Predict probabilities for the "TRUE" (churned) class
rf_probs <- predict(rf_model, newdata = test, type = "prob")[, "TRUE"]
# Generate ROC curve object
roc_obj <- roc(test$Churned, rf_probs)
# Compute AUC (Area Under Curve)
auc_value <- auc(roc_obj)
# Plot ROC curve with AUC in title
plot(roc_obj, main = paste("ROC Curve (AUC =", round(auc_value, 3), ")"))
# Print AUC value to console
print(paste("AUC:", round(auc_value, 3)))




