#Diabetes Prediction using Linear Regression, kNN and CART
#Data Ingestion and Pre-Processing

rm(list=ls()); gc()
library(rpart); library(rpart.plot)
setwd("/Users/migoc/DATA/JUPYTER WORK/01 DATA PROJECTS/Diabetes Prediction")


dat = read.csv('diabetes_dataset.csv', stringsAsFactors=T, head=T)

#Explore Dataset
str(dat)

hist(dat$Diabetes_binary, main = 'Frequency of Category Distribution')
hist(dat$BMI, main = 'BMI distribution')
hist(dat$GenHlth, main = ' Frequency of General Health Score')
hist(dat$Age, main = 'Frequency of Age Distribution')
hist(dat$Education)

#Dimension of the dataset
dim(dat)

#Summary of the dataset
summary(dat)

#Columns
colnames(dat)

#Rename the Column Diabetes_binary to Diabetes_Type
colnames(dat)[1] <- "Diabetes_Type"

#Cleaning Dataset

#Missing Value Count
sum(is.na(dat))

#Unique Values

#BMI
sort(unique(dat$BMI))

#General Health
sort(unique(dat$GenHlth))
# 1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor

#Mental Health Scale
sort(unique(dat$MentHlth))

#Physical Health
sort(unique(dat$PhysHlth))

#Age
sort(unique(dat$Age))

#Education
sort(unique(dat$Education))

#Income
sort(unique(dat$Income))

#Diabetes Type
sort(unique(dat$Diabetes_Type))


library(dplyr)

#Replacing the levels of General Health
#Before : 1= Excellent, 2 = Very Good, 3=Good, 4=Fair, 5=Poor
#After : 5= Excellent, 4= Very Good, 3 = Good, 2= Fair, 1=Poor
dat <- dat %>% mutate(GenHlth = recode(GenHlth, `1` = 5, `2` = 4, `3` = 3, `4` = 2, `5` = 1))

diabetes_type_counts <- table(dat$Diabetes_Type)

#Changing the values of '2.0' to '1.0' to make it Binary Logistic

dat$Diabetes_Type[dat$Diabetes_Type== 2.0] <- 1.0

counts_before <- table(dat$Diabetes_Type)

#We see a class imbalance of the target variable
#We try to both Oversample minority class(1's) using SMOTE
#Also we do Undersampling of the majority class(0's)

#First we apply SMOTE (Oversampling technique)

library(smotefamily)
library(dplyr)
library(tidyverse)
sum(is.na(dat))
smote <- SMOTE(dat, dat$Diabetes_Type)
oversampled_dataset <- smote$data[,-23]
counts_after_oversampling <- table(oversampled_dataset$Diabetes_Type)

#Exploring dataset after oversampling
sum(is.na(oversampled_dataset))

str(oversampled_dataset)

#We floor the values of variables because after SMOTE, so the synthetic values generated are consistent
oversampled_dataset <- floor(oversampled_dataset)

#Checking the correlation
library(gplots)
correlation_matrix_oversampled <- cor(oversampled_dataset)
heatmap.2(correlation_matrix_oversampled,
          col = colorRampPalette(c("blue", "white", "red"))(20),
          main = "Correlation Heatmap"
)


#Replacing the floating point variables of Diabetes_Type to Integers
oversampled_dataset$Diabetes_Type <- as.integer(oversampled_dataset$Diabetes_Type)


######Undersampling########
# Identify the indices of the majority class (0's)
indices_majority <- which(dat$Diabetes_Type == 0)

# Sample a subset of indices from the majority class
indices_majority_undersampled <- sample(indices_majority, length(which(dat$Diabetes_Type == 1)))

# Combine indices of the majority and minority classes
indices_undersampled <- c(indices_majority_undersampled, which(dat$Diabetes_Type == 1))

# Create the undersampled dataset
undersampled_dataset <- dat[indices_undersampled, ]


counts_after_undersampling <- table(undersampled_dataset$Diabetes_Type)

install.packages(c("gplots", "RColorBrewer"))
library(gplots)
library(RColorBrewer)
correlation_matrix_undersampled <- cor(undersampled_dataset)
heatmap.2(correlation_matrix_undersampled,
          col = colorRampPalette(c("blue", "white", "red"))(20),
          main = "Correlation Heatmap"
)


#Replacing the floating point variables of Diabetes_Type to Integers
undersampled_dataset$Diabetes_Type <- as.integer(undersampled_dataset$Diabetes_Type)


#Comparing the Target Variable Counts
counts_before
counts_after_undersampling
counts_after_oversampling

##Variable names to be used for further##
dataset <- dat #Original Dataset
undersampled_dataset #undersampled Dataset
oversampled_dataset #SMOTE oversampled Dataset

#########################################################################

###############################LOGISTIC REGRESSION#######################

#########################################################################

# Logistic Regression with Forward Selection
forward_model <- glm(Diabetes_Type ~ ., data = dataset, family = binomial)
forward_model <- step(forward_model, direction = "forward")

# Displaying Regression Table for Forward Selection
summary(forward_model)

# Logistic Regression with Backward Elimination
backward_model <- glm(Diabetes_Type ~ ., data = dataset, family = binomial)
backward_model <- step(backward_model, direction = "backward")

# Displaying Regression Table for Backward Elimination
summary(backward_model)

# Logistic Regression with Stepwise Selection
stepwise_model <- glm(Diabetes_Type ~ ., data = dataset, family = binomial)
stepwise_model <- step(stepwise_model, direction = "both")

# Displaying Regression Table for Stepwise Selection
summary(stepwise_model)

########################################PRINT REGRESSION TABLE FOR EACH METHOD#######################
# Load the broom package
# Install broom if not already installed
if (!requireNamespace("broom", quietly = TRUE)) {
  install.packages("broom")
}

# Load the installed packages
library(broom)


# Logistic Regression with Forward Selection
forward_model <- glm(Diabetes_Type ~ ., data = dataset, family = binomial)
forward_model <- step(forward_model, direction = "forward")

# Extract tidy information
tidy_forward <- tidy(forward_model)

# Print the regression table for Forward Selection
cat("\nRegression Table for Forward Selection:\n")
print(tidy_forward)

# Logistic Regression with Backward Elimination
backward_model <- glm(Diabetes_Type ~ ., data = dataset, family = binomial)
backward_model <- step(backward_model, direction = "backward")

# Extract tidy information
tidy_backward <- tidy(backward_model)

# Print the regression table for Backward Elimination
cat("\nRegression Table for Backward Elimination:\n")
print(tidy_backward)

# Logistic Regression with Stepwise Selection
stepwise_model <- glm(Diabetes_Type ~ ., data = dataset, family = binomial)
stepwise_model <- step(stepwise_model, direction = "both")

# Extract tidy information
tidy_stepwise <- tidy(stepwise_model)

# Print the regression table for Stepwise Selection
cat("\nRegression Table for Stepwise Selection:\n")
print(tidy_stepwise)


############################## IDENTIFY VARIABLES THAT ARE IMPORTANT AND NOT IMPORTANT#################################
# Load the dplyr package if not already loaded
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}

# Load the dplyr package
library(dplyr)

# Function to filter and arrange variables by p-value
filter_and_arrange <- function(model) {
  tidy_result <- tidy(model)
  significant_vars <- filter(tidy_result, p.value < 0.05) %>% arrange(p.value)
  non_significant_vars <- filter(tidy_result, p.value >= 0.05)
  return(list(significant_vars, non_significant_vars))
}

# Apply the function for each model
result_forward <- filter_and_arrange(forward_model)
result_backward <- filter_and_arrange(backward_model)
result_stepwise <- filter_and_arrange(stepwise_model)

# Extract significant and non-significant variables
significant_forward <- result_forward[[1]]
non_significant_forward <- result_forward[[2]]

significant_backward <- result_backward[[1]]
non_significant_backward <- result_backward[[2]]

significant_stepwise <- result_stepwise[[1]]
non_significant_stepwise <- result_stepwise[[2]]
options(dplyr.print_max = 20)  # Adjust the number as needed

# Print the results
cat("\nSignificant Variables for Forward Selection (p-value < 0.05), sorted by p-value:\n")
print(significant_forward)

cat("\nNon-Significant Variables for Forward Selection (p-value >= 0.05):\n")
print(non_significant_forward)

cat("\nSignificant Variables for Backward Elimination (p-value < 0.05), sorted by p-value:\n")
print(significant_backward)

cat("\nNon-Significant Variables for Backward Elimination (p-value >= 0.05):\n")
print(non_significant_backward)

cat("\nSignificant Variables for Stepwise Selection (p-value < 0.05), sorted by p-value:\n")
print(significant_stepwise)

cat("\nNon-Significant Variables for Stepwise Selection (p-value >= 0.05):\n")
print(non_significant_stepwise)




######################Correlation############################
# Extracting coefficients from Forward Selection model
coefficients_forward <- coef(forward_model)

# Displaying coefficients from Forward Selection
cat("Coefficients from Forward Selection model:\n")
print(coefficients_forward)

# Identifying variables with positive coefficients (indicating positive influence on Diabetes_Type)
positive_vars_forward <- names(coefficients_forward[coefficients_forward > 0])

# Display variables with positive coefficients from Forward Selection
cat("Variables with positive coefficients in Forward Selection model:\n")
print(positive_vars_forward)

# Extracting coefficients from Backward Elimination model
coefficients_backward <- coef(backward_model)

# Displaying coefficients from Backward Elimination
cat("Coefficients from Backward Elimination model:\n")
print(coefficients_backward)

# Identifying variables with positive coefficients (indicating positive influence on Diabetes_Type)
positive_vars_backward <- names(coefficients_backward[coefficients_backward > 0])

# Display variables with positive coefficients from Backward Elimination
cat("Variables with positive coefficients in Backward Elimination model:\n")
print(positive_vars_backward)

# Extracting coefficients from Stepwise Selection model
coefficients_stepwise <- coef(stepwise_model)

# Displaying coefficients from Stepwise Selection
cat("Coefficients from Stepwise Selection model:\n")
print(coefficients_stepwise)

# Identifying variables with positive coefficients (indicating positive influence on Diabetes_Type)
positive_vars_stepwise <- names(coefficients_stepwise[coefficients_stepwise > 0])

# Display variables with positive coefficients from Stepwise Selection
cat("Variables with positive coefficients in Stepwise Selection model:\n")
print(positive_vars_stepwise)

#########################BEST VARIABLE SELECTION METHOD#######################

# Install caTools if not already installed
if (!requireNamespace("caTools", quietly = TRUE)) {
  install.packages("caTools")
}

# Load the installed package
library(caTools)

# Split the dataset into training and testing sets
split <- sample.split(dataset$Diabetes_Type, SplitRatio = 0.7)
dat.train <- subset(dataset, split == TRUE)
dat.test <- subset(dataset, split == FALSE)

# Logistic Regression with Forward Selection
forward_model <- glm(Diabetes_Type ~ ., data = dat.train, family = binomial)
forward_model <- step(forward_model, direction = "forward")

# Logistic Regression with Backward Elimination
backward_model <- glm(Diabetes_Type ~ ., data = dat.train, family = binomial)
backward_model <- step(backward_model, direction = "backward")

# Logistic Regression with Stepwise Selection
stepwise_model <- glm(Diabetes_Type ~ ., data = dat.train, family = binomial)
stepwise_model <- step(stepwise_model, direction = "both")

# Function to calculate accuracy
calculate_accuracy <- function(predictions, actual) {
  confusion_matrix <- table(Actual = actual, Predicted = predictions)
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  return(accuracy)
}

# Make predictions on the test set for each model
predictions_forward <- predict(forward_model, newdata = dat.test, type = "response")
predictions_backward <- predict(backward_model, newdata = dat.test, type = "response")
predictions_stepwise <- predict(stepwise_model, newdata = dat.test, type = "response")

# Convert probabilities to binary predictions (0 or 1) using a cutoff of 0.5
predicted_classes_forward <- ifelse(predictions_forward > 0.5, 1, 0)
predicted_classes_backward <- ifelse(predictions_backward > 0.5, 1, 0)
predicted_classes_stepwise <- ifelse(predictions_stepwise > 0.5, 1, 0)

# Calculate accuracy for each model
accuracy_forward <- calculate_accuracy(predicted_classes_forward, dat.test$Diabetes_Type)
accuracy_backward <- calculate_accuracy(predicted_classes_backward, dat.test$Diabetes_Type)
accuracy_stepwise <- calculate_accuracy(predicted_classes_stepwise, dat.test$Diabetes_Type)

# Compare accuracies and select the best method
best_method <- which.max(c(accuracy_forward, accuracy_backward, accuracy_stepwise))

# Display results
cat("Accuracy for Forward Selection:", round(accuracy_forward, 2), "\n")
cat("Accuracy for Backward Elimination:", round(accuracy_backward, 2), "\n")
cat("Accuracy for Stepwise Selection:", round(accuracy_stepwise, 2), "\n")

cat("The best variable selection method is:")
if (best_method == 1) {
  cat("Forward Selection\n")
} else if (best_method == 2) {
  cat("Backward Elimination\n")
} else {
  cat("Stepwise Selection\n")
}



###############ACCURACY,SENSITIVITY,SPECIFICITY
# Function to calculate Accuracy, Sensitivity, and Specificity from a confusion matrix
calculate_metrics <- function(conf_matrix) {
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
  specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
  return(c(accuracy, sensitivity, specificity))
}

# Define cutoff values
cutoff_values <- c(0.5, 0.4, 0.3, 0.2, 0.1)

# Function to calculate metrics for a given cutoff and model
calculate_metrics_with_cutoff <- function(model, dat_test, cutoff) {
  # Make predictions on the test set
  predictions <- predict(model, newdata = dat_test, type = "response")
  
  # Convert probabilities to binary predictions using the cutoff
  predicted_classes <- ifelse(predictions > cutoff, 1, 0)
  
  # Create a confusion matrix
  conf_matrix <- table(Actual = dat_test$Diabetes_Type, Predicted = predicted_classes)
  
  # Calculate metrics
  metrics <- calculate_metrics(conf_matrix)
  
  return(metrics)
}

# Function to summarize results for a given model and cutoff
summarize_results <- function(model, dat_test, cutoff, method) {
  # Calculate metrics for the given model and cutoff
  metrics <- calculate_metrics_with_cutoff(model, dat_test, cutoff)
  
  # Return a named list of results
  results <- c(
    Cutoff = cutoff,
    Method = method,
    Val_Accuracy = round(metrics[1], 2),
    Sensitivity = round(metrics[2], 2),
    Specificity = round(metrics[3], 2)
  )
  
  return(results)
}

# Create an empty list to store the results
results_list <- list()

# Loop over cutoff values and summarize results for each model
for (cutoff in cutoff_values) {
  # Forward Selection
  results_forward <- summarize_results(forward_model, dat.test, cutoff, "Forward")
  results_list <- c(results_list, results_forward)
  
  # Backward Elimination
  results_backward <- summarize_results(backward_model, dat.test, cutoff, "Backward")
  results_list <- c(results_list, results_backward)
  
  # Stepwise Selection
  results_stepwise <- summarize_results(stepwise_model, dat.test, cutoff, "Stepwise")
  results_list <- c(results_list, results_stepwise)
}

# Convert the list of results to a data frame
results_df <- as.data.frame(matrix(unlist(results_list), ncol = 5, byrow = TRUE))
colnames(results_df) <- c("Cutoff", "Method", "Val_Accuracy", "Sensitivity", "Specificity")

# Display the results in tabular format
print(results_df)


##########AOC CURVE##########################################
# Install pROC if not already installed
if (!requireNamespace("pROC", quietly = TRUE)) {
  install.packages("pROC")
}

# Load the installed package
library(pROC)


# Function to calculate AUC for a given model and cutoff
calculate_auc <- function(model, dat_test, method) {
  # Make predictions on the test set
  predictions <- predict(model, newdata = dat_test, type = "response")
  
  # Create a ROC curve
  roc_curve <- roc(dat_test$Diabetes_Type, predictions)
  
  # Calculate AUC
  auc_value <- auc(roc_curve)
  
  cat("AUC for", method, ":", round(auc_value, 2), "\n")
  
  # Return AUC value
  return(auc_value)
}

# Calculate and display AUC for each model
auc_forward <- calculate_auc(forward_model, dat.test, "Forward Selection")
auc_backward <- calculate_auc(backward_model, dat.test, "Backward Elimination")
auc_stepwise <- calculate_auc(stepwise_model, dat.test, "Stepwise Selection")

# Find the best AUC
auc_values <- c(auc_forward, auc_backward, auc_stepwise)
best_method <- names(auc_values)[which.max(auc_values)]

cat("Best Method:", best_method, "with AUC =", round(max(auc_values), 2), "\n")


#######################ODDS RATIO on FORWARD MODEL ################################


# Extract coefficients from the model
coefficients_logreg <- coef(backward_model)

# Calculate odds ratios
odds_ratios <- exp(coefficients_logreg)

# Display odds ratios with variable names
cat("Odds Ratios from Logistic Regression model:\n")
print(data.frame(Variable = names(odds_ratios), OddsRatio = odds_ratios))

# Interpretation
cat("\nInterpretation:\n")
cat("For a one-unit increase in the predictor variable:\n")

# Loop through each variable
for (i in seq_along(odds_ratios)) {
  cat("   - A one-unit increase in", names(odds_ratios)[i], "is associated with a",
      round(odds_ratios[i], 2), "times increase in the odds of the event.\n")
}




#########################################################################

###############################CLASSIFICATION TREE#######################

#########################################################################

#*UnderSampled*#
library(caret)
library(rpart)
library(rpart.plot)
#Splitting data
set.seed(151)  # for reproducibility
split_index <- createDataPartition(undersampled_dataset$Diabetes_Type, p = 0.8, list = FALSE)
train_data_cart <- undersampled_dataset[split_index, ]
test_data_cart <- undersampled_dataset[-split_index, ]

# Classification Tree with rpart
control <- rpart.control(minsplit = 5, minbucket = 5, maxdepth =30)
fit = rpart(Diabetes_Type ~ ., method="class", data=train_data_cart, control = control) # same as using all other variables as predictors
K <- length(train_data_cart)
# Minimum Error Tree
pfit.me = prune(fit, cp = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
rpart.plot(pfit.me, main = 'Min Error Tree')


# Best Pruned Tree
ind = which.min(fit$cptable[,"xerror"]) # xerror: cross-validation error
se1 = fit$cptable[ind,"xstd"]/sqrt(K) # 1 standard error
xer1 = min(fit$cptable[,"xerror"]) + se1 # targeted error: min + 1 SE
ind0 = which.min(abs(fit$cptable[1:ind,"xerror"] - xer1)) # select the tree giving closest xerror to xer1
pfit.bp = prune(fit, cp = fit$cptable[ind0,"CP"])
rpart.plot(pfit.bp, main = 'Best Pruned Tree')

## Prediction
# Using the default threshold of 0.5
yhat = predict(pfit.me, test_data_cart, type = "class")

# Check the lengths of yhat and undersampled_dataset$Diabetes_Type
if (length(yhat) != length(undersampled_dataset$Diabetes_Type)) {
  stop("Lengths of yhat and undersampled_dataset$Diabetes_Type do not match.")
}

# Check for missing values
if (any(is.na(yhat)) || any(is.na(undersampled_dataset$Diabetes_Type))) {
  stop("There are missing values in yhat or undersampled_dataset$Diabetes_Type.")
}

# Create confusion matrix
conf_matrix = table(yhat, test_data_cart$Diabetes_Type)

# Display confusion matrix
print(conf_matrix)



# Calculate specificity and sensitivity
TN = conf_matrix[1, 1]  # True Negatives
FP = conf_matrix[1, 2]  # False Positives
FN = conf_matrix[2, 1]  # False Negatives
TP = conf_matrix[2, 2]  # True Positives

specificity = TN / (TN + FP)
sensitivity = TP / (TP + FN)

accuracy = (TP + TN) / sum(conf_matrix)

# Output specificity and sensitivity
cat("Specificity:", specificity, "\n")
cat("Sensitivity:", sensitivity, "\n")

accuracy

# If you want to use a different cutoff (0.5 in this case)
prob1 = predict(pfit.me, test_data_cart, type = "prob")[, 2]
pred.class = as.numeric(prob1 > 0.1)
ytest = as.numeric(test_data_cart$Diabetes_Type) - 1
err.me.newCut = mean(pred.class != ytest)

# Calculate specificity and sensitivity with the new cutoff
conf_matrix_newCut = table(pred.class, ytest)
TN_newCut = conf_matrix_newCut[1, 1]
FP_newCut = conf_matrix_newCut[1, 2]
FN_newCut = conf_matrix_newCut[2, 1]
TP_newCut = conf_matrix_newCut[2, 2]

specificity_newCut = TN_newCut / (TN_newCut + FP_newCut)
sensitivity_newCut = TP_newCut / (TP_newCut + FN_newCut)

accuracy_newCut = (TP + TN) / sum(conf_matrix)

# Output specificity and sensitivity with the new cutoff
cat("Specificity with new cutoff:", specificity_newCut, "\n")
cat("Sensitivity with new cutoff:", sensitivity_newCut, "\n")
accuracy_newCut
#*****************************************************************#
#*Oversampled*#

#Splitting data
set.seed(161)  # for reproducibility
split_index <- createDataPartition(oversampled_dataset$Diabetes_Type, p = 0.8, list = FALSE)
train_data_cart <- oversampled_dataset[split_index, ]
test_data_cart <- oversampled_dataset[-split_index, ]

# Classification Tree with rpart
control <- rpart.control(minsplit = 5, minbucket = 5, maxdepth =30)
fit = rpart(Diabetes_Type ~ ., method="class", data=train_data_cart, control = control) # same as using all other variables as predictors
# Minimum Error Tree
pfit.me = prune(fit, cp = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
rpart.plot(pfit.me, main = 'Min Error Tree')

# Best Pruned Tree
ind = which.min(fit$cptable[,"xerror"]) # xerror: cross-validation error
se1 = fit$cptable[ind,"xstd"]/sqrt(K) # 1 standard error
xer1 = min(fit$cptable[,"xerror"]) + se1 # targeted error: min + 1 SE
ind0 = which.min(abs(fit$cptable[1:ind,"xerror"] - xer1)) # select the tree giving closest xerror to xer1
pfit.bp = prune(fit, cp = fit$cptable[ind0,"CP"])
rpart.plot(pfit.bp, main = 'Best Pruned Tree')

## Prediction
# Using the default threshold of 0.5
yhat = predict(pfit.bp, test_data_cart, type = "class")

# Check the lengths of yhat and undersampled_dataset$Diabetes_Type
if (length(yhat) != length(undersampled_dataset$Diabetes_Type)) {
  stop("Lengths of yhat and undersampled_dataset$Diabetes_Type do not match.")
}

# Check for missing values
if (any(is.na(yhat)) || any(is.na(undersampled_dataset$Diabetes_Type))) {
  stop("There are missing values in yhat or undersampled_dataset$Diabetes_Type.")
}

# Create confusion matrix
conf_matrix = table(yhat, test_data_cart$Diabetes_Type)

# Display confusion matrix
print(conf_matrix)



# Calculate specificity and sensitivity
TN = conf_matrix[1, 1]  # True Negatives
FP = conf_matrix[1, 2]  # False Positives
FN = conf_matrix[2, 1]  # False Negatives
TP = conf_matrix[2, 2]  # True Positives

specificity = TN / (TN + FP)
sensitivity = TP / (TP + FN)

accuracy = (TP + TN) / sum(conf_matrix)

# Output specificity and sensitivity
cat("Specificity:", specificity, "\n")
cat("Sensitivity:", sensitivity, "\n")

accuracy

# If you want to use a different cutoff (0.5 in this case)
prob1 = predict(pfit.bp, test_data_cart, type = "prob")[, 2]
pred.class = as.numeric(prob1 > 0.1)
ytest = as.numeric(test_data_cart$Diabetes_Type) - 1
err.bp.newCut = mean(pred.class != ytest)

# Calculate specificity and sensitivity with the new cutoff
conf_matrix_newCut = table(pred.class, ytest)
TN_newCut = conf_matrix_newCut[1, 1]
FP_newCut = conf_matrix_newCut[1, 2]
FN_newCut = conf_matrix_newCut[2, 1]
TP_newCut = conf_matrix_newCut[2, 2]

specificity_newCut = TN_newCut / (TN_newCut + FP_newCut)
sensitivity_newCut = TP_newCut / (TP_newCut + FN_newCut)

accuracy_newCut = (TP + TN) / sum(conf_matrix)

# Output specificity and sensitivity with the new cutoff
cat("Specificity with new cutoff:", specificity_newCut, "\n")
cat("Sensitivity with new cutoff:", sensitivity_newCut, "\n")
accuracy_newCut


#########################################################################

###############################RANDOM FOREST#######################

#########################################################################


##*Undersampled*##
# Install and load the randomForest package
install.packages("randomForest")
install.packages("caret")
library(caret)
library(randomForest)


# Split your data into training and testing sets
set.seed(123)  # for reproducibility
split_index <- createDataPartition(undersampled_dataset$Diabetes_Type, p = 0.8, list = FALSE)
train_data <- undersampled_dataset[split_index, ]
test_data <- undersampled_dataset[-split_index, ]

train_data$Diabetes_Type <- as.factor(train_data$Diabetes_Type)

# Train the Random Forest model
# Adjust the formula based on your actual variable names
rf_model <- randomForest(Diabetes_Type ~ ., data = train_data, ntree = 100)
# Make predictions on the test set
predictions <- predict(rf_model, newdata = test_data)

# Evaluate the model performance
confusion_matrix_rf <- table(predictions, test_data$Diabetes_Type)
print(confusion_matrix_rf)

# Calculate accuracy
accuracy_rf_undersampling <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
print(paste("Accuracy:", round(accuracy, 4)))

sensitivity <- confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[2, ])
specificity <- confusion_matrix_rf[1, 1] / sum(confusion_matrix_rf[1, ])

print(paste("Sensitivity:", round(sensitivity, 4)))
print(paste("Specificity:", round(specificity, 4)))
#**********Oversampled***********#

# Split your data into training and testing sets
set.seed(129)  # for reproducibility
split_index <- createDataPartition(oversampled_dataset$Diabetes_Type, p = 0.8, list = FALSE)
train_data <- oversampled_dataset[split_index, ]
test_data <- oversampled_dataset[-split_index, ]

train_data$Diabetes_Type <- as.factor(train_data$Diabetes_Type)

# Train the Random Forest model
# Adjust the formula based on your actual variable names
rf_model_oversampling <- randomForest(Diabetes_Type ~ ., data = train_data, ntree = 100)

# Make predictions on the test set
predictions <- predict(rf_model_oversampling, newdata = test_data)

# Evaluate the model performance
confusion_matrix_rf <- table(predictions, test_data$Diabetes_Type)
print(confusion_matrix_rf)

# Calculate accuracy
accuracy_rf_smote <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
print(paste("Accuracy:", round(accuracy, 4)))

sensitivity <- confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[2, ])
specificity <- confusion_matrix_rf[1, 1] / sum(confusion_matrix_rf[1, ])

print(paste("Sensitivity:", round(sensitivity, 4)))
print(paste("Specificity:", round(specificity, 4)))

###**Complete Dataset**###

# Split your data into training and testing sets
set.seed(131)  # for reproducibility
split_index <- createDataPartition(dat$Diabetes_Type, p = 0.8, list = FALSE)
train_data <- oversampled_dataset[split_index, ]
test_data <- oversampled_dataset[-split_index, ]

train_data$Diabetes_Type <- as.factor(train_data$Diabetes_Type)

# Train the Random Forest model
# Adjust the formula based on your actual variable names
rf_model_complete <- randomForest(Diabetes_Type ~ ., data = train_data, ntree = 100)

# Make predictions on the test set
predictions <- predict(rf_model_complete, newdata = test_data)

# Evaluate the model performance
confusion_matrix_rf <- table(predictions, test_data$Diabetes_Type)
print(confusion_matrix_rf)

# Calculate accuracy
accuracy_rf_fullmodel <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
print(paste("Accuracy:", round(accuracy, 4)))

sensitivity <- confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[2, ])
specificity <- confusion_matrix_rf[1, 1] / sum(confusion_matrix_rf[1, ])

print(paste("Sensitivity:", round(sensitivity, 4)))
print(paste("Specificity:", round(specificity, 4)))
