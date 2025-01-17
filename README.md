# Machine-Learning_Electricity-Load-Due-to-Climate-Change

## A) DATA PREPROCESSING

```{r data_processing}
# Load data
data <- read.csv("Data.csv")

# Exclude "Month", "Day", and "Hour" from the data
data <- data[, !(names(data) %in% c("Month", "Day", "Hour"))]

# Convert necessary columns to factors
data$Season <- as.factor(data$Season)
data$Day.of.the.week <- as.factor(data$Day.of.the.week)
data$Time.of.the.day <- as.factor(data$Time.of.the.day)

# Identify numeric columns to standardize
numcol <- sapply(data, is.numeric)

# Exclude "Demand Actual" the dependent variable
numcol["Demand.Actual"] <- FALSE 

# Standardize numeric predictors
data_std <- data
data_std[, numcol] <- scale(data[, numcol], 
                            center = TRUE, 
                            scale = TRUE)

# Create full matrix
## Include dummy variables for factors
## One dummy variable is removed for each factor
matrix <- model.matrix(Demand.Actual ~ ., data_std)[, -1]

# Add "Demand Actual" back
data_std <- cbind(Demand.Actual = data$Demand.Actual, 
                  as.data.frame(matrix))
head(data_std)
```

## B) EXPLORATORY DATA ANALYSIS

**Visualization of Dataset**

```{r}
# Load necessary libraries
library(ggplot2)
library(reshape2)

# Histogram of Demand Actual
hist(data_std$Demand.Actual, main="Histogram of Demand Actual", xlab="Demand", col="blue", breaks=30)

# Boxplot of Demand Actual
boxplot(data_std$Demand.Actual, main="Boxplot of Demand Actual", horizontal=TRUE)

# Correlation matrix
numcol <- sapply(data_std, is.numeric)
cor_matrix <- cor(data_std[, numcol], use = "complete.obs")
cor_melted <- melt(cor_matrix)
ggplot(cor_melted, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") + 
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name = "Correlation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed() +
  labs(title = "Correlation Heatmap for All Variables", x = "Variables", y = "Variables")

```


## C) INITIAL MODEL BUILDING

**MODEL 1: Fitting a linear model using Ordinary Least Squares**

```{r ols}
# Load the library for evaluation metrics
library(Metrics)

# Split the data into training and testing sets
set.seed(123)
train_index <- sample(1:nrow(data_std), 
                      size = 0.8 * nrow(data_std))
train_data <- data_std[train_index, ]
test_data <- data_std[-train_index, ]

# Fit the OLS model
ols_model <- lm(Demand.Actual ~ ., data = train_data)

# Make predictions on the test set
ols_predictions <- predict(ols_model, newdata = test_data)

# Evaluate model performance
mse_ols <- mse(test_data$Demand.Actual, ols_predictions)
rmse_ols <- rmse(test_data$Demand.Actual, ols_predictions)
r2_ols <- 1 - sum((test_data$Demand.Actual - ols_predictions)^2) / 
                  sum((test_data$Demand.Actual - mean(test_data$Demand.Actual))^2)

cat("Model Evaluation:\n")
cat("Mean Squared Error (MSE):", mse_ols, "\n")
cat("Root Mean Squared Error (RMSE):", rmse_ols, "\n")
cat("R-squared:", r2_ols, "\n")

# Plot predictions vs actual values
plot(test_data$Demand.Actual, ols_predictions, 
     xlab = "Actual Values", 
     ylab = "Predicted Values", 
     main = "OLS: Predicted vs Actual")
abline(0, 1, col = "red", lwd = 2)  # Add a reference line
```

**b) Assumption Checking for OLS Model**

1.  Linearity

```{r}
# Scatterplot matrix for numeric predictors and response variable
# Numeric predictors vs response variable
numeric_predictors <- data_std[, numcol]
scatter_data <- cbind(Demand.Actual = data_std$Demand.Actual, numeric_predictors)

pairs(scatter_data,
      main = "Scatterplot Matrix for Linearity Check")
```

2.  Constant Variance (Homoscedasticity)

```{r}
# Residuals vs Fitted Values plot
plot(ols_model$fitted.values, ols_model$residuals,
     main = "Residuals vs Fitted Values",
     xlab = "Fitted Values",
     ylab = "Residuals",
     pch = 20)  # Points style
abline(h = 0, col = "red", lwd = 2)  # Horizontal line at zero
```

3.  Normality Check

```{r}
# Extract numeric residuals from the model
residuals_ols <- residuals(ols_model)

# Histogram of residuals
hist(residuals_ols, 
     breaks = 30, 
     col = "orange", 
     main = "Histogram of Residuals with Normal Curve", 
     xlab = "Residuals", 
     freq = FALSE)  # Plot density instead of counts

# Density line for the residuals
lines(density(residuals_ols),  
      col = "red", 
      lwd = 2)

# Add a theoretical normal distribution curve
x_vals <- seq(min(residuals_ols), max(residuals_ols), length = 100)
y_vals <- dnorm(x_vals, mean = mean(residuals_ols), sd = sd(residuals_ols))
lines(x_vals, y_vals, col = "blue", lwd = 2, lty = 2)  # Normal curve

# Add legend
legend("topright", 
       legend = c("Density of Residuals", "Normal Distribution"), 
       col = c("red", "blue"), 
       lwd = 2, 
       lty = c(1, 2))

# Q-Q Plot of residuals
qqnorm(residuals_ols, main = "Q-Q Plot of Residuals")
qqline(residuals_ols, col = "red")  # Reference line
```

**ANALYSIS:** 

In general, the assumption check plots above indicate potential issues with the OLS model:

-   Heteroscedasticity: The residuals show a non-constant variance, with wider spread at lower fitted values and clustering at higher fitted values. This violates the assumption of homoscedasticity, which is critical for reliable statistical inference.

-   Outliers: There are notable outliers that could unduly influence the model's estimates and reduce its predictive accuracy.

-   Possible Non-linearity: Although there is no strong visible pattern in the residuals, some clustering may hint at potential non-linearity or omitted variable bias.

Conclusion: The model may not fully satisfy the assumptions of linear regression. To address this issues, the alternative modeling approaches will be applied and also be compared with the OLS metrics.


**MODEL 2: Fitting a non-linear model using Support Vector Regression**

```{r svr}
# Load the necessary library
library(e1071)
library(caret)
library(Metrics)

# set the tune parameter: epsilon and cost
tune_grid <- expand.grid(
  C = c(0.1, 1, 10, 100),
  sigma = c(0.01, 0.1, 1)
)

# Set the train control for cross-validation
tr <- trainControl(method = "cv", number = 5)
# Perform the grid search
svr_tuning <- train(Demand.Actual ~ ., 
                    data = train_data, 
                    method = "svmRadial", 
                    tuneGrid = tune_grid, 
                    trControl = tr)
                   
# Final fit with the best hyperparameters
svr_model <- svm(Demand.Actual ~ ., 
                 data = train_data, 
                 type = "eps-regression",  
                 kernel = "radial",        
                 cost = svr_tuning$bestTune$C,
                 gamma = svr_tuning$bestTune$sigma,
                 epsilon = 0.1)

cat("Best hyperparameters:\n")
print(svr_tuning$bestTune)

# Make predictions on the test set
svr_predictions <- predict(svr_model, test_data)

# Evaluate model performance
mse_svr <- mse(test_data$Demand.Actual, svr_predictions)
rmse_svr <- rmse(test_data$Demand.Actual, svr_predictions)
r2_svr <- 1 - sum((test_data$Demand.Actual - svr_predictions)^2) / 
                  sum((test_data$Demand.Actual - mean(test_data$Demand.Actual))^2)

cat("Model Evaluation:\n")
cat("Mean Squared Error (MSE):", mse_svr, "\n")
cat("Root Mean Squared Error (RMSE):", rmse_svr, "\n")
cat("R-squared:", r2_svr, "\n")

# Plot predictions vs actual values
plot(test_data$Demand.Actual, svr_predictions, 
     xlab = "Actual Values", 
     ylab = "Predicted Values", 
     main = "SVR: Predicted vs Actual")
abline(0, 1, col = "red", lwd = 2)  # Add a reference line
```

**MODEL 3: Fitting a non-linear model using Bagging**

```{r bagging}
# Load the necessary library
library(randomForest)
library(Metrics)

# Fit the bagging model
bag_model <- randomForest(Demand.Actual ~ ., 
                           data = train_data,
                           mtry = ncol(train_data) - 1,  # Set mtry to number of predictors
                           importance = TRUE)            # Compute feature importance

# Make predictions on the test data
bag_predictions <- predict(bag_model, newdata = test_data)

# Evaluate model performance
mse_bag <- mse(test_data$Demand.Actual, bag_predictions)
rmse_bag <- rmse(test_data$Demand.Actual, bag_predictions)
r2_bag <- 1 - sum((test_data$Demand.Actual - bag_predictions)^2) / 
                  sum((test_data$Demand.Actual - mean(test_data$Demand.Actual))^2)

cat("Model Evaluation:\n")
cat("Mean Squared Error (MSE):", mse_bag, "\n")
cat("Root Mean Squared Error (RMSE):", rmse_bag, "\n")
cat("R-squared:", r2_bag, "\n")

# Plot predictions vs actual values
plot(test_data$Demand.Actual, bag_predictions, 
     xlab = "Actual Values", 
     ylab = "Predicted Values", 
     main = "Bagged Random Forest: Predicted vs Actual")
abline(0, 1, col = "red", lwd = 2)  # Add a reference line

# Feature importance plot
importance <- importance(bag_model)  # Extract importance measures
varImpPlot(bag_model)               # Plot feature importance
```

**MODEL 4: Fitting a non-linear model using Random Forest**

```{r rf}
# Load the necessary library
library(randomForest)
library(Metrics)

# Fit the Random Forest model
rf_model <- randomForest(Demand.Actual ~ ., 
                         data = train_data,
                         mtry = floor(sqrt(ncol(train_data) - 1)),  # Optimal mtry for Random Forest
                         importance = TRUE)                         # Compute feature importance

# Make predictions on the test data
rf_predictions <- predict(rf_model, newdata = test_data)

# Evaluate model performance
mse_rf <- mse(test_data$Demand.Actual, rf_predictions)
rmse_rf <- rmse(test_data$Demand.Actual, rf_predictions)
r2_rf <- 1 - sum((test_data$Demand.Actual - rf_predictions)^2) / 
                  sum((test_data$Demand.Actual - mean(test_data$Demand.Actual))^2)

cat("Model Evaluation:\n")
cat("Mean Squared Error (MSE):", mse_rf, "\n")
cat("Root Mean Squared Error (RMSE):", rmse_rf, "\n")
cat("R-squared:", r2_rf, "\n")

# Plot predictions vs actual values
plot(test_data$Demand.Actual, rf_predictions, 
     xlab = "Actual Values", 
     ylab = "Predicted Values", 
     main = "Random Forest: Predicted vs Actual")
abline(0, 1, col = "red", lwd = 2)  # Add a reference line

# Feature importance plot
importance <- importance(rf_model)  # Extract importance measures
varImpPlot(rf_model)               # Plot feature importance
```

**MODEL 5: Fitting a non-linear model using Boosting**

```{r boosting}
# Load the necessary library
library(gbm)
library(Metrics)

# Fit the Boosting model
set.seed(123)

gr <- expand.grid(
  shrinkage = c(0.3, 0.1, 0.05, 0.01, 0.001),
  interaction.depth = c(1,3,5),
  n.trees = seq(1000,7000, by=500),
  n.minobsinnode = 10
)

boosting_tune <- train(Demand.Actual ~ ., 
                    data = train_data, 
                    method = "gbm", 
                    tuneGrid = gr, 
                    trControl = tr,
                    verbose = FALSE)

# Display the best parameters
print(boosting_tune$bestTune)

boosting_model <- gbm(
      formula = Demand.Actual ~ ., 
      data = train_data, 
      distribution = "gaussian", 
      n.trees = boosting_tune$bestTune$n.trees, 
      interaction.depth = boosting_tune$bestTune$interaction.depth, 
      shrinkage = boosting_tune$bestTune$shrinkage, 
      n.minobsinnode = boosting_tune$bestTune$n.minobsinnode, 
      cv.folds = 5
)

# Identify the optimal number of trees using cross-validation
optimal_trees <- gbm.perf(boosting_model, method = "cv", plot.it = TRUE)

# Make predictions on the test data
boosting_predictions <- predict(boosting_model, newdata = test_data, n.trees = optimal_trees)

# Evaluate model performance
mse_boost <- mse(test_data$Demand.Actual, boosting_predictions)
rmse_boost <- rmse(test_data$Demand.Actual, boosting_predictions)
r2_boost <- 1 - sum((test_data$Demand.Actual - boosting_predictions)^2) / 
                   sum((test_data$Demand.Actual - mean(test_data$Demand.Actual))^2)

cat("Model Evaluation:\n")
cat("Mean Squared Error (MSE):", mse_boost, "\n")
cat("Root Mean Squared Error (RMSE):", rmse_boost, "\n")
cat("R-squared:", r2_boost, "\n")

# Plot predictions vs actual values
plot(test_data$Demand.Actual, boosting_predictions, 
     xlab = "Actual Values", 
     ylab = "Predicted Values", 
     main = "Boosting: Predicted vs Actual")
abline(0, 1, col = "red", lwd = 2)  # Add a reference line

# Summary of variable importance
summary(boosting_model, n.trees = optimal_trees)
```

## D) PERFORMANCE METRICS OF INITIAL MODELS

**Summary of performance metrics for all initial models with all predictors** (Using different methods to fit the weather data)

```{r summary}
# Performance metrics for each model with all predictors
full_model_summary <- data.frame(
  Model = c("OLS", "SVR", "Bagging", "Random Forest", "Boosting"),
  MSE = c(mse_ols, mse_svr, mse_bag, mse_rf, mse_boost),
  RMSE = c(rmse_ols, rmse_svr, rmse_bag, rmse_rf, rmse_boost),
  R_squared = c(r2_ols, r2_svr, r2_bag, r2_rf, r2_boost)
)
print(full_model_summary)
```


## E) REGULARIZATION/SHRINKAGE

```{r}
# Load library
library(glmnet)

# Prepare data for glmnet (model.matrix creates a matrix suitable for glmnet)
x_train <- model.matrix(Demand.Actual ~ ., data = train_data)[, -1]
y_train <- train_data$Demand.Actual

x_test <- model.matrix(Demand.Actual ~ ., data = test_data)[, -1]
y_test <- test_data$Demand.Actual

# 1) Ridge Regression
ridge_model <- cv.glmnet(x_train, y_train, alpha = 0)  # alpha = 0 for Ridge
ridge_coef <- coef(ridge_model, s = "lambda.min")[-1, 1]
selected_ridge <- names(ridge_coef[ridge_coef != 0])

ridge_predictions <- predict(ridge_model, x_test, s = "lambda.min")
ridge_r2 <- R2(ridge_predictions, y_test)
ridge_mse <- mse(y_test, ridge_predictions)
ridge_rmse <- rmse(y_test, ridge_predictions)

# 2) Lasso Regression
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)  # alpha = 1 for Lasso
lasso_coef <- coef(lasso_model, s = "lambda.min")[-1, 1]
selected_lasso <- names(lasso_coef[lasso_coef != 0])

lasso_predictions <- predict(lasso_model, x_test, s = "lambda.min")
lasso_r2 <- R2(lasso_predictions, y_test)
lasso_mse <- mse(y_test, lasso_predictions)
lasso_rmse <- rmse(y_test, lasso_predictions)

# 3) ElasticNet
elastic_model <- cv.glmnet(x_train, y_train, alpha = 0.5)  # alpha = 0.5 for ElasticNet
elastic_coef <- coef(elastic_model, s = "lambda.min")[-1, 1]
selected_elastic <- names(elastic_coef[elastic_coef != 0])

elastic_predictions <- predict(elastic_model, x_test, s = "lambda.min")
elastic_r2 <- R2(elastic_predictions, y_test)
elastic_mse <- mse(y_test, elastic_predictions)
elastic_rmse <- rmse(y_test, elastic_predictions)
```

**Summary: Variable Selection and Regularization**

```{r}
# Summary of selected variables
all_variables <- unique(c(
  selected_ridge,
  selected_lasso,
  selected_elastic
))

selection_summary <- sapply(all_variables, function(var) {
  c(
    Ridge = ifelse(var %in% selected_ridge, "✓", ""),
    Lasso = ifelse(var %in% selected_lasso, "✓", ""),
    ElasticNet = ifelse(var %in% selected_elastic, "✓", "")
  )
}) |> t()

selection_summary_df <- as.data.frame(selection_summary)
selection_summary_df <- cbind(Variable = rownames(selection_summary), selection_summary_df)
rownames(selection_summary_df) <- NULL

# Summary table
selection_metrics <- data.frame(
  Method = c("Ridge", "Lasso", "ElasticNet"),
  R2 = c(ridge_r2, lasso_r2, elastic_r2),
  MSE = c(ridge_mse, lasso_mse, elastic_mse),
  RMSE = c(ridge_rmse, lasso_rmse, elastic_rmse)
)

cat("Summary of Selected Variables:\n")
print(selection_summary_df)

cat("\nPerformance Metrics for Variable Selection and Regularization:\n")
print(selection_metrics)
```


## F) FINAL MODEL BUILDING

(Refit the Boosting model with reduced predictors)

**Reduced Predictors**

```{r}
# Remove the variables 'Solar.radiation' and 'solar.radiation..clear.sky.' from the dataset
reduced_train_data <- train_data[, !names(train_data) %in% c("Solar.radiation", "Solar.radiation..clear.sky.")]
reduced_test_data <- test_data[, !names(test_data) %in% c("Solar.radiation", "Solar.radiation..clear.sky.")]
```

**Boosting Model with Reduced Predictors**

```{r}
# Boosting (Gradient Boosting Machine)
library(gbm)

set.seed(123)  # For reproducibility

final_boosting_model <- gbm(
  Demand.Actual ~ ., 
  data = reduced_train_data, 
  distribution = "gaussian", 
  n.trees = 5000,            # Maximum number of trees
  interaction.depth = 5,     # Depth of trees
  shrinkage = 0.01,          # Learning rate
  cv.folds = 5               # Number of cross-validation folds
)

# Determine the optimal number of trees using cross-validation
best_trees <- gbm.perf(final_boosting_model, method = "cv", plot.it = TRUE)

# Make predictions on test data
final_predictions <- predict(final_boosting_model, 
                              newdata = reduced_test_data, 
                              n.trees = best_trees)

# Performance Metrics
final_mse <- mse(reduced_test_data$Demand.Actual, final_predictions)
final_rmse <- rmse(reduced_test_data$Demand.Actual, final_predictions)
final_r2 <- 1 - sum((reduced_test_data$Demand.Actual - final_predictions)^2) / 
                 sum((reduced_test_data$Demand.Actual - mean(reduced_test_data$Demand.Actual))^2)
cat("Final Model Evaluation with Reduced Predictors:\n")
cat("Mean Squared Error (MSE):", final_mse, "\n")
cat("Root Mean Squared Error (RMSE):", final_rmse, "\n")
cat("R-squared:", final_r2, "\n")

plot(reduced_test_data$Demand.Actual, final_predictions, 
     xlab = "Actual Values", 
     ylab = "Predicted Values", 
     main = "Final Boosting Model: Predicted vs Actual")
abline(0, 1, col = "red", lwd = 2)  

# Summary of variable importance
summary(boosting_model, n.trees = optimal_trees)
```

## G) PERFORMANCE METRICS OF FINAL MODEL

**Summary of Boosting Model with Full Predictors vs with Reduced Predictors**

```{r}
# Metrics for Full Predictors Model
full_mse <- mse(test_data$Demand.Actual, boosting_predictions)
full_rmse <- rmse(test_data$Demand.Actual, boosting_predictions)
full_r2 <- 1 - sum((test_data$Demand.Actual - boosting_predictions)^2) / 
                  sum((test_data$Demand.Actual - mean(test_data$Demand.Actual))^2)

# Metrics for Reduced Predictors Model
reduced_mse <- mse(reduced_test_data$Demand.Actual, final_predictions)
reduced_rmse <- rmse(reduced_test_data$Demand.Actual, final_predictions)
reduced_r2 <- 1 - sum((reduced_test_data$Demand.Actual - final_predictions)^2) / 
                     sum((reduced_test_data$Demand.Actual - mean(reduced_test_data$Demand.Actual))^2)

performance_comparison <- data.frame(
  Model = c("Boosting (Full Predictors)", "Boosting (Reduced Predictors)"),
  MSE = c(full_mse, reduced_mse),
  RMSE = c(full_rmse, reduced_rmse),
  R_squared = c(full_r2, reduced_r2)
)
cat("Performance Metrics Comparison:\n")
print(performance_comparison)
```


## H) FINAL MODEL VALIDATION

1) The Chi-Square test: to check if the distribution of predicted values matches the distribution of observed values.

```{r}
# Observed: Actual Demand Values
observed <- reduced_test_data$Demand.Actual

# Predicted: Predicted Values from final Boosting Model
predicted <- final_predictions

# Bin the observed & predicted values into intervals
observed_bins <- cut(observed, breaks = quantile(observed, probs = seq(0, 1, 0.1)), include.lowest = TRUE)
predicted_bins <- cut(predicted, breaks = quantile(predicted, probs = seq(0, 1, 0.1)), include.lowest = TRUE)

# Frequency tables
observed_freq <- table(observed_bins)
predicted_freq <- table(predicted_bins)

# Perform the Chi-Square Test
chi_square_test <- chisq.test(x = observed_freq, p = predicted_freq / sum(predicted_freq))
cat("Chi-Square Test of Goodness of Fit for Final Model (Boosting):\n")
print(chi_square_test)
```

2) Residual Analysis: involves analyzing the differences between observed and predicted values to check for patterns or systematic errors.

```{r}
# Calculate residuals
boosting_residuals <- reduced_test_data$Demand.Actual - final_predictions

# Residual plot
plot(boosting_predictions, boosting_residuals,
     main = "Residual Plot for Final Model (Boosting)",
     xlab = "Predicted Values",
     ylab = "Residuals",
     pch = 20, col = "blue")
abline(h = 0, col = "red", lwd = 2)  # Add a horizontal reference line at 0

# Histogram of residuals
hist(boosting_residuals, 
     main = "Histogram of Residuals for Final Model (Boosting)", 
     xlab = "Residuals", 
     col = "lightblue", 
     breaks = 20,
     freq = FALSE)

# Density line for the residuals
lines(density(boosting_residuals),  
      col = "red", 
      lwd = 2)

# Add a theoretical normal distribution curve
x_vals <- seq(min(boosting_residuals), max(boosting_residuals), length = 100)
y_vals <- dnorm(x_vals, mean = mean(boosting_residuals), sd = sd(boosting_residuals))
lines(x_vals, y_vals, col = "blue", lwd = 2, lty = 2)  # Normal curve

# Add legend
legend("topright", 
       legend = c("Density of Residuals", "Normal Distribution"), 
       col = c("red", "blue"), 
       lwd = 2, 
       lty = c(1, 2))

# Q-Q Plot of residuals
qqnorm(boosting_residuals, main = "Q-Q Plot of Residuals")
qqline(boosting_residuals, col = "red")  # Reference line
```
