# **********************************************************************
# Fitting the Non-Linear Model to the Dataset ----
#
# Purpose ----
# To fit the model to the dataset. The fitting is required to identify
# the numeric values of the parameters, which cannot variate (unlike
# the variables provided as input).
# **********************************************************************

# Install and Load the Required Packages ----
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Load the Dataset ----
# URL of the dataset
url <- "https://raw.githubusercontent.com/course-files/ME555-Design-Optimization-for-Welfare-Economics/main/Datasets/siaya.csv" # nolint

# Load the dataset from the URL
data <- read.csv(url)

# View the first few rows of the dataset
head(data)

# View the correlations
View(cor(data))

# Data Transformation ----
# Purpose of the data transformation: To expose the data to the algorithms
# in a better way.

## Yeo-Johnson Transform on the Dataset ----
# The Yeo-Johnson power transform reduces the skewness by shifting the
# distribution of a variable and making the variable have a more
# Gaussian-like distribution. It can handle 0 and negative values in the
# variable.

# BEFORE
summary(data)

# Calculate the skewness (measure of asymmetry) before the Yeo-Johnson
# transform
# 1.	Skewness between -0.4 and 0.4 (inclusive) implies that there is no skew
# in the distribution of results; the distribution of results is symmetrical;
# it is a normal distribution; a Gaussian distribution.
# 2.	Skewness above 0.4 implies a positive skew; a right-skewed distribution.
# 3.	Skewness below -0.4 implies a negative skew; a left-skewed distribution.
sapply(data,  skewness, type = 2)

# Plot histograms of key variables to visualize the skewness before the
# Yeo-Johnson transform
hist(data[, 24], main = names(data)[24])
hist(data[, 25], main = names(data)[25])
hist(data[, 26], main = names(data)[26])
hist(data[, 27], main = names(data)[27])
hist(data[, 28], main = names(data)[28])
hist(data[, 29], main = names(data)[29])
hist(data[, 30], main = names(data)[30])
hist(data[, 31], main = names(data)[31])
hist(data[, 32], main = names(data)[32])

model_of_the_transform <- preProcess(data, method = c("YeoJohnson"))
print(model_of_the_transform)
data_yeo_johnson_transform <- predict(model_of_the_transform, data)

# AFTER
summary(data_yeo_johnson_transform)

# Calculate the skewness after the Yeo-Johnson transform
sapply(data_yeo_johnson_transform,  skewness, type = 2)

# Plot histograms of key variables to visualize the skewness after the
# Yeo-Johnson transform
hist(data_yeo_johnson_transform[, 24], main = names(data_yeo_johnson_transform)[24])
hist(data_yeo_johnson_transform[, 25], main = names(data_yeo_johnson_transform)[25])
hist(data_yeo_johnson_transform[, 26], main = names(data_yeo_johnson_transform)[26])
hist(data_yeo_johnson_transform[, 27], main = names(data_yeo_johnson_transform)[27])
hist(data_yeo_johnson_transform[, 28], main = names(data_yeo_johnson_transform)[28])
hist(data_yeo_johnson_transform[, 29], main = names(data_yeo_johnson_transform)[29])
hist(data_yeo_johnson_transform[, 30], main = names(data_yeo_johnson_transform)[30])
hist(data_yeo_johnson_transform[, 31], main = names(data_yeo_johnson_transform)[31])
hist(data_yeo_johnson_transform[, 32], main = names(data_yeo_johnson_transform)[32])

## The Normalize Transform on the Dataset ----
# Purpose: Ensures the numerical data are between [0, 1] (inclusive)
# This makes it easier to work with the data
model_of_the_transform <- preProcess(data_yeo_johnson_transform, method = c("range"))
print(model_of_the_transform)
data_normalize_transform <- predict(model_of_the_transform, data_yeo_johnson_transform)
summary(data_normalize_transform)

## The Standardize Transform on the Dataset ----
# Purpose: Ensures that each numeric attribute has a mean value of 0 and a
# standard deviation of 1. This is done by combining the scale data
# transform (divide each value by the standard deviation) and the centre data
# transform (subtract the mean from each value).
# BEFORE
summary(data_normalize_transform)
sapply(data_normalize_transform[,], sd)

model_of_the_transform <- preProcess(data_normalize_transform,
                                     method = c("scale", "center"))
print(model_of_the_transform)
data_standardize_transform <- predict(model_of_the_transform, data_normalize_transform)

# AFTER
summary(data_standardize_transform)
sapply(data_standardize_transform[,], sd)

data <- data_standardize_transform

# Split the Dataset ----
# Define an 80:10:10 train:test:validate data split of the dataset.
# That is, 80% of the original data will be used to train the model,
# 10% of the original data can be used to test the model and the
# remaining 10% of the original data can be used to validate the model.

train_index1 <- createDataPartition(data$i,
                                    p = 0.8,
                                    list = FALSE)
train_data <- data[train_index1, ]
test_validate <- data[-train_index1, ]

train_index2 <- createDataPartition(test_validate$i,
                                    p = 0.5,
                                    list = FALSE)
test_data <- test_validate[train_index2, ]
validate_data <- test_validate[-train_index2, ]

# Fit the Model to the Dataset ----
# We use a 5-fold cross validation with 3 repeats
# We also apply the standardize data transform while fitting the model
## Train Control ----
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

## 1. Generalized Linear Model (glm) ----
model_glm <-
  train(qs_stddev ~ r_var + water_energy_util_var +
          dem_var + supp_var + diff_sup_dem_var + tal_var +
          vert_s + exp_var + mal_var,
        data = train_data,
        na.action = na.omit, method = "glm", metric = "RMSE",
        preProcess = c("center", "scale"),
        trControl = train_control)

## 2. Boosted Generalized Linear Model (glmboost) ----
model_glmboost <-
  train(qs_stddev ~ r_var + water_energy_util_var +
          dem_var + supp_var + diff_sup_dem_var + tal_var +
          vert_s + exp_var + mal_var,
        data = train_data,
        na.action = na.omit, method = "glmboost", metric = "RMSE",
        preProcess = c("center", "scale"),
        trControl = train_control)

## 3. Bayesian Generalized Linear Model (bayesglm) ----
model_bayesglm <-
  train(qs_stddev ~ r_var + water_energy_util_var +
          dem_var + supp_var + diff_sup_dem_var + tal_var +
          vert_s + exp_var + mal_var,
        data = train_data,
        na.action = na.omit, method = "bayesglm", metric = "RMSE",
        preProcess = c("center", "scale"),
        trControl = train_control)

## 4. Generalized Linear Model with Stepwise Feature Selection (glmStepAIC) ----
model_glmStepAIC <-
  train(qs_stddev ~ r_var + water_energy_util_var +
          dem_var + supp_var + diff_sup_dem_var + tal_var +
          vert_s + exp_var + mal_var,
        data = train_data,
        na.action = na.omit, method = "glmStepAIC", metric = "RMSE",
        preProcess = c("center", "scale"),
        trControl = train_control)

## 5. Partial Least Squares Generalized Linear Models (plsRglm) ----
model_plsRglm <-
  train(qs_stddev ~ r_var + water_energy_util_var +
          dem_var + supp_var + diff_sup_dem_var + tal_var +
          vert_s + exp_var + mal_var,
        data = train_data,
        na.action = na.omit, method = "plsRglm", metric = "RMSE",
        preProcess = c("center", "scale"),
        trControl = train_control)

# Model Performance Comparison ----
### Notes on R_Squared in Engineering, Physics, and Economics ----
# A “good” R_Squared value depends heavily on the context of the research,
# the domain, and the specific application. In fields where data can be highly
# controlled and variables are well-understood, such as physics or engineering,
# a high R_Squared value close to 1 might be expected for a "good" model.

# In contrast, in fields dealing with complex systems or human behaviour, such
# as social sciences or economics, a lower R_Squared might still be considered
# acceptable. In these fields, phenomena are influenced by many factors, some of
# which are difficult to measure or include in the model, leading to lower
# R_Squared values.

# General guidelines:
## •	R_Squared > 0.9: The model explains most of the variability of the response
#     data around its mean, which is typically considered excellent in many
#     applications.
## •	0.7 ≤ R_Squared ≤ 0.9: The model explains a substantial amount of
#     variability, considered good in many fields.
## •	0.5 < R_Squared ≤0.7: The model explains a moderate amount of variability,
#     which can be acceptable depending on the domain.
## •	R_Squared ≤0.5: The model does not explain much variability in the
#     response; its predictive capability can be considered weak.

# However, a model can have a high R_Squared value but might still fail at
# predictive tasks, especially if it is overfitting the training data.
# Conversely, a model with a lower R_Squared value might be more generalizable
# or useful in practice.

## Resamples Function ----
# Call the `resamples()` Function to Compare the Performance
# We then create a list of the model results and pass the list as an argument
# to the `resamples` function.
results <- resamples(list("Boosted \nGeneralized Linear \nModel (glmboost)" = model_glmboost, "Bayesian \nGeneralized Linear \nModel (bayesglm)" = model_bayesglm, "Generalized Linear Model \nwith Stepwise Feature \nSelection (glmStepAIC)" = model_glmStepAIC, "Partial Least Squares \nGeneralized Linear \nModels (plsRglm)" = model_plsRglm))

## 1. Table Summary ----
summary(results)

## 2. Box and Whisker Plot ----
scales <- list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(results, scales = scales)

## 3. Dot Plots ----
# They show both the mean estimated accuracy as well as the 95% confidence
# interval (e.g. the range in which 95% of observed scores fell).
scales <- list(x = list(relation = "free"), y = list(relation = "free"))
dotplot(results, scales = scales)

# Display the Parameter Values based on the Best-Performing Model ----
# The best-performing model is:
# The Generalized Linear Model with Stepwise Feature Selection (glmStepAIC)
# "Best" is defined as the one with the highest R_Squared value (0.8706160).
# Other metrics considered are the lowest RMSE and the lowest MAE.
summary(model_glmStepAIC)
saveRDS(model_glmStepAIC, "./1.Curve-Fitting/Models/model_glmStepAIC.rds")
