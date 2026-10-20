
# HEART DISEASE PREDICTION
# Logistic Regression | R

# 1. Load dataset
heart <- read.csv(
  "data/heart_disease.csv",
  na.strings = c("", "?", "NA")
)

cat("Dataset dimensions:", dim(heart), "\n")
print(head(heart))
print(colSums(is.na(heart)))

# 2. Prepare target and predictors
stopifnot("num" %in% names(heart))
heart$num <- as.integer(heart$num > 0)

features <- setdiff(names(heart), "num")

# Reproduce original assignment's numeric encoding
heart[features] <- lapply(
  heart[features], as.numeric
)

# 3. Stratified 80/20 train-test split
set.seed(123)

train_index <- unlist(lapply(
  split(seq_len(nrow(heart)), heart$num),
  function(idx) sample(idx, floor(0.8 * length(idx)))
))

train <- heart[train_index, ]
test <- heart[-train_index, ]

cat("Training rows:", nrow(train), "\n")
cat("Testing rows:", nrow(test), "\n")

# 4. Impute missing values using training medians
for (feature in features) {
  med <- median(train[[feature]], na.rm = TRUE)
  
  if (!is.finite(med)) {
    stop(paste("No observed training values:", feature))
  }
  
  train[[feature]][is.na(train[[feature]])] <- med
  test[[feature]][is.na(test[[feature]])] <- med
}

# 5. Scale using training statistics only
train_means <- sapply(train[features], mean)
train_sds <- sapply(train[features], sd)
train_sds[train_sds == 0] <- 1

train[features] <- as.data.frame(
  scale(train[features],
        center = train_means,
        scale = train_sds)
)

test[features] <- as.data.frame(
  scale(test[features],
        center = train_means,
        scale = train_sds)
)

# 6. Train Logistic Regression
model <- glm(
  num ~ .,
  data = train,
  family = binomial()
)

print(summary(model))

# 7. Predict test observations
probabilities <- predict(
  model,
  newdata = test,
  type = "response"
)

predictions <- ifelse(probabilities >= 0.5, 1L, 0L)
actual <- test$num

# 8. Confusion matrix
confusion <- table(
  Predicted = factor(predictions, levels = c(0, 1)),
  Actual = factor(actual, levels = c(0, 1))
)

print(confusion)

# 9. Calculate evaluation metrics
TP <- confusion["1", "1"]
TN <- confusion["0", "0"]
FP <- confusion["1", "0"]
FN <- confusion["0", "1"]

safe_divide <- function(a, b) {
  if (b == 0) return(NA_real_)
  as.numeric(a / b)
}

accuracy <- (TP + TN) / sum(confusion)
precision <- safe_divide(TP, TP + FP)
recall <- safe_divide(TP, TP + FN)

f1 <- if (is.na(precision) || is.na(recall)) {
  NA_real_
} else {
  safe_divide(2 * precision * recall, precision + recall)
}

results <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1-Score"),
  Score = round(c(accuracy, precision, recall, f1), 4)
)

print(results)

# 10. Save results
dir.create("outputs", showWarnings = FALSE)

write.csv(
  results,
  "outputs/model_metrics.csv",
  row.names = FALSE
)

write.csv(
  as.data.frame.matrix(confusion),
  "outputs/confusion_matrix.csv"
)

cat("Model completed successfully!\n")





print(confusion)
print(results)


# HEART DISEASE PREDICTION
# Improved Logistic Regression Model

# 1. Load dataset
heart <- read.csv(
  "data/heart_disease.csv",
  na.strings = c("", "?", "NA")
)

stopifnot("num" %in% names(heart))
stopifnot(!anyNA(heart$num))

# Binary target
heart$num <- as.integer(heart$num > 0)

# 2. Define feature types
categorical <- c(
  "sex", "cp", "fbs", "restecg",
  "exang", "slope", "thal"
)

numeric_features <- c(
  "age", "threstbps", "chol",
  "thalach", "oldpeak", "ca"
)

# 3. Stratified train-test split
set.seed(123)

train_index <- unlist(lapply(
  split(seq_len(nrow(heart)), heart$num),
  function(idx) sample(idx, floor(0.8 * length(idx)))
))

train <- heart[train_index, ]
test <- heart[-train_index, ]

cat("Training:", nrow(train), "records\n")
cat("Testing:", nrow(test), "records\n")

# 4. Impute numeric features using training medians
for (feature in numeric_features) {
  
  med <- median(train[[feature]], na.rm = TRUE)
  
  if (!is.finite(med)) {
    stop(paste("Invalid median:", feature))
  }
  
  train[[feature]][is.na(train[[feature]])] <- med
  test[[feature]][is.na(test[[feature]])] <- med
}

# 5. Impute categorical features using training modes
get_mode <- function(x) {
  observed <- x[!is.na(x)]
  
  if (length(observed) == 0) {
    stop("Cannot calculate mode: all values missing")
  }
  
  names(sort(table(observed), decreasing = TRUE))[1]
}

for (feature in categorical) {
  
  mode_value <- get_mode(train[[feature]])
  
  train[[feature]][is.na(train[[feature]])] <-
    as.numeric(mode_value)
  
  test[[feature]][is.na(test[[feature]])] <-
    as.numeric(mode_value)
  
  # Use consistent categories across training and test
  levels_all <- sort(unique(train[[feature]]))
  
  train[[feature]] <- factor(
    train[[feature]], levels = levels_all
  )
  
  if (!all(test[[feature]] %in% levels_all)) {
    stop(paste(
      "Unseen category in test data:", feature
    ))
  }
  
  test[[feature]] <- factor(
    test[[feature]], levels = levels_all
  )
}

# 6. Scale numeric features using training statistics
train_means <- sapply(train[numeric_features], mean)
train_sds <- sapply(train[numeric_features], sd)

train_sds[is.na(train_sds) | train_sds == 0] <- 1

train[numeric_features] <- as.data.frame(
  scale(
    train[numeric_features],
    center = train_means,
    scale = train_sds
  )
)

test[numeric_features] <- as.data.frame(
  scale(
    test[numeric_features],
    center = train_means,
    scale = train_sds
  )
)

# 7. Train Logistic Regression model
model <- glm(
  num ~ .,
  data = train,
  family = binomial()
)

print(summary(model))

# 8. Predict on test data
probabilities <- predict(
  model, newdata = test, type = "response"
)

predictions <- ifelse(probabilities >= 0.5, 1L, 0L)

# 9. Confusion matrix
confusion <- table(
  Predicted = factor(predictions, levels = c(0, 1)),
  Actual = factor(test$num, levels = c(0, 1))
)

print(confusion)

# 10. Calculate metrics
TP <- confusion["1", "1"]
TN <- confusion["0", "0"]
FP <- confusion["1", "0"]
FN <- confusion["0", "1"]

safe_div <- function(a, b) {
  if (b == 0) return(NA_real_)
  as.numeric(a / b)
}

accuracy <- (TP + TN) / sum(confusion)
precision <- safe_div(TP, TP + FP)
recall <- safe_div(TP, TP + FN)
specificity <- safe_div(TN, TN + FP)

f1 <- if (is.na(precision) || is.na(recall)) {
  NA_real_
} else {
  safe_div(2 * precision * recall, precision + recall)
}

results <- data.frame(
  Metric = c(
    "Accuracy", "Precision",
    "Recall", "Specificity", "F1-Score"
  ),
  Score = round(
    c(accuracy, precision, recall, specificity, f1),
    4
  )
)

print(results)

# 11. Save results
dir.create("outputs", showWarnings = FALSE)

write.csv(
  results,
  "outputs/model_metrics.csv",
  row.names = FALSE
)

write.csv(
  as.data.frame.matrix(confusion),
  "outputs/confusion_matrix.csv"
)

cat("\nImproved model completed successfully!\n")





# Show final test results
print(confusion)
print(results)

# Confirm model diagnostics
print(model$converged)
print(sum(is.na(coef(model))))

# Verify categorical feature encoding
print(sapply(
  train[c("sex", "cp", "fbs", "restecg",
          "exang", "slope", "thal")],
  class
))
