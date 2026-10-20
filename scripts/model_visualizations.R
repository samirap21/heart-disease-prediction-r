
# HEART DISEASE PREDICTION
# Improved 5-Fold Stratified Cross-Validation

# 1. Load dataset
heart <- read.csv(
  "data/heart_disease.csv",
  na.strings = c("", "?", "NA")
)

stopifnot("num" %in% names(heart))
stopifnot(!anyNA(heart$num))

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

# 3. Recreate original 80/20 split
set.seed(123)

train_index <- unlist(lapply(
  split(seq_len(nrow(heart)), heart$num),
  function(idx) sample(idx, floor(0.8 * length(idx)))
))

train <- heart[train_index, ]

# 4. Create five stratified folds
set.seed(456)

folds <- integer(nrow(train))

for (class in unique(train$num)) {
  ids <- which(train$num == class)
  folds[ids] <- sample(
    rep(1:5, length.out = length(ids))
  )
}

# Helper function for categorical imputation
get_mode <- function(x) {
  observed <- x[!is.na(x)]
  
  if (length(observed) == 0) {
    stop("Cannot calculate mode")
  }
  
  names(sort(table(observed), decreasing = TRUE))[1]
}

safe_divide <- function(a, b) {
  if (b == 0) return(NA_real_)
  as.numeric(a / b)
}

# 5. Train and validate on each fold
cv_results <- data.frame()

for (fold in 1:5) {
  
  fold_train <- train[folds != fold, ]
  fold_val <- train[folds == fold, ]
  
  # Impute numeric predictors using fold training data
  for (feature in numeric_features) {
    med <- median(
      fold_train[[feature]], na.rm = TRUE
    )
    
    if (!is.finite(med)) {
      stop(paste("Invalid median:", feature))
    }
    
    fold_train[[feature]][is.na(fold_train[[feature]])] <- med
    fold_val[[feature]][is.na(fold_val[[feature]])] <- med
  }
  
  # Impute and encode categorical predictors
  for (feature in categorical) {
    mode_value <- as.numeric(
      get_mode(fold_train[[feature]])
    )
    
    fold_train[[feature]][is.na(fold_train[[feature]])] <-
      mode_value
    
    fold_val[[feature]][is.na(fold_val[[feature]])] <-
      mode_value
    
    levels_train <- sort(
      unique(fold_train[[feature]])
    )
    
    if (!all(fold_val[[feature]] %in% levels_train)) {
      stop(paste(
        "Unseen validation category in fold",
        fold, ":", feature
      ))
    }
    
    fold_train[[feature]] <- factor(
      fold_train[[feature]],
      levels = levels_train
    )
    
    fold_val[[feature]] <- factor(
      fold_val[[feature]],
      levels = levels_train
    )
  }
  
  # Scale numeric predictors using fold training data
  means <- sapply(
    fold_train[numeric_features], mean
  )
  
  sds <- sapply(
    fold_train[numeric_features], sd
  )
  
  sds[is.na(sds) | sds == 0] <- 1
  
  fold_train[numeric_features] <- as.data.frame(
    scale(
      fold_train[numeric_features],
      center = means,
      scale = sds
    )
  )
  
  fold_val[numeric_features] <- as.data.frame(
    scale(
      fold_val[numeric_features],
      center = means,
      scale = sds
    )
  )
  
  # Train Logistic Regression
  model <- glm(
    num ~ .,
    data = fold_train,
    family = binomial()
  )
  
  # Predict validation data
  probabilities <- predict(
    model,
    newdata = fold_val,
    type = "response"
  )
  
  if (anyNA(probabilities)) {
    stop(paste(
      "Missing predictions in fold", fold
    ))
  }
  
  predicted <- ifelse(
    probabilities >= 0.5, 1L, 0L
  )
  
  actual <- fold_val$num
  
  # Confusion matrix counts
  TP <- sum(predicted == 1 & actual == 1)
  TN <- sum(predicted == 0 & actual == 0)
  FP <- sum(predicted == 1 & actual == 0)
  FN <- sum(predicted == 0 & actual == 1)
  
  # Evaluation metrics
  accuracy <- (TP + TN) / length(actual)
  precision <- safe_divide(TP, TP + FP)
  recall <- safe_divide(TP, TP + FN)
  specificity <- safe_divide(TN, TN + FP)
  f1 <- safe_divide(2 * TP, 2 * TP + FP + FN)
  
  cv_results <- rbind(
    cv_results,
    data.frame(
      Fold = fold,
      Accuracy = accuracy,
      Precision = precision,
      Recall = recall,
      Specificity = specificity,
      F1 = f1
    )
  )
  
  cat("Completed fold:", fold, "\n")
}

# 6. Display results
print(round(cv_results, 4))

cv_summary <- data.frame(
  Metric = names(cv_results)[-1],
  Mean = colMeans(
    cv_results[-1], na.rm = TRUE
  ),
  SD = sapply(
    cv_results[-1], sd, na.rm = TRUE
  )
)

cat("\nMean cross-validation performance:\n")
print(round(cv_summary[, c("Mean", "SD")], 4))

# 7. Save results
dir.create("outputs", showWarnings = FALSE)

write.csv(
  cv_results,
  "outputs/cross_validation_results.csv",
  row.names = FALSE
)

write.csv(
  cv_summary,
  "outputs/cross_validation_summary.csv",
  row.names = FALSE
)

cat("\nImproved cross-validation completed successfully!\n")
