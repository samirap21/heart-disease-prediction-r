
# HEART DISEASE PREDICTION
# 5-Fold Stratified Cross-Validation

# Load original dataset
heart <- read.csv(
  "data/heart_disease.csv",
  na.strings = c("", "?", "NA")
)

stopifnot("num" %in% names(heart))
stopifnot(all(!is.na(heart$num)))

heart$num <- as.integer(heart$num > 0)

features <- setdiff(names(heart), "num")

heart[features] <- lapply(
  heart[features], as.numeric
)

# Recreate the original train-test split
set.seed(123)
train_index <- unlist(lapply(
  split(seq_len(nrow(heart)), heart$num),
  function(idx) sample(idx, floor(0.8 * length(idx)))
))

train <- heart[train_index, ]

# Create stratified folds within training data
set.seed(456)

folds <- integer(nrow(train))

for (class in unique(train$num)) {
  ids <- which(train$num == class)
  folds[ids] <- sample(rep(1:5, length.out = length(ids)))
}

cv_results <- data.frame()

for (fold in 1:5) {
  
  fold_train <- train[folds != fold, ]
  fold_val <- train[folds == fold, ]
  
  # Impute using fold-training medians only
  for (feature in features) {
    med <- median(fold_train[[feature]], na.rm = TRUE)
    
    if (!is.finite(med)) {
      stop(paste("Invalid median:", feature))
    }
    
    fold_train[[feature]][is.na(fold_train[[feature]])] <- med
    fold_val[[feature]][is.na(fold_val[[feature]])] <- med
  }
  
  # Scale using fold-training statistics only
  means <- sapply(fold_train[features], mean)
  sds <- sapply(fold_train[features], sd)
  sds[is.na(sds) | sds == 0] <- 1
  
  fold_train[features] <- as.data.frame(
    scale(fold_train[features], center = means, scale = sds)
  )
  
  fold_val[features] <- as.data.frame(
    scale(fold_val[features], center = means, scale = sds)
  )
  
  # Train model
  model <- glm(
    num ~ .,
    data = fold_train,
    family = binomial()
  )
  
  # Predict
  prob <- predict(model, newdata = fold_val, type = "response")
  pred <- ifelse(prob >= 0.5, 1L, 0L)
  
  actual <- fold_val$num
  
  TP <- sum(pred == 1 & actual == 1)
  TN <- sum(pred == 0 & actual == 0)
  FP <- sum(pred == 1 & actual == 0)
  FN <- sum(pred == 0 & actual == 1)
  
  safe_div <- function(a, b) {
    if (b == 0) return(NA_real_)
    a / b
  }
  
  accuracy <- (TP + TN) / length(actual)
  precision <- safe_div(TP, TP + FP)
  recall <- safe_div(TP, TP + FN)
  
  f1 <- safe_div(
    2 * TP,
    2 * TP + FP + FN
  )
  
  cv_results <- rbind(
    cv_results,
    data.frame(
      Fold = fold,
      Accuracy = accuracy,
      Precision = precision,
      Recall = recall,
      F1 = f1
    )
  )
}

print(round(cv_results, 4))

cat("\nMean cross-validation performance:\n")

cv_summary <- data.frame(
  Metric = names(cv_results)[-1],
  Mean = colMeans(cv_results[-1], na.rm = TRUE),
  SD = sapply(cv_results[-1], sd, na.rm = TRUE)
)

print(cv_summary)

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

cat("\nCross-validation completed successfully!\n")
