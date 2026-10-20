
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

