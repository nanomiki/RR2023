# Credit Risk Analysis
# dataset: https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset?resource=download
# Adam Łaziński 418193
# Mikołaj Marszałek 457902

library(tidyverse) # data manipulation and visualization
library(dplyr)
library(ROSE) # for oversampling using SMOTE

# Load dataset
path <- "...loan.csv"
data <- read.csv(path, stringsAsFactors = FALSE)
head(data)

# Preparation of data for analysis

# 1. Default flag investigation and synthesis
table(data$loan_status)

# Transforming 'loan_status' to 'default_flag' - dependent variable.
# A client who's status is default, charged off or late (be it 16 or 120 days) is considered a bad client. We're interested not only in clients defaulting but also those that are going to be late, since it's a sign a client may default, which means they are also worse than clients paying on time.

data$default_flag <- as.integer(data$loan_status %in% c('Charged Off', 'Late (31-120 days)', 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Charged Off '))
head(data)

# 2. Initial variables investigation
str(data)

# Investigate 'grade'
table(data$grade)
summary(data)

# 3. Initial column elimination
# Drop columns that have less than circa 30% of non-missing values, in that case 250k
data <- data[, colSums(!is.na(data)) >= 250000]
head(data)

# Check column information
str(data)

# Let's check some columns that we suspect might have really concentrated values. We can then delete them to save space and computation time.
table(data$pymnt_plan)
table(data$application_type)
table(data$acc_now_delinq)
table(data$policy_code)
table(data$collections_12_mths_ex_med)
table(data$mths_since_last_delinq)
hist(data$mths_since_last_delinq, breaks = 150)
table(data$initial_list_status)
table(data$url)
table(data$next_pymnt_d)

# We are deleting columns with very concentrated values.
data <- data[, !(names(data) %in% c("application_type", "acc_now_delinq", "policy_code", "collections_12_mths_ex_med", "url", "pymnt_plan", "next_pymnt_d"))]

# We are investigating fields such as 'title' and 'emp_title' (employer title). These variables don't make much sense since each value is taken by 1-2 observations, thus we delete them.
table(data$emp_title)

# Looks like a good variable but we have to merge smaller counts into one
emp_title_counts <- table(data$emp_title)
emp_title_counts_grouped <- aggregate(emp_title_counts, by = list(ifelse(emp_title_counts >= 3000, names(emp_title_counts), "other")), FUN = sum)
head(emp_title_counts_grouped)

# After merge it doesn't look that good anymore, too fragmented. Maybe if we were modeling via Weight of Evidence and segmented it even further, for example, group together only those that occur less than 50 times and then group similar default frequencies.
data <- data[, !(names(data) %in% c("emp_title"))]

table(data$title)
table(data$zip_code)

# 'title' and 'zip_code' don't make much sense as variables, thus we delete them.
data <- data[, !(names(data) %in% c("title", "zip_code"))]

# 'loan_amnt', 'funded_amnt', and 'funded_amnt_inv' seem very similar in terms of values, most likely two of them will be thrown out at later stages.
hist(data$loan_amnt - data$funded_amnt)

# 4. Date variables
# We are transforming date variables. Instead of a date, they become the number of days from that time until now.
table(data$earliest_cr_line)
table(data$last_credit_pull_d)
table(data$issue_d)

data$cr_history_length <- as.numeric(as.Date("2023-01-01") - as.Date(data$earliest_cr_line))
data$since_last_cr_pull <- as.numeric(as.Date("2023-01-01") - as.Date(data$last_credit_pull_d))
data$since_issue <- as.numeric(as.Date("2023-01-01") - as.Date(data$issue_d))
data$since_last_pnmt <- as.numeric(as.Date("2023-01-01") - as.Date(data$last_pymnt_d))

# We can now drop original date variables.
data <- data[, !(names(data) %in% c("earliest_cr_line", "last_credit_pull_d", "issue_d", "last_pymnt_d"))]
head(data)

# 5. Transform some object columns into numbers
# 'term' variable consists only of '36 months' or '60 months', we can simply transform that into numbers.
data$term <- as.numeric(substr(data$term, 1, 2))

# We are transforming employment length to a number.
table(data$emp_length)

data$emp_length_num <- NA
data$emp_length_num[data$emp_length == "< 1 year"] <- 0
data$emp_length_num[data$emp_length == "1 year"] <- 1
data$emp_length_num[data$emp_length == "2 years"] <- 2
data$emp_length_num[data$emp_length == "3 years"] <- 3
data$emp_length_num[data$emp_length == "4 years"] <- 4
data$emp_length_num[data$emp_length == "5 years"] <- 5
data$emp_length_num[data$emp_length == "6 years"] <- 6
data$emp_length_num[data$emp_length == "7 years"] <- 7
data$emp_length_num[data$emp_length == "8 years"] <- 8
data$emp_length_num[data$emp_length == "9 years"] <- 9
data$emp_length_num[data$emp_length == "10+ years"] <- 10

data <- data[, !(names(data) %in% c("emp_length"))]

# 6. Drop the original target column and take one more look at the data
# We can drop 'loan_status', as we have transformed it into a default flag before.
data <- data[, !(names(data) %in% c("loan_status"))]

summary(data)

hist(data$recoveries, breaks = 100)
hist(data$collection_recovery_fee, breaks = 100)
hist(data$total_rec_late_fee, breaks = 100)

# As we can see the 3 variables mentioned above: 'recoveries', 'collection_recovery_fee', and 'total_rec_late_fee' have a very high concentration, almost all of the values are 0, thus we can delete them.
data <- data[, !(names(data) %in% c("total_rec_late_fee", "recoveries", "collection_recovery_fee"))]

str(data)

# 7. ETL process complete
# During the process, there were no data artificially added to the set.
# We have finished the ETL process, we are saving the updated dataframe as a new csv.
write.csv(data, file = "...loan_etl_full.csv", row.names = FALSE)

# Feature transformation
# 1. Correlation analysis
# We are loading the prepared data.
data_etl <- read.csv("/content/drive/MyDrive/ML_project_credit_risk/loan_etl_full.csv", stringsAsFactors = FALSE)

head(data_etl)

str(data_etl)

# Find correlation with the target and sort
correlations <- sort(cor(data_etl$default_flag, data_etl[-ncol(data_etl)]), decreasing = TRUE)

# Display correlations
cat("Most Positive Correlations:\n")
tail(correlations, 13)
cat("\nMost Negative Correlations:\n")
head(correlations, 10)

cor(data_etl[c("default_flag", "total_pymnt", "total_pymnt_inv")])

cor(data_etl[c("default_flag", "out_prncp", "out_prncp_inv")])

cor(data_etl[c("default_flag", "loan_amnt", "funded_amnt", "funded_amnt_inv")])

# As mentioned earlier, 'loan_amnt', 'funded_amnt', and 'funded_amnt_inv' are very similar, with a correlation over 0.98. Thus, we pick only the one with the highest correlation with our target variable ('default_flag'), which is 'loan_amnt', and drop the rest. Same with 'total_pymnt', 'total_pymnt_inv', and 'out_prncp'.

data_etl <- subset(data_etl, select = -c(funded_amnt, funded_amnt_inv, total_pymnt, out_prncp))

head(data_etl)

# 2. Object type vars
# We already have quite some number of variables fairly correlated with our default flag. Now investigate a couple of promising object type variables. Since we already transformed the majority of variables into numeric types, we're left with only 6 object type vars: grade, sub_grade, home_ownership, verification_status, purpose, and state. Since grade can be thought of as ordinal (A is best, etc.), we can transform it into numerical vars.

# Grade - looks promising
par(mfrow = c(1, 2))
pie(table(data_etl$grade[data_etl$default_flag == 1]))
pie(table(data_etl$grade[data_etl$default_flag == 0]))

# Sub grade
par(mfrow = c(1, 2))
pie(table(data_etl$sub_grade[data_etl$default_flag == 1]))
pie(table(data_etl$sub_grade[data_etl$default_flag == 0]))

# Home Ownership
# Does look promising - visible difference, but it will be convenient to merge the smallest counts into one category
par(mfrow = c(1, 2))
pie(table(data_etl$home_ownership[data_etl$default_flag == 1]))
pie(table(data_etl$home_ownership[data_etl$default_flag == 0]))

s <- table(data_etl$home_ownership)
merged <- aggregate(s, by = list(ifelse(s >= 3000, names(s), "other")), sum)
merged

data_etl$home_own_merged <- ifelse(data_etl$home_ownership %in% c("MORTGAGE", "RENT", "OWN"), data_etl$home_ownership, "OTHER")
data_etl <- subset(data_etl, select = -home_ownership)

# Verification Status
par(mfrow = c(1, 2))
pie(table(data_etl$verification_status[data_etl$default_flag == 1]))
pie(table(data_etl$verification_status[data_etl$default_flag == 0]))

# Purpose
par(mfrow = c(1, 2))
pie(table(data_etl$purpose[data_etl$default_flag == 1]))
pie(table(data_etl$purpose[data_etl$default_flag == 0]))

# Address State
par(mfrow = c(1, 2))
pie(table(data_etl$addr_state[data_etl$default_flag == 1]))
pie(table(data_etl$addr_state[data_etl$default_flag == 0]))

# Initial List Status
par(mfrow = c(1, 2))
pie(table(data_etl$initial_list_status[data_etl$default_flag == 1]))
pie(table(data_etl$initial_list_status[data_etl$default_flag == 0]))

# Of the variables plotted above, there are a couple of promising variables: 'home_ownership', 'initial_list_status', 'grade'

# 3. Feature final selection and encoding
# We select numerical variables correlated with default flag above 5%: 'home_ownership', 'initial_list_status', and 'grade'.

# Find correlation with the target and sort
correlations <- cor(data_etl)[, "default_flag"]
correlations <- sort(correlations)

# Display correlations
print("Most Positive Correlations:")
print(correlations[length(correlations) - 12:length(correlations)])
print("\nMost Negative Correlations:")
print(correlations[1:10])

data_selected <- data_etl[c("default_flag", "since_last_pnmt", "since_issue", "int_rate", "since_last_cr_pull", "total_rec_int", "out_prncp_inv", "last_pymnt_amnt", "total_rec_prncp", "home_own_merged", "initial_list_status", "grade", "verification_status")]

# At the very beginning, 'grade' variable is ordinal so we can transform it into numbers:
data_selected$grade_num <- as.integer(factor(data_selected$grade, levels = unique(data_selected$grade)))

# Correlation heatmap
corr <- cor(data_selected)
heatmap(corr, annot = TRUE, vmin = -1.0, cmap = "mako", main = "Correlation Heatmap")

# We're left with 3 object variables that have to be one hot encoded: 'home ownership', 'verification_status', and 'initial_list_status'.
h_ownership_dummies <- as.data.frame(model.matrix(~home_own_merged - 1, data = data_selected))
initial_status_dummies <- as.data.frame(model.matrix(~initial_list_status - 1, data = data_selected))
verification_dummies <- as.data.frame(model.matrix(~verification_status - 1, data = data_selected))
data_selected <- cbind(data_selected, h_ownership_dummies, initial_status_dummies, verification_dummies)
data_selected <- data_selected[, !grepl("home_own_merged|initial_list_status|verification_status", names(data_selected))]

# Check for Nulls in the final dataset
sum(is.na(data_selected))

# The smallest number of non-nulls is in 'since_last_pnmt' which is still around 98%, so we can drop all the rows containing N/A
data_selected <- na.omit(data_selected)

# Split, divide, and balance the set
y <- data_selected[, 1]
x <- data_selected[, -1]

# Using SMOTE to handle imbalanced data
smote_obj <- SMOTE(y ~ ., data = data_selected)
data_balanced <- data.frame(smote_obj)

# Scaling Data using Standard Scaler
sc <- scale(x_smote)

# Split Train and Test
set.seed(10)
split <- sample.split(y_smote, SplitRatio = 0.75)
x_train <- sc[split, ]
x_test <- sc[!split, ]
y_train <- y_smote[split]
y_test <- y_smote[!split]

# As a comparison, we're gonna test the model on a natural, imbalanced dataset.
sc_un <- scale(x)
set.seed(10)
split_un <- sample.split(y, SplitRatio = 0.75)
x_train_un <- sc_un[split_un, ]
x_test_un <- sc_un[!split_un, ]
y_train_un <- y[split_un]
y_test_un <- y[!split_un]

# Modelling
# 1. Logistic Regression

# Train
model_log <- glm(y_train ~ ., family = binomial(), data = data.frame(x_train, y_train))
summary(model_log)

# Predict
probabilities <- predict(model_log, newdata = data.frame(x_test), type = "response")
y_pred <- ifelse(probabilities >= 0.5, 1, 0)

# Metrics
log_accuracy <- sum(y_pred == y_test) / length(y_test)
log_recall <- sum(y_pred[y_test == 1] == 1) / sum(y_test == 1)
log_precision <- sum(y_pred[y_test == 1] == 1) / sum(y_pred == 1)
log_rocauc <- roc(y_pred, y_test)$auc

cat(format(log_accuracy, digits = 4), "- Log Accuracy\n")
cat(format(log_recall, digits = 4), "- Log Recall\n")
cat(format(log_precision, digits = 4), "- Log Precision\n")
cat(format(log_rocauc, digits = 4), "- Log ROC AUC\n")

# Test on unbalanced dataset
probabilities_un <- predict(model_log, newdata = data.frame(x_test_un), type = "response")
y_pred_un <- ifelse(probabilities_un >= 0.5, 1, 0)

log_accuracy_un <- sum(y_pred_un == y_test_un) / length(y_test_un)
log_recall_un <- sum(y_pred_un[y_test_un == 1] == 1) / sum(y_test_un == 1)
log_precision_un <- sum(y_pred_un[y_test_un == 1] == 1) / sum(y_pred_un == 1)
log_rocauc_un <- roc(y_pred_un, y_test_un)$auc

cat(format(log_accuracy_un, digits = 4), "- Log Accuracy\n")
cat(format(log_recall_un, digits = 4), "- Log Recall\n")
cat(format(log_precision_un, digits = 4), "- Log Precision\n")
cat(format(log_rocauc_un, digits = 4), "- Log ROC AUC\n")

# Predictions
probabilities <- predict(model_log, newdata = data.frame(x_test), type = "response")
df_prediction_prob <- data.frame(prob_0 = 1 - probabilities, prob_1 = probabilities)
df_prediction_target <- data.frame(predicted_TARGET = ifelse(probabilities >= 0.5, 1, 0))
df_test_dataset <- data.frame(Actual_Outcome = y_test)

df <- cbind(df_test_dataset, df_prediction_prob, df_prediction_target)
df <- df[order(df$prob_0, decreasing = TRUE), ]
df

# Confusion Matrix
confusion_matrix_logit <- table(Actual_Outcome = y_test, Predicted_TARGET = y_pred)
confusion_matrix_logit

# KNN
n <- 2
knn <- knn.train(x_train, y_train, k = n)
y_pred <- knn.predict(knn, x_test)

# KNN on sample data
set.seed(10)
sample <- data_selected[sample(nrow(data_selected), 20000), ]
y_knn_sample <- sample[, 1]
x_knn_sample <- sample[, -1]

x_smote_knn_sample <- smote(x_knn_sample, y_knn_sample)$data
x_smote_knn_sample <- scale(x_smote_knn_sample)

set.seed(10)
split_knn_sample <- sample.split(y_smote_knn_sample, SplitRatio = 0.75)
x_train_knn_sample <- x_smote_knn_sample[split_knn_sample, ]
x_test_knn_sample <- x_smote_knn_sample[!split_knn_sample, ]
y_train_knn_sample <- y_smote_knn_sample[split_knn_sample]
y_test_knn_sample <- y_smote_knn_sample[!split_knn_sample]

n <- 2
knn_sample <- knn.train(x_train_knn_sample, y_train_knn_sample, k = n)
y_pred_knn_sample <- knn.predict(knn_sample, x_test_knn_sample)

# Metrics
log_accuracy <- sum(y_pred_knn_sample == y_test_knn_sample) / length(y_test_knn_sample)
log_recall <- sum(y_pred_knn_sample[y_test_knn_sample == 1] == 1) / sum(y_test_knn_sample == 1)
log_precision <- sum(y_pred_knn_sample[y_test_knn_sample == 1] == 1) / sum(y_pred_knn_sample == 1)
log_rocauc <- roc(y_pred_knn_sample, y_test_knn_sample)$auc

cat(format(log_accuracy, digits = 4), "- Log Accuracy\n")
cat(format(log_recall, digits = 4), "- Log Recall\n")
cat(format(log_precision, digits = 4), "- Log Precision\n")
cat(format(log_rocauc, digits = 4), "- Log ROC AUC\n")

# Confusion matrix
confusion_matrix <- table(y_test, y_pred)
print(confusion_matrix)

# 3. SVM
svm <- svm(x_train, y_train, kernel = "linear")
y_pred <- predict(svm, x_test)

# Metrics
log_accuracy <- sum(y_test == y_pred) / length(y_test)
log_recall <- sum(y_test == 1 & y_pred == 1) / sum(y_test == 1)
log_precision <- sum(y_test == 1 & y_pred == 1) / sum(y_pred == 1)
log_rocauc <- roc(y_pred, y_test)$auc

cat(sprintf("%.4f", log_accuracy), "- Log Accuracy\n")
cat(sprintf("%.4f", log_recall), "- Log Recall\n")
cat(sprintf("%.4f", log_precision), "- Log Precision\n")
cat(sprintf("%.4f", log_rocauc), "- Log ROC AUC\n")

# Confusion matrix
table(y_test, y_pred)

# Metrics look good, but the message 'liblinear failed to converge' suggests that the algorithm does not work well, even though our training data has been standardized.

# 4. Decision trees
library(rpart)

dtree <- rpart(y_train ~ ., data = cbind(x_train, y_train),
               method = "class", control = rpart.control(minsplit = 5, minbucket = 5, maxdepth = 5))
y_pred <- predict(dtree, newdata = x_test, type = "class")

log_accuracy <- sum(y_test == y_pred) / length(y_test)
log_recall <- sum(y_test == 1 & y_pred == 1) / sum(y_test == 1)
log_precision <- sum(y_test == 1 & y_pred == 1) / sum(y_pred == 1)
log_rocauc <- roc(y_pred, y_test)$auc

cat(sprintf("%.4f", log_accuracy), "- Log Accuracy\n")
cat(sprintf("%.4f", log_recall), "- Log Recall\n")
cat(sprintf("%.4f", log_precision), "- Log Precision\n")
cat(sprintf("%.4f", log_rocauc), "- Log ROC AUC\n")

table(y_test, y_pred)

# Seems that decision trees have the highest metrics of all - 96% ROC AUC

# 5. Visualization
library(ROCR)

logit_roc_auc <- roc.area(prediction(response = y_pred, predictor = y_test))
tree_roc_auc <- roc.area(prediction(response = y_pred, predictor = y_test))

fpr <- performance(prediction(response = y_pred, predictor = y_test), "fpr")@x.values[[1]]
tpr <- performance(prediction(response = y_pred, predictor = y_test), "tpr")@y.values[[1]]
fpr1 <- performance(prediction(response = y_pred, predictor = y_test), "fpr")@x.values[[1]]
tpr1 <- performance(prediction(response = y_pred, predictor = y_test), "tpr")@y.values[[1]]

plot(fpr, tpr, type = "l", col = "blue", lwd = 2, xlim = c(0, 1), ylim = c(0, 1),
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     main = "Receiver operating characteristic")
lines(fpr1, tpr1, type = "l", col = "red", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "black")
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"),
       col = c("blue", "red"), lty = 1, lwd = 2)
