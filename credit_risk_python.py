#Credit Risk Analysis
#dataset: https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset?resource=download
#Adam Łaziński 418193
#Mikołaj Marszałek 457902

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC

#set display max column to see all column
pd.set_option("display.max_columns", None)
pd.get_option("display.max_columns")

#set display max column to see all column
pd.set_option("display.max_rows", 150)
pd.get_option("display.max_rows")

#Load dataset

path = "...loan.csv"
data = pd.read_csv(path, engine='python')
data.head()

#Preparation of data for analysis

#1. Default flag investigation and synthesis
data['loan_status'].value_counts()

#Transforming 'loan_status' to 'default_flag' - dependent variable. A client who's status is default, charged off or late (be it 16 or 120 days) is cosindered a bad client. We're interested not only in clients defaulting but also those that are going to be late, since it's a sign a client may default, which means they are also worse than clients paying on time.

data['default_flag']= data.apply(lambda row: int(row['loan_status'] in ['Charged Off','Late (31-120 days)', 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Charged Off ']), axis=1)
data.head()

#2. Initial variables investigation|
print(data.dtypes)

#Investigate 'grade'
data['grade'].value_counts()
data.describe()

#3. Initial column elimination
#Drop columns that have less than circa 30% of non missing values, in that case 250k

data.dropna('columns',thresh=250000, inplace=True)
data.head()

data.info()

#Let's check some columns that we suspect might have really concentraded values. We can then delete them to save space and computation time.
data['pymnt_plan'].value_counts()

data['application_type'].value_counts()

data['acc_now_delinq'].value_counts()

data['policy_code'].value_counts()

data['collections_12_mths_ex_med'].value_counts()

data['pymnt_plan'].value_counts()

data['mths_since_last_delinq'].value_counts()

data['mths_since_last_delinq'].hist(bins=150)

data['initial_list_status'].value_counts()

data['url'].value_counts()

data['next_pymnt_d'].value_counts()

#We are deleting columns with very concentrated values.
del data['application_type']
del data['acc_now_delinq']
del data['policy_code']
del data['collections_12_mths_ex_med']
del data['url']
del data['pymnt_plan']
del data['next_pymnt_d']

#We are investigating fields such as 'title' and 'emp_title' (employer title), these variables don't make much sense since each value is taken by 1-2 observations, thus we delete them.

data['emp_title'].value_counts()

#Looks like a good variable but we have to merge smaller counts into one so that the number of groups is not that large

s=data['emp_title'].value_counts()
s.groupby(np.where(s>=3000,s.index,'other')).sum()#.plot.pie()

#After merge it doesn't look that good anymore, too fragmented. Maybe if we were modelling via Weight of Evidence and segmented it even further, for example group together only those that occur less than 50 times and then group similiar default frequencies...

del data['emp_title']

data['title'].value_counts()

data['zip_code'].value_counts()

#'title' and 'zip_code' don't make much sense as variables, thus we delete them.

del data['zip_code']
del data['title']

#'loan_amnt'; 'funded_amnt' and 'funded_amnt_inv' seem very similar in terms of values, most likely two of them will be thrown out at later stages.

(data['loan_amnt']-data['funded_amnt']).hist()

#4. Date variables
#We are transforming date variables. Instead of a date they become number of days from that time until now.

data['earliest_cr_line'].value_counts()

data['last_credit_pull_d'].value_counts()

data['issue_d'].value_counts()

data['cr_history_length']=(pd.to_datetime('Jan-2023') - pd.to_datetime(data['earliest_cr_line'])).dt.days
data['since_last_cr_pull']=(pd.to_datetime('Jan-2023') - pd.to_datetime(data['last_credit_pull_d'])).dt.days
data['since_issue']=(pd.to_datetime('Jan-2023') - pd.to_datetime(data['issue_d'])).dt.days
data['since_last_pnmt']=(pd.to_datetime('Jan-2023') - pd.to_datetime(data['last_pymnt_d'])).dt.days

#We can now drop original date variables.

del data['earliest_cr_line']
del data['last_credit_pull_d']
del data['issue_d']
del data['last_pymnt_d']

data.head()

#5. Transform some object columns into numbers
#'term' variable consists only of '36 months' or '60 months', we can simply transform that into numbers.

data['term']=pd.to_numeric(data['term'].str[:3])

#We are transforming employment length to number.

data['emp_length'].value_counts()

data['emp_length_num']=data['emp_length'].map({'< 1 year':0,'1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5, '6 years':6, '7 years': 7, '8 years': 8, '9 years':9, '10+ years':10})

del data['emp_length']

#6. Drop the original target column and take one more look at the data
#We can drop 'loan_status', as we have transformed it into a default flag before.

del data['loan_status']

data.describe()

data['recoveries'].hist(bins=100)

data['collection_recovery_fee'].hist(bins=100)

data['total_rec_late_fee'].hist(bins=100)

#As we can see the 3 variables mentioned above: 'recoveries'; 'collection_recovery_fee' and 'total_rec_late_fee' have a very high concentration, almost all of the values are 0, thus we can delete them.

del data['total_rec_late_fee']
del data['recoveries']
del data['collection_recovery_fee']

data.info()

#7. ETL process complete
#During the process there were no data artficially added to the set
#We have finished ETL process, we are saving updated dataframe as new csv.

data.to_csv("...loan_etl_full.csv", index=False)

#Feature transformation
#1.Correlation analysis
#We are loading the prepared data.

path = "/content/drive/MyDrive/ML_project_credit_risk/loan_etl_full.csv"
data_etl = pd.read_csv(path, engine='python')
#data_etl=data #if proceeding in one go

data_etl.head()

data_etl.info()

# Find correlation with the target and sort
correlations = data_etl.corr()['default_flag'].sort_values()

# Display correlations
print('Most Postive Correlations:\n', correlations.tail(13))
print('\nMost Negative Correlations:\n', correlations.head(10))

data_etl.corr()

data_etl[['default_flag','total_pymnt','total_pymnt_inv']].corr()

data_etl[['default_flag','out_prncp','out_prncp_inv']].corr()

data_etl[['default_flag','loan_amnt','funded_amnt','funded_amnt_inv']].corr()

#As mentioned earlier, 'loan_amnt'; 'funded_amnt' and 'funded_amnt_inv' are very similar, correlation over 0,98 thus we pick only the one with the highest correlation with our target variable ('default_flag'), which is 'loan_amnt' and drop the rest. Same with 'total_pymnt'; 'total_pymnt_inv' and 'out_prncp'.

del data_etl['funded_amnt']
del data_etl['funded_amnt_inv']
del data_etl['total_pymnt']
del data_etl['out_prncp']

data_etl.head()

#2. Object type vars
#We already have quite some number of variables fairly correlated with our default flag. Now investigate a couple of promising object type variables. Since we already transformed majority of variables into numeric types we're left with only 6 object type vars: grade, sub grade, home ownership, verification status, purpose and state. Since grade can be thougt of as ordinal (A is best etc) we can transform it into numerical vars

#Grade - looks promising

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
data_etl.loc[data_etl["default_flag"]==1, "grade"].reset_index().groupby("grade").size().plot(kind="pie", ax=ax1)
data_etl.loc[data_etl["default_flag"]==0, "grade"].reset_index().groupby("grade").size().plot(kind="pie", ax=ax2)

ax1.set_xlabel("target 1")
ax2.set_xlabel("target 0")

#Sub grade

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
data_etl.loc[data_etl["default_flag"]==1, "sub_grade"].reset_index().groupby("sub_grade").size().plot(kind="pie", ax=ax1)
data_etl.loc[data_etl["default_flag"]==0, "sub_grade"].reset_index().groupby("sub_grade").size().plot(kind="pie", ax=ax2)

ax1.set_xlabel("target 1")
ax2.set_xlabel("target 0")

#Home Ownership
#Does look promising- visible difference, but it will be convenient to merge the smallest counts into one category

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
data_etl.loc[data_etl["default_flag"]==1, "home_ownership"].reset_index().groupby("home_ownership").size().plot(kind="pie", ax=ax1)
data_etl.loc[data_etl["default_flag"]==0, "home_ownership"].reset_index().groupby("home_ownership").size().plot(kind="pie", ax=ax2)

ax1.set_xlabel("target 1")
ax2.set_xlabel("target 0")

s=data_etl['home_ownership'].value_counts()
merged=s.groupby(np.where(s>=3000,s.index,'other')).sum()

merged

data_etl['home_own_merged']=data_etl['home_ownership'].map({'MORTGAGE': 'MORTGAGE','RENT': 'RENT', 'OWN': 'OWN', 'OTHER': 'OTHER', 'NONE': 'OTHER', 'ANY': 'OTHER'})

del data_etl['home_ownership']

#Verification Status

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
data_etl.loc[data_etl["default_flag"]==1, "verification_status"].reset_index().groupby("verification_status").size().plot(kind="pie", ax=ax1)
data_etl.loc[data_etl["default_flag"]==0, "verification_status"].reset_index().groupby("verification_status").size().plot(kind="pie", ax=ax2)

ax1.set_xlabel("target 1")
ax2.set_xlabel("target 0")

#Purpose

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
data_etl.loc[data_etl["default_flag"]==1, "purpose"].reset_index().groupby("purpose").size().plot(kind="pie", ax=ax1)
data_etl.loc[data_etl["default_flag"]==0, "purpose"].reset_index().groupby("purpose").size().plot(kind="pie", ax=ax2)

ax1.set_xlabel("target 1")
ax2.set_xlabel("target 0")

#Address State

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
data_etl.loc[data_etl["default_flag"]==1, "addr_state"].reset_index().groupby("addr_state").size().plot(kind="pie", ax=ax1)
data_etl.loc[data_etl["default_flag"]==0, "addr_state"].reset_index().groupby("addr_state").size().plot(kind="pie", ax=ax2)

ax1.set_xlabel("target 1")
ax2.set_xlabel("target 0")

#Initial List Status

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
data_etl.loc[data_etl["default_flag"]==1, "initial_list_status"].reset_index().groupby("initial_list_status").size().plot(kind="pie", ax=ax1)
data_etl.loc[data_etl["default_flag"]==0, "initial_list_status"].reset_index().groupby("initial_list_status").size().plot(kind="pie", ax=ax2)

ax1.set_xlabel("target 1")
ax2.set_xlabel("target 0")

#Of the variables plotted above, there are a couple of promising variables:

#'home_ownership'
#'initial_list_status'
#'grade'

#3. Feature final selection and encoding
#We select numerical variables correlated with default flag above 5%, 'home_ownership'; 'intiial_list_status' and 'grade'.

# Find correlation with the target and sort
correlations = data_etl.corr()['default_flag'].sort_values()

# Display correlations
print('Most Postive Correlations:\n', correlations.tail(13))
print('\nMost Negative Correlations:\n', correlations.head(10))

data_selected=data_etl[['default_flag','since_last_pnmt','since_issue','int_rate','since_last_cr_pull','total_rec_int','out_prncp_inv','last_pymnt_amnt','total_rec_prncp','home_own_merged',
                       'initial_list_status','grade', 'verification_status']]

#At the very beginning, 'grade' variable is ordinal so we can transform it into numbers:

test_keys=data_selected['grade'].unique().tolist()
test_values=np.arange(7).tolist()
res = {test_keys[i]: test_values[i] for i in range(len(test_keys))}
data_selected['grade_num']=data_selected['grade'].map(res)
del data_selected['grade']

data_selected['grade_num'].value_counts()

#4.Correlation heatmap

corr = data_selected.corr()

plt.figure(figsize=(15,12))
sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
plt.title('Correlation Heatmap')
plt.show()

#We're left with 3 object variables that have to be one hot encoded: 'home ownership'; 'verification_status' and 'initial_list_status'.

h_ownership_dummies = pd.get_dummies(data_selected['home_own_merged'],prefix='home_own_merged')
initial_status_dummies = pd.get_dummies(data_selected['initial_list_status'],prefix='initial_list_status')
verification_dummies=pd.get_dummies(data_selected['verification_status'],prefix='verification_status')

data_selected = pd.concat([data_selected, h_ownership_dummies, initial_status_dummies, verification_dummies], axis=1)

data_selected=data_selected.drop(['home_own_merged','initial_list_status','verification_status'], axis=1)

#Now it's time to check for Nulls in the final dataset

data_selected.info()

#The smallest number of non-nulls is in 'since_last_pnmt' which is still around 98%, so we can drop all the rows containing N/A

data_selected.head()

data_selected.dropna(axis=0, how='any', inplace=True)

#5. Split, divide and balance the set

y = data_selected.iloc[:, 0].values
x = data_selected.iloc[:, 1:].values

#6. Using SMOTE to handle imbalance data

smote=SMOTE()

x_smote, y_smote = smote.fit_resample(x, y)

#7. Scaling Data using Standard Scaler and splitting

#Standardize features by removing the mean and scaling to unit variance.

#Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

sc = StandardScaler()
x_smote = sc.fit_transform(x_smote)

x_smote

x_smote.mean()

#8. Split Train and Test

x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size = .25, random_state = 10)

#As a comparison we're gonna test the model on a natural, imbalanced dataset.

x_1=sc.fit_transform(x)

x_train_un, x_test_un, y_train_un, y_test_un = train_test_split(x_1, y, test_size = .25, random_state = 10)

#Modelling
#1. Logistic Regression
#Most widely used in credit risk analysis, simple logistic regression model

#Train

classifier =  LogisticRegression(max_iter=3000)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

#Metrics

log_accuracy=accuracy_score(y_test, y_pred)
log_recall=recall_score(y_test,y_pred)
log_precision=precision_score(y_test,y_pred)
log_rocauc=roc_auc_score(y_test,y_pred)

print('{:.4f}'.format(log_accuracy), '- Log Accuracy')
print('{:.4f}'.format(log_recall), '- Log Recall')
print('{:.4f}'.format(log_precision), '- Log Precision')
print('{:.4f}'.format(log_rocauc), '- Log ROC AUC')

#Test on unbalanced dataset

y_pred_un=classifier.predict(x_test_un)

log_accuracy=accuracy_score(y_test_un, y_pred_un)
log_recall=recall_score(y_test_un,y_pred_un)
log_precision=precision_score(y_test_un,y_pred_un)
log_rocauc=roc_auc_score(y_test_un,y_pred_un)
print('{:.4f}'.format(log_accuracy), '- Log Accuracy')
print('{:.4f}'.format(log_recall), '- Log Recall')
print('{:.4f}'.format(log_precision), '- Log Precision')
print('{:.4f}'.format(log_rocauc), '- Log ROC AUC')

#Predictions

predictions = classifier.predict_proba(x_test)
predictions

df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(classifier.predict(x_test), columns = ['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test,columns= ['Actual Outcome'])

df=pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)
df.sort_values(by=['prob_0'],  ascending=[False],inplace=True)

df

#Confusion Matrix

confusion_matrix_logit = confusion_matrix(y_test, y_pred)
print(confusion_matrix_logit)
#pd.crosstab(y_test,y_pred)

#2. KNN

n = 2
knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

#KNN is too time-complex for such large dataset so we are performing it on the sample

data_knn_sample = data_selected.sample(n=20000)
y_knn_sample = data_knn_sample.iloc[:, 0].values
x_knn_sample = data_knn_sample.iloc[:, 1:].values

#Using SMOTE
smote=SMOTE()
x_smote_knn_sample, y_smote_knn_sample = smote.fit_resample(x_knn_sample, y_knn_sample)
sc = StandardScaler()
x_smote_knn_sample = sc.fit_transform(x_smote_knn_sample)

x_train_knn_sample, x_test_knn_sample, y_train_knn_sample, y_test_knn_sample = train_test_split(x_smote_knn_sample, y_smote_knn_sample, test_size = .25, random_state = 10)

#KNN on sample data
n = 2
knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(x_train_knn_sample, y_train_knn_sample)
y_pred = knn.predict(x_test_knn_sample)

#Metrics

log_accuracy=accuracy_score(y_test_knn_sample, y_pred)
log_recall=recall_score(y_test_knn_sample,y_pred)
log_precision=precision_score(y_test_knn_sample,y_pred)
log_rocauc=roc_auc_score(y_test_knn_sample,y_pred)

print('{:.4f}'.format(log_accuracy), '- Log Accuracy')
print('{:.4f}'.format(log_recall), '- Log Recall')
print('{:.4f}'.format(log_precision), '- Log Precision')
print('{:.4f}'.format(log_rocauc), '- Log ROC AUC')

#Confusion matrix

#confusion_matrix = confusion_matrix(y_test, y_pred)
#print(confusion_matrix)
pd.crosstab(y_test_knn_sample,y_pred)

#3. SVM

svm = LinearSVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)

#Metrics

log_accuracy=accuracy_score(y_test, y_pred)
log_recall=recall_score(y_test,y_pred)
log_precision=precision_score(y_test,y_pred)
log_rocauc=roc_auc_score(y_test,y_pred)

print('{:.4f}'.format(log_accuracy), '- Log Accuracy')
print('{:.4f}'.format(log_recall), '- Log Recall')
print('{:.4f}'.format(log_precision), '- Log Precision')
print('{:.4f}'.format(log_rocauc), '- Log ROC AUC')

#Confusion matrix

pd.crosstab(y_test,y_pred)

#Metrics looks good but the message 'liblinear failed to converge' suggests that the algorithm does not work so well, even though our training data has been standardized.

#4. Decision trees

dtree = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=5)
dtree.fit(x_train, y_train)
y_pred = dtree.predict(x_test)

log_accuracy=accuracy_score(y_test, y_pred)
log_recall=recall_score(y_test,y_pred)
log_precision=precision_score(y_test,y_pred)
log_rocauc=roc_auc_score(y_test,y_pred)

print('{:.4f}'.format(log_accuracy), '- Log Accuracy')
print('{:.4f}'.format(log_recall), '- Log Recall')
print('{:.4f}'.format(log_precision), '- Log Precision')
print('{:.4f}'.format(log_rocauc), '- Log ROC AUC')

pd.crosstab(y_test,y_pred)

#Seems that decision tress have the highest metrics of all - 96% ROC AUC

#5. Visualization

logit_roc_auc = roc_auc_score(y_test, classifier.predict(x_test))
tree_roc_auc = roc_auc_score(y_test, dtree.predict(x_test))
#knn_roc_auc = roc_auc_score(y_test, knn.predict(x_test))

fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(x_test)[:,1])
#fpr, tpr, thresholds = roc_curve(y_test, knn.predict_proba(x_test)[:,1])
fpr1, tpr1, thresholds = roc_curve(y_test, dtree.predict_proba(x_test)[:,1])
#fpr, tpr, thresholds = roc_curve(y_test, svm.decision_function(x_test))

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(fpr1, tpr1, label='Tree Regression (area = %0.2f)' % tree_roc_auc)
#plt.plot(fpr, tpr, label='Support Vector Machines (area = %0.2f)' % svm_roc_auc)
#plt.plot(fpr, tpr, label='K-nearest neighbours (area = %0.2f)' % knn_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
