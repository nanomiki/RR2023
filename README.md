# RR2023
## Project Title:
"Comparing Python and R for Credit Risk Analysis using Machine Learning techniques."
## Project aim
This project aims to compare Python and R as technologies used for Credit Risk Analysis using Machine Learning techniques (logistic regression, KNN, SVM and decision trees). Our goal was to create a project in Python, reproduce it using R and compare strengths and weaknesses of both technologies. 
## Use of git and reproducibility
During the development of the project we used Git for version control. The main branch was used as the master and was protected from direct modification (approval for every pull request). In order to ensure reproducibility of the results random states in both SMOTE and train_test_split functions were set to 10.
## High level review of the project
Both files .r and .py include data inspection, ETL process, data analysis with ML techniques and comparison using Log Accuracy, Log Recall, Log Precision and Log ROC AUC metrics.
## Dataset
For the project we have used this data set https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset?resource=download, as explained in files we had to introduce extensive ETL process for handling missing values, highly correlated values and values without that did not make sense for our project. After all the exclusions and transformations we were left with ca. 800k observations.
## Development
The main steps we made in the project were:
* Load the dataset
* Import packages and functions needed for the analysis
* Data transformation
* Visual data investigation and preliminary statistical analysis
* Feature transformation
* Feature selection and encoding
* Check correctness of data in the final dataset
* Split, standarize and balance the dataset
* Perform modelling with LR, KNN, SVM and Decision trees
* Obtain final metrics and graphs
* Rewrite the whole code in R, perform the same analyses and compare

## Reasoning behind the data transformations and exclusions
Throughout the project we have conducted quite a lot of transformations and exclusions at varying stages.
* Obtain a binary flag from the credit status
* Get rid of columns having more than 30% of missing values
* Get rid of columns having too concentrated values or non-numerical and too fragmented values
* Transform date variables into the number of days from the date of analysis until the variable date
* Get rid of variables that are virtually the same, e.g. corr coeff of 0.98
* Minor transformations such as merge category 'any' and 'other' in' 'home_ownership' variable

## Results
As the project was primarily done in Python, we have assesed the results of the Python code. The best metrics of all the ML algorithms were achievied while using decision trees - 96% ROC AUC. KNN was too time-complex to calculate on the whole dataset so was performed only on a small, randomly chosen sub-sample. SVM failed to converge which suggests that whatever the result produced, should be taken with a grain of salt. Taking all of this into account, the oldest and simplest method, Linear Regression, which was the fastest and achieved only a marginally worse result than Decision Trees. 
## Comparison and conclusions 
We have found pros and cons of using both technologies for this project.
On one hand Python is a highly versatile language with a wide range of libraries and frameworks, it  has gained popularity as a go-to language for Machine Learning tasks.
It provides a rich ecosystem of libraries like NumPy, Pandas, and Scikit-learn that facilitate data manipulation, analysis, and modeling.
Additionally we were able to find lot of materials and guidelines for performing the analysis efficiently.
On the other hand R was specifically designed for statistical analysis and data manipulation, which was helpful during the ETL process.
But even though R in some parts have a more intuitive syntax and functions specifically designed for working with structured data, our technology of choice is Python.
Python integration with other tools and frameworks that are commonly used in the machine learning and data science domain,
such as TensorFlow and PyTorch allows to leverage the power of deep learning and neural networks for credit risk analysis.
When it comes to the comparison of results, both R and Python yielded similarly satisfying results. Code in R took a bit less time to complie and produce results, probably due to the amount of libraries in Python which slowed things down a bit.
