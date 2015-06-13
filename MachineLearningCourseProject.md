# Practical Machine Learning - Predicting Dumbell lift techniques

###Executive Summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this report, we aim to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to create a model to predict the way in which the dumbell lifts are performed (whether this be the correct way or one of 4 incorrect ways). 

###Read in the Data

First, we read in the data from the website URLs. Two distinct datasets are available - a training set to define and test our model, and a testing set to which our finalised model will be applied:


```r
#Read in the testing and training data from their respective URLs
download.file(url = "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = './data_train.csv',mode="wb")
download.file(url = "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = './data_test.csv',mode="wb")

train <-read.csv("data_train.csv", na.strings=c("NA","#DIV/0!",""))
test <-read.csv("data_test.csv", na.strings=c("NA","#DIV/0!",""))
```

###Process the Data

Next, we process the data to ensure that it is of a good quality for modelling. For the purposes of reproducibility, we set the seed to 1 during these calculations.

The first step is to impute values for the NA values in the dataset. Each NA value will be replaced with the mean value from the column in which it lies:


```r
set.seed(1)

suppressWarnings(suppressMessages(library(Hmisc)))
for (i in 1:length(train) ) {
   train[,i] <- impute(train[,i],fun=mean)
}
```

Then we remove all of the columns which have a near zero variance, to reduce the complexity of fitting a model to this data with a minimal impact on the model accuracy:


```r
suppressWarnings(suppressMessages(library(caret)))
set.seed(1)

train <- train[,-nearZeroVar(train)]
```

Next, we remove the first 7 columns from the data, since this data relates to metadata (such as the ID of the individual taking the study) which are irrelevant as predictors for our model:


```r
train <- train[,8:length(train)]
```

####Cross Validation Processing
Finally, we split the training data into a training and a testing subset of data. This will allow us to train our model on the cross-validation training subset, and test it on the cross-validation testing subset. We use the standard 60:40 split for this purpose:


```r
inTrain <- createDataPartition(y=train$classe, p=0.6, list=FALSE)
subTrain <- train[inTrain, ]
subTest <- train[-inTrain, ]
```

###Fitting a Model and using Cross Validation

Now that the data has been cleaned and processed, it is possible to fit a model to the data. We take the "classe" column as the outcome, with all the remaining fields as the predictors.

A random forest model was used for the fit. We have also used cross validation to help overcome the issue of overfitting, thus resulting in a more representative value for the sample error. We have chosen the number of resampling iterations to be set to 3 (aka 3 fold cross validation), as a weigh-off between accuracy and time to run the model:


```r
set.seed(1)

fit <- train(classe~., data=subTrain, method="rf",trControl=trainControl(method = "cv", number = 4))
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.3
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:Hmisc':
## 
##     combine
```

Now that the model has been fitted, we can test it against the test subset we created from our training data. In doing so, we omit the final column of data since this represents the actual outcome:


```r
prediction <- predict(fit, subTest[,1:length(subTest)-1], type="raw")
```

###Out of Sample Error

We use the predicted results of the section above to create the confusion matrix for the fit, and to study the out of sample error:


```r
confMatr <- confusionMatrix(prediction,subTest$classe)
confMatr
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2226    8    0    0    0
##          B    5 1503    6    0    0
##          C    1    7 1358   19    0
##          D    0    0    4 1266    6
##          E    0    0    0    1 1436
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9927          
##                  95% CI : (0.9906, 0.9945)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9908          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9973   0.9901   0.9927   0.9844   0.9958
## Specificity            0.9986   0.9983   0.9958   0.9985   0.9998
## Pos Pred Value         0.9964   0.9927   0.9805   0.9922   0.9993
## Neg Pred Value         0.9989   0.9976   0.9985   0.9970   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2837   0.1916   0.1731   0.1614   0.1830
## Detection Prevalence   0.2847   0.1930   0.1765   0.1626   0.1832
## Balanced Accuracy      0.9979   0.9942   0.9943   0.9915   0.9978
```

As we can see from the output of the confusion matrix, the out of sample error (calculated as "1-accuracy"") is approximately 0.7% (=1-0.993). This is a good result. Given that the final test set consists of just 20 records, we would expect no more than 1 of these records to be predicted incorrectly.

###Predictions for the test data

We now have an accurate working model which we can use to predict the Classe field for the 20 records we have been provided in the test data.

First we process the test data in the same way that we did for the training data, and then we predict the Classe value for each record using our model:


```r
#Impute missing values
for (i in 1:length(test) ) {
   test[,i] <- impute(test[,i],fun=mean)
}

#Remove irrelevant predictors
test <- test[,8:length(test)]

#Predict the value of classe for the test data, excluding the problem_id column as a predictor
predictionFinal <- predict(fit, test[,1:length(test)-1], type="raw")
```

Using the code provided in the assignment instructions, we then generate the files required for the Project Submission. As we expected due to our out of sample error rate of 0.7%, the test predictions generated a score of 20/20.

###Conclusions

Using the random forest machine learning algorithm, with cross validation set at 3 resampling iterations, we produced a model which could predict which of 5 ways dumbell lifts were being performed by an individual, with an out of sample error rate of 0.7%.

Across 20 test records to be submitted for this assignment, this out of sample error leads us to expect no more than 1 of these predictions to be incorrect. This was indeed the case, with all 20 results being correct on the first submission.
