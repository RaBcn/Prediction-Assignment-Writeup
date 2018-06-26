---
title: "Machine Learning Writeup"
author: "Ramon Serres"
date: "25 de junio de 2018"
output: html_document
---

# Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

# Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 


# Goal

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-). You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details.



# Cross-validation

Cross-validation will be performed by subsampling our training data into 2 subsamples: sub_Training data (75% of the original Training data set) and sub_Testing data (25% of the original Training data set). 

We will build our models will on the sub_Training data set, and tested on the sub_Testing data.

We will pick the best model and it will be finally tested on the original Testing data set.


# Expected out-of-sample error

The expected out-of-sample error will correspond to the quantity: 1-accuracy in the cross-validation data.

Accuracy :  the proportion of correct classified observation over the total sample in the subTesting data
set. 


# Packages loading

Loading required package: lattice
Loading required package: ggplot2
Loading required package: caret
Loading required package: randomForest
Loading required package: rpart
Loading required package: rpart.plot


```{r}

suppressMessages(library(lattice))
suppressMessages(library(ggplot2))
suppressMessages(library(caret))
suppressMessages(library(randomForest))
suppressMessages(library(rpart))
suppressMessages(library(rpart.plot))

```

# Getting and cleaning data

## Data can be found in the following links

```{r}

trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

```

## Load data

```{r}

set.seed(2222)

training <- read.csv(trainUrl, na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(testUrl, na.strings=c("NA","#DIV/0!",""))

```
## Clean data 

We want to remove columns with 100 % empty rows , also remove firts 7 columns from our dataset as those do not contain any relevant information for our prediction purposes

```{r}

training<-training[,colSums(is.na(training)) == 0]
testing <-testing[,colSums(is.na(testing)) == 0]

training   <-training[,-c(1:7)]
testing <-testing[,-c(1:7)]

```

## Cross validation

Use 70% of training set data to built a model, and use the rest to test the model

Partitioning training data set in training set into two sub datasets Training (70%) 
and Testing (30%)

```{r}

inTrain <- createDataPartition(y=training$classe, p=0.75, list=FALSE)

sub_Training <- training[inTrain, ]
sub_Testing <- training[-inTrain, ]

```

## Prediction Option 1: Decisions Tree using train and rpart functions

Let's start by creating a simple Decision Tree so that we have a clean and visual picture of the importance of each feauture 

```{r}

suppressMessages(library(rattle))

model_train_rpart<-train(classe ~ .,data = sub_Training,method = "rpart")
model_rpart <- rpart(classe ~ ., data=sub_Training,method="class")


fancyRpartPlot(model_train_rpart$finalModel)

fancyRpartPlot(model_rpart)

```

##  See Predicting accuracy with decision trees using both models

```{r}

predictions_model_train_rpart <- predict(model_train_rpart, sub_Testing)
confusionMatrix(predictions_model_train_rpart, sub_Testing$classe)

predictions_model_rpart <- predict(model_rpart, sub_Testing,type = "class")
confusionMatrix(predictions_model_rpart, sub_Testing$classe)

```

## Predict classe with all the other variables using a random forest ("rf"),boosted trees ("gbm") and linear  discriminant analysis ("lda") model.

Stack the predictions together using random forests ("rf")

What is the resulting accuracy on the test set? Is it better or worse than each of the individual predictions?

```{r}
mod_rf <- randomForest(classe ~ .,data = sub_Training,ntree=500)
# mod_gbm <- train(classe ~ .,data = sub_Training, method = "gbm")

# skip boosted trees method as taking too long to compute 

mod_lda <- train(classe ~ .,data = sub_Training, method = "lda")

pred_rf <- predict(mod_rf,sub_Testing)

pred_lda <- predict(mod_lda,sub_Testing)
```

# Accuracy using random forests
```{r}
confusionMatrix(pred_rf, sub_Testing$classe)$overall[1]
```

# Accuracy using linear discriminant analysis


```{r}
confusionMatrix(pred_lda, sub_Testing$classe)$overall[1]
```

# Create a staked model 

```{r}

predDF <- data.frame(pred_rf, pred_lda,classe=sub_Testing$classe)
combModFit <- train(classe ~ ., method = "rf", data = predDF)

combPred <- predict(combModFit, predDF)

```

# Stacked Accuracy
```{r}

confusionMatrix(combPred, sub_Testing$classe)$overall[1]

```

# Prediction Model to Use:

Random Forest algorithm alone performed better than LDA or Decition Trees .Also Stacked model did not perform any better. So we will choose Ramdom Forest (Accuracy was 0.9961 ). The expected out-of-sample error is estimated at 0.004, or 0.4%


# Predict classes of 20 test data
```{r}

predictfinal <- predict(mod_rf, testing, type="class")
predictfinal
```


