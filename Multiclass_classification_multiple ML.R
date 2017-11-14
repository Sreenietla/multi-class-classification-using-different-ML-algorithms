# Setting the working Directory 
#setwd("~/Insofe/Labs/PGP_Batch_31CUTE02__Instructions_and_Exam_details/data")
getwd()
set.seed(8284)

# -------------------------------------------
# Loading common packages and libraries
# --------------------------------------------
install.packages("DMwR")
install.packages("caret")
install.packages("randomForest")
library(caret)
library(vegan)
library(dummies)
library(MASS) 
library(ROCR)
library(DMwR)
library(randomForest)

# ----------------------------------------------------
# (a) Performing required data preprocessing steps
# -----------------------------------------------------

data0<- read.csv("Contraceptive method choice.csv")
sum(is.na(data0)) # no missing values 
str(data0)
summary(data0)

# converting categorical data and numeric data
num_data<-subset(data0,select=c(Wife_age,Number_of_children_ever_born))
cat_data<-subset(data0,select=-c(Wife_age,Number_of_children_ever_born))
cat_data<-data.frame(apply(cat_data,2,function(x){as.factor(x)}))
data_2 <- cbind(cat_data,num_data)

# Splitting data into test and train 
indices <- createDataPartition(y=data_2$Method,p=0.8,list=F)
train_data <- data_2[indices,]
test_data <- data_2[-indices,]
str(train_data)
str(test_data)

# -------------------------------------------
# (b) BUILDING CLASSIFICATION MODELS 
# ----------------------------------------------

# ************* S-V-M *********************
# converting full dataset into categorical & then
# Splitting data into test and train 

catdata_svm<-data.frame(apply(data0,2,function(x){as.factor(x)}))
indices_svm <- createDataPartition(y=catdata_svm$Method,p=0.8,list=F)
train_svm <- catdata_svm[indices_svm,]
test_svm <- catdata_svm[-indices_svm,]
str(train_svm);str(test_svm)
library(e1071)
ctrl <- trainControl(method="cv", number=3,search = 'random')
linear.svm.tune <- train(Method~.,data=train_svm, method = "svmLinear",metric="Accuracy",trControl=ctrl)
linear.svm.tune
plot(linear.svm.tune)
predict.svm <-predict(linear.svm.tune,test_svm[,-10])
table(predict.svm,test_svm[,10])
confusionMatrix(predict.svm,test_svm[,10]) #(c) METRICS


# **********  LOGISTIC REGRESSION ***********
library("nnet")
model_glm <- multinom(Method~., data = train_svm)
summary(model_glm)
pred_glm <- predict(model_glm, test_svm[,-10])
table(pred_glm, test_svm[,10])
confusionMatrix(pred_glm, test_svm[,10]) # (c) METRICS


# ************** RPART *******************
library(rpart)
library(rpart.plot)

rpart_Decision_tree <- rpart(Method~., data=train_data)
summary(rpart_Decision_tree)
plotcp(rpart_Decision_tree)
rpart.plot(rpart_Decision_tree)
predict_rpart <- predict(rpart_Decision_tree,test_data[,-8],'class')
table(predict_rpart,test_data[,8])
confusionMatrix(predict_rpart,test_data[,8]) # (c) METRICS


# ***********RANDOM FOREST ************* 

model_RF_fit <- randomForest(Method ~ Wife_education + Husband_education+ Wife_religion
                          + Wife_Employment+ Husband_Occupation + Standard_of_living+Media_Exposure
                          +Wife_age+Number_of_children_ever_born, 
                          data = train_data, importance = TRUE, ntree = 900,
                          mtry=10)
model_RF_fit$predicted
varImpPlot(model_RF_fit)
predict_RF <-predict(model_RF_fit,test_data[,-8])
table(predict_RF,test_data[,8])
confusionMatrix(predict_RF,test_data[,8]) #(c) METRICS

# --------------------------------------
# (c) Evaluation of Metrics (Accuracy)
#-----------------------------------------
    # SVM ---------------------- 53.2%
    # Logistic Regression -------55.6%
    # Rpart ---------------------55.6%
    # Random Forest -------------53.5%

#-------------------------------------------------------------
# (d) Rpart & Logistic classification models both gives the better accuracy overall
#     It also depends on differnt train test data split and hyper parameters.
# ------------------------------------------------------------




