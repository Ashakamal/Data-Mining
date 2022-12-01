##Used Packages
library(knitr)
library(caret)
library(gmodels)
library(ROCR)
library(naivebayes)
library(C50)
library(randomForest)
library(dplyr)
##Reading the Dataset from local machine
#setwd("C:/Data_Mining/Assesment/Rcode")
data<-read.csv(file="./Fulldataset.csv",stringsAsFactors = T)
######Data exploration#################################################
# To check the dimension of the data 
dim(data)
##Exploring the Age attribute
hist(data$age, col="green",main="Age histogram",xlab="Age",xlim=c(10,100),breaks=10,border="blue")
##Exploring of balance Attribute
boxplot(data$balance,xlab="Balance Amount")
##Exploring the job Attribute
barplot(table(data$job),las=2)
##Exploring the Marital attribute
barplot(table(data$marital),las=2)
##Exploring the poutcome attribute.
pie(table(data$poutcome))
##Exploring the month attribute
barplot(table(data$month),las=2)
##Exploring the previous Attribute
barplot(table(data$previous))

##Exploring the campaign Attribute
hist(data$campaign, col="gray",xlab="Campaign Contacts",xlim=c(0,30),border="red")

######Data pre processing#####################################################

##Checking for the Missing data
anyNA(data)
###Data cleansing
##Removing the unwanted columns day and previous 
bank = subset(data, select = -c(day, previous))
head(bank,2)
##Checking For Outliers
boxplot(bank)
##Removing outliers
bank<-bank[!bank$balance>5e+04,]
bank<-bank[!bank$balance==-8019,]
bank<-bank[!bank$balance==-6847,]
bank<-bank[!bank$duration>3500,]
##ploting after removing outliers
boxplot(bank)
boxplot(bank$balance)
boxplot(bank$duration)
##Dimension of data after outlier removal
dim(bank)
##changing the categorical factors into numerical
bank$poutcome<-as.numeric(factor(bank$poutcome))
bank$housing<-as.numeric(factor(bank$housing))
bank$loan<-as.numeric(factor(bank$loan))
##checking the data structure
str(bank)
###Spliting the data in to train and test data
split<-createDataPartition(bank$y,p=0.7,list = FALSE)
train<-bank[split,]
test<-bank[-split,]
dim(train)
dim(test)
###Feature Scaling standardizing and normalizing
train[, c(1,6)] = scale(train[, c(1,6)])
test[, c(1,6)] = scale(test[, c(1,6)])
head(train,2)

######Experiments######################################################

#####Logistic Regression Model########################################
##Model is built using k-fold algorithm
set.seed(1)
kfoldrepeated<-trainControl(method = "repeatedcv",number = 5,repeats = 3)
model_bin<-train(y ~ .,data=train,method="glm",trControl=kfoldrepeated)
print(model_bin)

####Evaluating the model Using Train Data##
pred_train<-predict(model_bin,train)
##Evaluate the model results from the train using the confusion matrix
R<-confusionMatrix(as.factor(train$y),as.factor(pred_train))
##Calculating Precision
Precision <- R$byClass['Pos Pred Value'] 
##Calculating Recall
Recall <- R$byClass['Sensitivity']
##Calculating Specificity
Specificity<-R$byClass['Specificity']
##Calculating Accuracy
Accuaracy<-(R$table[1,1]+R$table[2,2])/(R$table[1,1]+R$table[1,2]+R$table[2,1]+R$table[2,2])
##Calculating Error Rate
ErrorRate<-(R$table[1,2]+R$table[2,1])/(R$table[1,1]+R$table[1,2]+R$table[2,1]+R$table[2,2])
cat("The Accuracy of the Train Data is :" ,Accuaracy)
cat("The Precision of the Train Data is :" ,Precision)
cat("The Recall of the Train Data is :" ,Recall)
cat("The Error Rate of the Train Data is :" ,ErrorRate)
cat("The Specificity of the Train Data is :" ,Specificity)
## Creating the table for confusion metrics
D<-table(train$y, pred_train)
D
##Proportion of prediction values
prop.table(D)
##Finding the value of Area under curve
pr<-prediction(as.numeric(pred_train),as.numeric(train$y))
auc<-performance(pr,measure = "auc")
auc<-auc@y.values
auc
##Plotting the performance of the model
prf<-performance(pr,measure = "tpr",x.measure = "fpr")
plot(prf,width=200,height=200)

##Evaluating the model using test data##
pred_test<-predict(model_bin,test)
##Evaluate the model results from the test using the confusion matrix
R1<-confusionMatrix(as.factor(test$y),as.factor(pred_test))
##Calculating Precision
Precision <- R1$byClass['Pos Pred Value']  
##Calculating Recall
Recall <- R1$byClass['Sensitivity']
##Calculating Specificity
Specificity<-R1$byClass['Specificity']
##Calculating Accuracy
Accuaracy<-(R1$table[1,1]+R1$table[2,2])/(R1$table[1,1]+R1$table[1,2]+R1$table[2,1]+R1$table[2,2])
##Calculating Error rate
ErrorRate<-(R1$table[1,2]+R1$table[2,1])/(R1$table[1,1]+R1$table[1,2]+R1$table[2,1]+R1$table[2,2])
cat("The Accuracy of the Train Data is :" ,Accuaracy)
cat("The Precision of the Train Data is :" ,Precision)
cat("The Recall of the Train Data is :" ,Recall)
cat("The Error Rate of the Train Data is :" ,ErrorRate)
cat("The Specificity of the Train Data is :" ,Specificity)
## Creating the table for confusion metrics
D<-table(test$y, pred_test)
D
##Proportion of prediction values
prop.table(D)
##Finding the value of Area under curve
pr_t<-prediction(as.numeric(pred_test),as.numeric(test$y))
auc_t<-performance(pr_t,measure = "auc")
auc_t<-auc_t@y.values
auc_t
##Plotting the performance of the model
prf_t<-performance(pr_t,measure = "tpr",x.measure = "fpr")
plot(prf_t)

#####Naive Bayes##########################################

##Building the model using Naive bayes algorithm
set.seed(1)
model_nb<-naive_bayes(y ~ ., data=train)
model_nb$prior

##Evaluating the model using Train data
pred_trainnb<-predict(model_nb,train)
##Evaluate the model results from the train using the confusion matrix
R2<-confusionMatrix(as.factor(train$y),as.factor(pred_trainnb))
##Calculating Precision
Precision <- R2$byClass['Pos Pred Value']
##Calculating Recall
Recall <- R2$byClass['Sensitivity']
##Calculating Specificity
Specificity<-R2$byClass['Specificity']
##Calculating Accuracy
Accuaracy<-(R2$table[1,1]+R2$table[2,2])/(R2$table[1,1]+R2$table[1,2]+R2$table[2,1]+R2$table[2,2])
##Calculating Error rate
ErrorRate<-(R2$table[1,2]+R2$table[2,1])/(R2$table[1,1]+R2$table[1,2]+R2$table[2,1]+R2$table[2,2])
cat("The Accuracy of the Train Data is :" ,Accuaracy)
cat("The Precision of the Train Data is :" ,Precision)
cat("The Recall of the Train Data is :" ,Recall)
cat("The Error Rate of the Train Data is :" ,ErrorRate)
cat("The Specificity of the Train Data is :" ,Specificity)
##Creating the table for confusion metrics
D<-table(train$y, pred_trainnb)
D
##Proportion of prediction
prop.table(D)
##Finding the value of Area under curve
pr<-prediction(as.numeric(pred_trainnb),as.numeric(train$y))
auc<-performance(pr,measure = "auc")
auc<-auc@y.values
auc
##Plotting the performance of the model
prf<-performance(pr,measure = "tpr",x.measure = "fpr")
plot(prf)

##Evaluating the model using test data
pred_testnb<-predict(model_nb,test)
##Evaluate the model results from the test using the confusion matrix
R3<-confusionMatrix(as.factor(test$y),as.factor(pred_testnb))
##Calculating Precision
Precision <- R3$byClass['Pos Pred Value'] 
##Calculating Recall
Recall <- R3$byClass['Sensitivity']
##Calculating specificity
Specificity<-R3$byClass['Specificity']
##Calculating Accuracy
Accuaracy<-(R3$table[1,1]+R3$table[2,2])/(R3$table[1,1]+R3$table[1,2]+R3$table[2,1]+R3$table[2,2])
##Calculating Error rate
ErrorRate<-(R3$table[1,2]+R3$table[2,1])/(R3$table[1,1]+R3$table[1,2]+R3$table[2,1]+R3$table[2,2])
cat("The Accuracy of the Train Data is :" ,Accuaracy)
cat("The Precision of the Train Data is :" ,Precision)
cat("The Recall of the Train Data is :" ,Recall)
cat("The Error Rate of the Train Data is :" ,ErrorRate)
cat("The Specificity of the Train Data is :" ,Specificity)
## Creating the table for confusion metrics
D<-table(test$y, pred_testnb)
D
##Proportions of prediction
prop.table(D)
##Finding the value of Area under curve
prnb_t<-prediction(as.numeric(pred_testnb),as.numeric(test$y))
aucnb_t<-performance(prnb_t,measure = "auc")
aucnb_t<-aucnb_t@y.values
aucnb_t
##Plotting the performance of the model
prfnb_t<-performance(prnb_t,measure = "tpr",x.measure = "fpr")
plot(prfnb_t)
##Evaluating the naive bayes model using the K fold algorithm ####
set.seed(1)
Knb<-trainControl(method = "repeatedcv",number = 5, repeats=3)
model1<-train(y ~age+job+marital+education+default+balance+housing+loan+contact+month+duration+campaign+pdays+poutcome,data=bank,method="naive_bayes",trControl=Knb)
print(model1)

#####C5.0 Decision Tree######################################################

##Building the model using C5.0 Decision tree algorithm
set.seed(1)
model_c50<-C5.0(y ~ ., data=train)
model_c50

##Evaluating the model using Train data
pred_trainc50<-predict(model_c50,train)
##Evaluate the model results from the train using the confusion matrix
R4<-confusionMatrix(as.factor(train$y),as.factor(pred_trainc50))
##Calculating Precision
Precision <- R4$byClass['Pos Pred Value']
##Calculating Recall
Recall <- R4$byClass['Sensitivity']
##Calculating Specificity
Specificity<-R4$byClass['Specificity']
##Calculating Accuracy
Accuaracy<-(R4$table[1,1]+R4$table[2,2])/(R4$table[1,1]+R4$table[1,2]+R4$table[2,1]+R4$table[2,2])
##Calculating Error rate
ErrorRate<-(R4$table[1,2]+R4$table[2,1])/(R4$table[1,1]+R4$table[1,2]+R4$table[2,1]+R4$table[2,2])
cat("The Accuracy of the Train Data is :" ,Accuaracy)
cat("The Precision of the Train Data is :" ,Precision)
cat("The Recall of the Train Data is :" ,Recall)
cat("The Error Rate of the Train Data is :" ,ErrorRate)
cat("The Specificity of the Train Data is :" ,Specificity)
## Creating the table for confusion metrics
D<-table(train$y, pred_trainc50)
D
##Proportion of prediction
prop.table(D)
##Finding the value of Area under curve
prc50<-prediction(as.numeric(pred_trainc50),as.numeric(train$y))
aucc50<-performance(prc50,measure = "auc")
aucc50<-aucc50@y.values
aucc50
##Plotting the performance of the model
prfc50<-performance(prc50,measure = "tpr",x.measure = "fpr")
plot(prfc50,width=200,height=200)

##Evaluating the model using test data
pred_testc50<-predict(model_c50,test)
##Evaluate the model results from the test using the confusion matrix
R5<-confusionMatrix(as.factor(test$y),as.factor(pred_testc50))
##Calculating Precision
Precision <- R5$byClass['Pos Pred Value'] 
##Calculating Recall
Recall <- R5$byClass['Sensitivity']
##Calculating Specificity
Specificity<-R5$byClass['Specificity']
##Calculating Accuracy
Accuaracy<-(R5$table[1,1]+R5$table[2,2])/(R5$table[1,1]+R5$table[1,2]+R5$table[2,1]+R5$table[2,2])
##Calculating Error rate
ErrorRate<-(R5$table[1,2]+R5$table[2,1])/(R5$table[1,1]+R5$table[1,2]+R5$table[2,1]+R5$table[2,2])
cat("The Accuracy of the Train Data is :" ,Accuaracy)
cat("The Precision of the Train Data is :" ,Precision)
cat("The Recall of the Train Data is :" ,Recall)
cat("The Error Rate of the Train Data is :" ,ErrorRate)
cat("The Specificity of the Train Data is :" ,Specificity)
## Creating the table for confusion metrics
D<-table(test$y, pred_testc50)
D
##Proportions of prediction
prop.table(D)
##Finding the value of Area under curve
prc50_t<-prediction(as.numeric(pred_testc50),as.numeric(test$y))
aucc50_t<-performance(prc50_t,measure = "auc")
aucc50_t<-aucc50_t@y.values
aucc50_t 
##Plotting the performance of the model
prfc50_t<-performance(prc50_t,measure = "tpr",x.measure = "fpr")
plot(prfc50,width=200,height=200)
##Evaluating the C5.0 Decision tree model using the K fold algorithm#####
set.seed(1)
Kdt<-trainControl(method = "repeatedcv",number = 5, repeats=3)
modelk<-train(y ~ .,data=bank,method="rpart",trControl=Kdt)
print(modelk)


######RANDOM FOREST###########################################################
##Building the model using Random forest algorithm
model_rf<-randomForest(y ~ ., data=train,mtry=3)
model_rf
plot(model_rf)

##Evaluating the model using Train data
pred_trainrf<-predict(model_rf,train)
##Evaluate the model results from the train using the confusion matrix
R6<-confusionMatrix(as.factor(train$y),as.factor(pred_trainrf))
##Calculating Precision
Precision <- R6$byClass['Pos Pred Value']
##Calculating Precision
Recall <- R6$byClass['Sensitivity']
##Calculating Specificity
Specificity<-R6$byClass['Specificity']
##Calculating Accuracy
Accuaracy<-(R6$table[1,1]+R6$table[2,2])/(R6$table[1,1]+R6$table[1,2]+R6$table[2,1]+R6$table[2,2])
##Calculating Error rate
ErrorRate<-(R6$table[1,2]+R6$table[2,1])/(R6$table[1,1]+R6$table[1,2]+R6$table[2,1]+R6$table[2,2])
cat("The Accuracy of the Train Data is :" ,Accuaracy)
cat("The Precision of the Train Data is :" ,Precision)
cat("The Recall of the Train Data is :" ,Recall)
cat("The Error Rate of the Train Data is :" ,ErrorRate)
cat("The Specificity of the Train Data is :" ,Specificity)
## Creating the table for confusion metrics
D<-table(train$y, pred_trainrf)
D
#Finding the important variables
varImpPlot(model_rf)
importance(model_rf)
##Proportions of prediction
prop.table(D)
##Finding the value of Area under curve
prrf<-prediction(as.numeric(pred_trainrf),as.numeric(train$y))
aucrf<-performance(prrf,measure = "auc")
aucrf<-aucrf@y.values
aucrf
##Plotting the performance of the model
prfrf<-performance(prrf,measure = "tpr",x.measure = "fpr")
plot(prfrf,width=200,height=200)
##Evaluating the model using test data
pred_testrf<-predict(model_rf,test)
##Evaluate the model results from the train using the confusion matrix
R7<-confusionMatrix(as.factor(test$y),as.factor(pred_testrf))
##Calculating Precision
Precision <- R7$byClass['Pos Pred Value']  
##Calculating Recall
Recall <- R7$byClass['Sensitivity']
##Calculating specificity
Specificity<-R7$byClass['Specificity']
##Calculating Accuracy
Accuaracy<-(R7$table[1,1]+R7$table[2,2])/(R7$table[1,1]+R7$table[1,2]+R7$table[2,1]+R7$table[2,2])
##Calculating Error rate
ErrorRate<-(R7$table[1,2]+R7$table[2,1])/(R7$table[1,1]+R7$table[1,2]+R7$table[2,1]+R7$table[2,2])
cat("The Accuracy of the Train Data is :" ,Accuaracy)
cat("The Precision of the Train Data is :" ,Precision)
cat("The Recall of the Train Data is :" ,Recall)
cat("The Error Rate of the Train Data is :" ,ErrorRate)
cat("The Specificity of the Train Data is :" ,Specificity)
## Creating the table for confusion metrics
D<-table(test$y, pred_testrf)
D
##Proportion of prediction
prop.table(D)
##Finding the value of Area under curve
prrf_t<-prediction(as.numeric(pred_testrf),as.numeric(test$y))
aucrf_t<-performance(pr,measure = "auc")
aucrf_t<-aucrf_t@y.values
aucrf_t
##Plotting the performance of the model
prfrf_t<-performance(prrf_t,measure = "tpr",x.measure = "fpr")
plot(prfrf_t,width=200,height=200)


