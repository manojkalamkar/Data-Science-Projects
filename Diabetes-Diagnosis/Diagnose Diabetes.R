#####================================================================================================================#####
##   Application      : Diagnose Diabetes                                                                               ##
##   ML Problem       : Binary Classification Model                                                                     ##
##   Model Version    : 1.0                                                                                             ##
##   Model Build Date : June 10, 2017                                                                                   ##
##   Team             : Manoj Kalamkar                                                                                  ##
##   Organization     : UPx Academy Certification Exam                                                                  ##
#####================================================================================================================#####

library(gtools)     # Load Package gtools
library(lattice)    # Load Lattice for caret package
library(ggplot2)    # Load Lattice for caret package
library(caret)      # Load Classification And REgression Training package
library(Rcpp)       # Load Lattice for Amelia package
library(Amelia)     # Load Package for Missingness Map Test
library(rattle)     # Load R Analytic Tool To Learn Easily - GUI Capability
library(rpart)      # Load Package for Recursive Partitioning
library(rpart.plot) # Load Package for Plotting Recursive Partitions
library(car)        # Load Package for Scatterplot Matrix
library(dplyr)      # Load Package dplyr
library(magrittr)
library(MASS)
library(gridExtra)
library(plotly)
library(formattable)

setwd("C:/Manoj/Data Science/Machine Learning/Certification Exam/Diabetics")
df_diabetes <- read.csv("pima-indians-diabetes.csv",header=FALSE) 

# Since there is no header to the dataset, assign meaningful headers
colnames(df_diabetes) <- c('times_pregnant','plasma','diastolic','triceps','insulin','bmi','pedigree','age','diabetes')

# convert diabetes feature to factor datatype
df_diabetes$diabetes <- as.factor(df_diabetes$diabetes)

# Understand dataset
dim(df_diabetes)
str(df_diabetes)
summary(df_diabetes)
View(df_diabetes)

attach(df_diabetes)

################################################## Univariate Analysis #######################################################

grid.arrange(qplot(times_pregnant), qplot(plasma), qplot(diastolic), qplot(triceps), qplot(insulin), qplot(bmi), qplot(pedigree))

table(df_diabetes$diabetes) %>% barplot(col = "wheat")

################################################## Bivariate Analysis #######################################################

ggplot(aes(x=times_pregnant,y=as.numeric(diabetes)),data = df_diabetes) + geom_jitter(aes(color= "red"),stat='summary',fun.y=mean)+ geom_smooth(method='lm', aes(group = 1))
ggplot(aes(x=plasma,y=as.numeric(diabetes)),data = df_diabetes) + geom_jitter(aes(color= "red"),stat='summary',fun.y=mean)+ geom_smooth(method='lm', aes(group = 1))
ggplot(aes(x=diastolic,y=as.numeric(diabetes)),data = df_diabetes) + geom_jitter(aes(color= "red"),stat='summary',fun.y=mean)+ geom_smooth(method='lm', aes(group = 1))
ggplot(aes(x=triceps,y=as.numeric(diabetes)),data = df_diabetes) + geom_jitter(aes(color= "red"),stat='summary',fun.y=mean)+ geom_smooth(method='lm', aes(group = 1))
ggplot(aes(x=insulin,y=as.numeric(diabetes)),data = df_diabetes) + geom_jitter(aes(color= "red"),stat='summary',fun.y=mean)+ geom_smooth(method='lm', aes(group = 1))
ggplot(aes(x=bmi,y=as.numeric(diabetes)),data = df_diabetes) + geom_jitter(aes(color= "red"),stat='summary',fun.y=mean)+ geom_smooth(method='lm', aes(group = 1))
ggplot(aes(x=pedigree,y=as.numeric(diabetes)),data = df_diabetes) + geom_jitter(aes(color= "red"),stat='summary',fun.y=mean)+ geom_smooth(method='lm', aes(group = 1))
ggplot(aes(x=age,y=as.numeric(diabetes)),data = df_diabetes) + geom_jitter(aes(color= "red"),stat='summary',fun.y=mean)+ geom_smooth(method='lm', aes(group = 1))

qplot(y=df_diabetes$times_pregnant, x= df_diabetes$diabetes, geom = "boxplot")
qplot(y=df_diabetes$plasma, x= df_diabetes$diabetes, geom = "boxplot")
qplot(y=df_diabetes$diastolic, x= df_diabetes$diabetes, geom = "boxplot")
qplot(y=df_diabetes$triceps, x= df_diabetes$diabetes, geom = "boxplot")
qplot(y=df_diabetes$insulin, x= df_diabetes$diabetes, geom = "boxplot")
qplot(y=df_diabetes$bmi, x= df_diabetes$diabetes, geom = "boxplot")
qplot(y=df_diabetes$pedigree, x= df_diabetes$diabetes, geom = "boxplot")
qplot(y=df_diabetes$age, x= df_diabetes$diabetes, geom = "boxplot")

################################################## Multivariate Analysis #######################################################

x <- qplot(x=df_diabetes$times_pregnant, y=df_diabetes$triceps, color=df_diabetes$diabetes, geom='point')+scale_shape(solid=FALSE); x
y <- qplot(x=df_diabetes$times_pregnant, y=df_diabetes$plasma, color=df_diabetes$diabetes, geom='point')+scale_shape(solid=FALSE); y
z <- qplot(x=df_diabetes$triceps, y=df_diabetes$plasma, color=df_diabetes$diabetes, geom='point')+scale_shape(solid=FALSE); z
a <- qplot(x=df_diabetes$triceps, y=df_diabetes$diastolic, color=df_diabetes$diabetes, geom='point')+scale_shape(solid=FALSE); a

################################################## Feature Engineering #######################################################

# Understand presence of missing values and effects with the features
missmap(df_diabetes ,main="Missingness MAP Test")  
sapply(df_diabetes , levels) #Shows that some features are missing data
#Shows that there are no missing values noticed

################################################## Build Predictive Models #######################################################

# Use 80% of the data to train the models and 20% to test the models

set.seed(1)
cdp_Index <- createDataPartition(df_diabetes$diabetes , p=0.80, list=FALSE) 
train_diabetes <- df_diabetes[cdp_Index,]
test_diabetes <- df_diabetes[-cdp_Index,]

dim(train_diabetes)
dim(test_diabetes)

#  Planning to run algorithms using 10-fold cross validation type of resampling

tC <- trainControl(method="cv", number=10)

###--------------------------------------------------------------------------------------------------------------------###
#  Build Models                                                                                                          #
#  - Linear Methods     - LDA, GLM                                                                                       #
#  - Non Linear Methods - kNN, SVM                                                                                       #
#  - Tree Methods       - CART                                                                                           #
#  - Ensemble Methods   - RF, GBM                                                                                        #
###--------------------------------------------------------------------------------------------------------------------###

set.seed(1)
fit.lda <- train(diabetes~., data=df_diabetes, method="lda", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.lda

set.seed(1)
fit.glm <- train(diabetes~., data=df_diabetes, method="glm", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.glm

set.seed(1)
fit.knn <- train(diabetes~., data=df_diabetes, method="knn", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.knn

set.seed(1)
fit.svm <- train(diabetes~., data=df_diabetes, method="svmRadial", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.svm

set.seed(1)
fit.cart <- train(diabetes~., data=df_diabetes, method="rpart", metric="Accuracy", trControl=tC)
fit.cart

set.seed(1)
fit.rf <- train(diabetes~., data=df_diabetes, method="rf", metric="Accuracy", trControl=tC)
fit.rf

set.seed(1)
fit.gbm <- train(diabetes~., data=df_diabetes, method="gbm", metric="Accuracy", trControl=tC, verbose=FALSE)
fit.gbm

#  Summarize the results of the models

results <- resamples(list(m_lda=fit.lda, m_glm=fit.glm, m_cart=fit.cart, m_knn=fit.knn, m_svm=fit.svm, m_rf=fit.rf, m_gbm=fit.gbm))
results

summary(results)
dotplot(results)
bwplot(results, col="red", main="Accuracy Results of Models")

# Predict models with the validation set of training data partition

predict_lda <- predict(fit.lda, test_diabetes)
confusionMatrix(predict_lda,test_diabetes$diabetes)  # Accuracy=0.78 Kappa=0.49
plot(predict_lda, col="pink", xlab="Predicted Diabetes", ylab="Predicted Diabetes", main="Linear - LDA Predictions")

predict_glm <- predict(fit.glm, test_diabetes)
confusionMatrix(predict_glm,test_diabetes$diabetes)  # Accuracy=0.78 Kappa=0.49

predict_knn <- predict(fit.knn, test_diabetes)
confusionMatrix(predict_knn,test_diabetes$diabetes)  # Accuracy=0.79 Kappa=0.52

predict_svm <- predict(fit.svm, test_diabetes)
confusionMatrix(predict_svm,test_diabetes$diabetes)  # Accuracy=0.81 Kappa=0.56
plot(predict_svm, col="purple", xlab="Predicted Diabetes Status", ylab="Predicted Diabetes", main="Ensemble - SVM Predictions")

predict_cart <- predict(fit.cart, test_diabetes)
confusionMatrix(predict_cart,test_diabetes$diabetes) # Accuracy=0.76 Kappa=0.45
fancyRpartPlot(fit.cart$finalModel)
plot(varImp(fit.cart)) #Shows plasma is major criteria for diagnose diabetes

predict_rf <- predict(fit.rf, test_diabetes)
confusionMatrix(predict_rf,test_diabetes$diabetes)   # Accuracy=1 Kappa=1
plot(varImp(fit.rf))
plot(predict_rf, col="purple", xlab="Predicted Diabetes Status", ylab="Predicted Diabetes", main="Ensemble - RF Predictions")

predict_gbm <- predict(fit.gbm, test_diabetes)
confusionMatrix(predict_gbm,test_diabetes$diabetes)  # Accuracy=0.82 Kappa=0.6
plot(varImp(fit.gbm))

fit_results <- data.frame(Class = test_diabetes$diabetes)

fit_results$LDA <- predict(fit.lda, test_diabetes, type="prob")
fit_results$CART <- predict(fit.cart, test_diabetes, type="prob")
fit_results$KNN <- predict(fit.knn, test_diabetes, type="prob")
fit_results$RF <- predict(fit.rf, test_diabetes, type="prob")
fit_results$GBM <- predict(fit.gbm, test_diabetes, type="prob")

# AUC Curve

library(ROCR)

pred <- prediction(as.numeric(predict_lda),as.numeric(test_diabetes$diabetes))
roc <- performance(pred,"tpr","fpr")
plot(roc, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")
abline(a=0,b=1)

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.1, auc, title = "LDA - AUC", cex = 1)

pred <- prediction(as.numeric(predict_glm),as.numeric(test_diabetes$diabetes))
roc <- performance(pred,"tpr","fpr")
plot(roc, add = TRUE, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.2, auc, title = "GLM - AUC", cex = 1)

pred <- prediction(as.numeric(predict_knn),as.numeric(test_diabetes$diabetes))
roc <- performance(pred,"tpr","fpr")
plot(roc, add = TRUE, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.3, auc, title = "KNN-AUC", cex = 1)

pred <- prediction(as.numeric(predict_svm),as.numeric(test_diabetes$diabetes))
roc <- performance(pred,"tpr","fpr")
plot(roc, add = TRUE, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.4, auc, title = "SVM-AUC", cex = 1)

pred <- prediction(as.numeric(predict_cart),as.numeric(test_diabetes$diabetes))
roc <- performance(pred,"tpr","fpr")
plot(roc, add = TRUE, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.4, auc, title = "CART-AUC", cex = 1)

pred <- prediction(as.numeric(predict_rf),as.numeric(test_diabetes$diabetes))
roc <- performance(pred,"tpr","fpr")
plot(roc, add = TRUE, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.5, auc, title = "RF-AUC", cex = 1)

pred <- prediction(as.numeric(predict_gbm),as.numeric(test_diabetes$diabetes))
roc <- performance(pred,"tpr","fpr")
plot(roc, add = TRUE, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.6, auc, title = "GBM-AUC", cex = 1)

