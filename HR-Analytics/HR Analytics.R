#####================================================================================================================#####
##   Application      : HR Analytics                                                                                    ##
##   ML Problem       : Binary Classification Model                                                                     ##
##   Model Version    : 1.0                                                                                             ##
##   Model Build Date : June 10, 2017                                                                                   ##
##   Team             : Manoj Kalamkar                                                                                  ##
##   Organization     : UPx Academy Certification Exam                                                                  ##
#####================================================================================================================#####

#    Load Required Packages
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

train_hr <- read.csv("HR_comma_sep.csv")

dim(train_hr)
str(train_hr)
summary(train_hr)
View(train_hr)
sapply(train_hr,class)

attach(train_hr)

################################################## Univariate Analysis #######################################################

hist(left, col = "grey", main = "Univariate Analysis - Left", ylab = "Frequency", freq = TRUE,labels = TRUE)
# About 23% employees left company with given data

hist(satisfaction_level, col = "grey", main = "Univariate Analysis - Satisfaction Level", ylab = "Frequency", freq = TRUE,labels = TRUE)
# More employees seem to satisfied

hist(last_evaluation, col = "grey", main = "Univariate Analysis - Last Evaluation", ylab = "Frequency", freq = FALSE)
# Evaluation data seem to be even 

hist(number_project, col = "grey", main = "Univariate Analysis - number_project", ylab = "Frequency", freq = FALSE)
# # of projects empoyees worked are higher for 3 & 4 projects

hist(average_montly_hours, col = "grey", main = "Univariate Analysis - Average Monthly Hours", ylab = "Frequency", freq = TRUE)
hist(time_spend_company, col = "grey", main = "Univariate Analysis - Time Spend In Company", ylab = "Frequency", freq = FALSE)
# Many employees joined very recently 2 to 3 years are more

hist(Work_accident, col = "grey", main = "Univariate Analysis - Work Accident", ylab = "Frequency", freq = FALSE)
# Seem to be higher work accident

hist(promotion_last_5years, col = "grey", main = "Univariate Analysis - Promotion Last 5 Years", ylab = "Frequency", freq = TRUE, labels = TRUE)
# very few employees got promotion in last 5 years. Only around 2%

barplot(table(sales))
# There are more Sales, Technical employees in company than others

barplot(table(salary))
# Low and Medium salaries are higher than High

plot(train_hr$salary, col="wheat")          #Overall employees had less higher than Low/Medium salaries as expected
prop.table(table(train_hr$salary))

################################################## Bivariate Analysis #######################################################

qplot(y=last_evaluation, x= factor(left), geom = "boxplot")
# Employees who left seem to have wide range of last evaluation

qplot(y=satisfaction_level, x= factor(left), geom = "boxplot")
# Satisfaction level seemed to be low for employees who left. This seems to be strong predictor

qplot(y=number_project, x= factor(left), geom = "boxplot")
# Wide range for employees who left

qplot(y=average_montly_hours, x= factor(left), geom = "boxplot")

qplot(y=time_spend_company, x= factor(left), geom = "boxplot")
# Employees who left have more time spend in company

###--------------------------------------------------------------------------------------------------------------------###
#  Analyse features influencing predictiveness (feature engineering)                                                     #
#  - Understand presence of missing values and effects with the features                                                 #
#  - Understand which features influence company leaving decision                                                       #
###--------------------------------------------------------------------------------------------------------------------###

missmap(train_hr,main="Missingness MAP Test")  
sapply(train_hr, levels) #Shows that some features are missing data
#Shows that there are no missing values noticed


#####================================================================================================================#####
##   Get ahead to build the Predictive Models                                                                           ##
##   - Use 80% of the data to train the models and 20% to validate the models                                           ##
##   - To train the model create 10 fold cross validation dataset                                                       ##
##   - Build the predictive models                                                                                      ##
##   - Apply the model to validate the accuracy with the validation data provided                                       ##
##   - Select the Best Model                                                                                            ##
#####================================================================================================================#####

set.seed(999)
cdp_Index <- createDataPartition(train_hr$left, p=0.80, list=FALSE) 
train_hr_left <- train_hr[cdp_Index,]
val_hr_left <- train_hr[-cdp_Index,]
dim(val_hr_left)  
dim(train_hr_left)  
head(train_hr_left)

train_hr_left$left = factor(train_hr_left$left)
  
###--------------------------------------------------------------------------------------------------------------------###
#  Planning to run algorithms using 10-fold cross validation type of resampling                                          #
###--------------------------------------------------------------------------------------------------------------------###

tC <- trainControl(method="cv", number=10)

###--------------------------------------------------------------------------------------------------------------------###
#  Build Models                                                                                                          #
#  - Linear Methods     - LDA, GLM                                                                                       #
#  - Non Linear Methods - kNN, SVM                                                                                       #
#  - Tree Methods       - CART                                                                                           #
#  - Ensemble Methods   - RF, GBM                                                                                        #
###--------------------------------------------------------------------------------------------------------------------###

set.seed(999)
fit.lda <- train(left~., data=train_hr_left, method="lda", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.lda

set.seed(999)
fit.glm <- train(left~., data=train_hr_left, method="glm", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.glm

set.seed(999)
fit.knn <- train(left~., data=train_hr_left, method="knn", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.knn

set.seed(999)
fit.svm <- train(left~., data=train_hr_left, method="svmRadial", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.svm

set.seed(999)
fit.cart <- train(left~., data=train_hr_left, method="rpart", metric="Accuracy", trControl=tC)
fit.cart

set.seed(999)
fit.rf <- train(left~., data=train_hr_left, method="rf", metric="Accuracy", trControl=tC)
fit.rf

set.seed(999)
fit.gbm <- train(left~., data=train_hr_left, method="gbm", metric="Accuracy", trControl=tC, verbose=FALSE)
fit.gbm

###--------------------------------------------------------------------------------------------------------------------###
#  Summarize the results of the models                                                                                   #
###--------------------------------------------------------------------------------------------------------------------###

results <- resamples(list(m_lda=fit.lda, m_glm=fit.glm, m_cart=fit.cart, m_knn=fit.knn, m_svm=fit.svm, m_rf=fit.rf, m_gbm=fit.gbm))
results

summary(results)
dotplot(results)
bwplot(results, col="red", main="Accuracy Results of Models")

print(fit.lda)
print(fit.glm)
print(fit.cart)
print(fit.knn)
print(fit.svm)
print(fit.rf)
print(fit.gbm)

###--------------------------------------------------------------------------------------------------------------------###
#  Predict models with the validation set of training data partition                                                     #
###--------------------------------------------------------------------------------------------------------------------###

predict_lda <- predict(fit.lda, val_hr_left)
confusionMatrix(predict_lda,val_hr_left$left)
plot(predict_lda, col="pink", xlab="Predicted Left Status", ylab="Predicted Left", main="Linear - LDA Predictions")

predict_glm <- predict(fit.glm, val_hr_left)
confusionMatrix(predict_glm,val_hr_left$left)

predict_knn <- predict(fit.knn, val_hr_left)
confusionMatrix(predict_knn,val_hr_left$left)

predict_svm <- predict(fit.svm, val_hr_left)
confusionMatrix(predict_svm,val_hr_left$left)
plot(predict_svm, col="purple", xlab="Predicted Left Status", ylab="Predicted Loans", main="Ensemble - SVM Predictions")

predict_cart <- predict(fit.cart, val_hr_left)
confusionMatrix(predict_cart,val_hr_left$left)
fancyRpartPlot(fit.cart$finalModel)
plot(varImp(fit.cart),top=20) #Shows Satisfaction level is major criteria to decide the left status

predict_rf <- predict(fit.rf, val_hr_left)
confusionMatrix(predict_rf,val_hr_left$left)
plot(varImp(fit.rf),top=20)
plot(predict_rf, col="purple", xlab="Predicted Left Status", ylab="Predicted Loans", main="Ensemble - RF Predictions")

predict_gbm <- predict(fit.gbm, val_hr_left)
confusionMatrix(predict_gbm,val_hr_left$left)  # Accuracy=0.809 Kappa=0.497
plot(varImp(fit.gbm),top=20)

fit_results <- data.frame(Class = val_hr_left$left)

fit_results$LDA <- predict(fit.lda, val_hr_left, type="prob")
fit_results$CART <- predict(fit.cart, val_hr_left, type="prob")
fit_results$KNN <- predict(fit.knn, val_hr_left, type="prob")
fit_results$RF <- predict(fit.rf, val_hr_left, type="prob")
fit_results$GBM <- predict(fit.gbm, val_hr_left, type="prob")

# AUC Curve

library(ROCR)

pred <- prediction(as.numeric(predict_lda),as.numeric(val_hr_left$left))
roc <- performance(pred,"tpr","fpr")
plot(roc, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")
abline(a=0,b=1)

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.1, auc, title = "LDA - AUC", cex = 1)

pred <- prediction(as.numeric(predict_glm),as.numeric(val_hr_left$left))
roc <- performance(pred,"tpr","fpr")
plot(roc, add = TRUE, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.2, auc, title = "GLM - AUC", cex = 1)

pred <- prediction(as.numeric(predict_knn),as.numeric(val_hr_left$left))
roc <- performance(pred,"tpr","fpr")
plot(roc, add = TRUE, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.3, auc, title = "KNN-AUC", cex = 1)

pred <- prediction(as.numeric(predict_svm),as.numeric(val_hr_left$left))
roc <- performance(pred,"tpr","fpr")
plot(roc, add = TRUE, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.4, auc, title = "SVM-AUC", cex = 1)

pred <- prediction(as.numeric(predict_cart),as.numeric(val_hr_left$left))
roc <- performance(pred,"tpr","fpr")
plot(roc, add = TRUE, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.4, auc, title = "CART-AUC", cex = 1)

pred <- prediction(as.numeric(predict_rf),as.numeric(val_hr_left$left))
roc <- performance(pred,"tpr","fpr")
plot(roc, add = TRUE, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.5, auc, title = "RF-AUC", cex = 1)

pred <- prediction(as.numeric(predict_gbm),as.numeric(val_hr_left$left))
roc <- performance(pred,"tpr","fpr")
plot(roc, add = TRUE, colorize=T, main = 'ROC Curve', ylab = "Sensitivity", xlab = "1-Specificity")

auc <- performance(pred,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(.6,.6, auc, title = "GBM-AUC", cex = 1)

