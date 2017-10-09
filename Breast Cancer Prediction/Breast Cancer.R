#####================================================================================================================#####
##   Application      : Breast Cancer                                                                                   ##
##   ML Problem       : Breast Cancer Wisconsin (Diagnostic) Data Set                                                   ##
##   Model Version    : 1.0                                                                                             ##
##   Model Build Date : May 29, 2017                                                                                     ##
##   Team             : Manoj Kalamkar                                                                                  ##
##   Organization     : Mango Corporation                                                                               ##
#####================================================================================================================#####
"
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. 
n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: 'Robust Linear Programming Discrimination of Two Linearly Inseparable Sets', 
Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Attribute Information:

1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area 
e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) 
h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ('coastline approximation' - 1)

The mean, standard error and 'worst' or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. 
For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant
"

###--------------------------------------------------------------------------------------------------------------------###
#    Load Required Packages                                                                                              #
###--------------------------------------------------------------------------------------------------------------------###

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

breast_data <- read.csv("data.csv")

breast_data <- subset(breast_data,select = -X)
  
dim(breast_data)
str(breast_data)
summary(breast_data)
View(breast_data)
sapply(breast_data,class)

attach(breast_data)

################################################## Univariate Analysis #######################################################

hist(diagnosis, col = "grey", main = "Univariate Analysis - Left", ylab = "Frequency", freq = TRUE,labels = TRUE)
# About 23% employees left company with given data

hist(radius_mean, col = "grey", main = "Univariate Analysis - Radius Mean", ylab = "Frequency", freq = TRUE,labels = TRUE)
# More employees seem to satisfied

hist(texture_mean, col = "grey", main = "Univariate Analysis - Texture Mean", ylab = "Frequency", freq = FALSE)
# Evaluation data seem to be even 

hist(perimeter_mean, col = "grey", main = "Univariate Analysis - Perimeter Mean", ylab = "Frequency", freq = FALSE)
# # of projects empoyees worked are higher for 3 & 4 projects

hist(average_montly_hours, col = "grey", main = "Univariate Analysis - Average Monthly Hours", ylab = "Frequency", freq = TRUE)
hist(time_spend_company, col = "grey", main = "Univariate Analysis - Time Spend In Company", ylab = "Frequency", freq = FALSE)
# Many employees joined very recently 2 to 3 years are more

hist(Work_accident, col = "grey", main = "Univariate Analysis - Work Accident", ylab = "Frequency", freq = FALSE)
# Seem to be very less work accident

hist(promotion_last_5years, col = "grey", main = "Univariate Analysis - Promotion Last 5 Years", ylab = "Frequency", freq = TRUE, labels = TRUE)
# very few employees got promotion in last 5 years. Only around 2%

barplot(table(sales))
# There are more Sales, Technical employees in company than others

barplot(table(salary))
# Low and Medium salaries are higher than High


################################################## Bivariate Analysis #######################################################

qplot(y=radius_mean, x= diagnosis, geom = "boxplot")
# Radius Mean seem to be strong predictor

qplot(y=texture_mean, x= diagnosis, geom = "boxplot")
# Satisfaction level seemed to be low for employees who left. This seems to be strong predictor

qplot(y=perimeter_mean, x= diagnosis, geom = "boxplot")
# Wide range for employees who left

qplot(y=area_mean, x= diagnosis, geom = "boxplot")

qplot(y=time_spend_company, x= factor(left), geom = "boxplot")
# Employees who left have more time spend in company

qplot(y=factor(promotion_last_5years), x= factor(left), geom = "boxplot")
# Employees who left have more time spend in company

plot(train_hr$salary, col="wheat")          #Overall employees had less higher than Low/Medium salaries as expected
prop.table(table(train_hr$salary))

scatterplotMatrix(train_hr,diagonal = "density")
scatterplotMatrix(train_hr, diagonal = "qqplot")

###--------------------------------------------------------------------------------------------------------------------###
#  Analyse features influencing predictiveness (feature engineering)                                                     #
#  - Understand presence of missing values and effects with the features                                                 #
#  - Understand which features influence company leaving decision                                                       #
###--------------------------------------------------------------------------------------------------------------------###

missmap(breast_data,main="Missingness MAP Test")  
sapply(breast_data, levels) #Shows that some features are missing data
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
fit.lda <- train(diagnosis~., data=breast_data, method="lda", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.lda

set.seed(999)
fit.glm <- train(diagnosis~., data=breast_data, method="glm", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.glm

set.seed(999)
fit.knn <- train(diagnosis~., data=breast_data, method="knn", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.knn

set.seed(999)
fit.svm <- train(diagnosis~., data=breast_data, method="svmRadial", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.svm

set.seed(999)
fit.cart <- train(diagnosis~., data=breast_data, method="rpart", metric="Accuracy", trControl=tC)
fit.cart

set.seed(999)
fit.rf <- train(diagnosis~., data=breast_data, method="rf", metric="Accuracy", trControl=tC)
fit.rf

set.seed(999)
fit.gbm <- train(diagnosis~., data=breast_data, method="gbm", metric="Accuracy", trControl=tC, verbose=FALSE)
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
confusionMatrix(predict_lda,val_hr_left$left)  # Accuracy=0.819 Kappa=0.518
plot(predict_lda, col="pink", xlab="Predicted Left Status", ylab="Predicted Left", main="Linear - LDA Predictions")

predict_glm <- predict(fit.glm, val_hr_left)
confusionMatrix(predict_glm,val_hr_left$left)  # Accuracy=0.819 Kappa=0.518

predict_knn <- predict(fit.knn, val_hr_left)
confusionMatrix(predict_knn,val_hr_left$left)  # Accuracy=0.809 Kappa=0.497

predict_svm <- predict(fit.svm, val_hr_left)
confusionMatrix(predict_svm,val_hr_left$left)  # Accuracy=0.819 Kappa=0.518
plot(predict_svm, col="purple", xlab="Predicted Loan Status", ylab="Predicted Loans", main="Ensemble - SVM Predictions")


predict_cart <- predict(fit.cart, val_hr_left)
confusionMatrix(predict_cart,val_hr_left$left) # Accuracy=0.819 Kappa=0.518
fancyRpartPlot(fit.cart$finalModel)
plot(varImp(fit.cart),top=20) #Shows Satisfaction level is major criteria to decide the left status

predict_rf <- predict(fit.rf, val_hr_left)
confusionMatrix(predict_rf,val_hr_left$left)   # Accuracy=0.819 Kappa=0.518
plot(varImp(fit.rf),top=20) #Shows Credit History and other features are used as criteria to decide the loan status
plot(predict_rf, col="purple", xlab="Predicted Loan Status", ylab="Predicted Loans", main="Ensemble - RF Predictions")


predict_gbm <- predict(fit.gbm, val_hr_left)
confusionMatrix(predict_gbm,val_hr_left$left)  # Accuracy=0.809 Kappa=0.497
plot(varImp(fit.gbm),top=20) #Shows Credit History and other features are used as criteria to decide the loan status

fit_results <- data.frame(Class = val_hr_left$left)

fit_results$LDA <- predict(fit.lda, val_hr_left, type="prob")
fit_results$CART <- predict(fit.cart, val_hr_left, type="prob")
fit_results$KNN <- predict(fit.knn, val_hr_left, type="prob")
fit_results$RF <- predict(fit.rf, val_hr_left, type="prob")
fit_results$GBM <- predict(fit.gbm, val_hr_left, type="prob")

#Lift Curve

trellis.par.set(caretTheme()) #is working

fit_lift.obj <- lift(Class ~ LDA$N+LDA$Y+CART$N+CART$Y+KNN$N+KNN$Y+RF$N+RF$Y, data = fit_results) #is working 
plot(fit_lift.obj,main="Lift Curve - Probability Threshold : % of Hits", auto.key=list(columns=8,lines=TRUE))

fit_lift.objlda <- lift(Class ~ LDA$N+LDA$Y, data = fit_results) #is working 
plot(fit_lift.objlda,main="Lift Curve - Probability Threshold : % of Hits - LDA", auto.key=list(columns=8,lines=TRUE))

fit_lift.objcart <- lift(Class ~ CART$N+CART$Y, data = fit_results) #is working 
plot(fit_lift.objcart,main="Lift Curve - Probability Threshold : % of Hits - CART", auto.key=list(columns=8,lines=TRUE))

fit_lift.objknn <- lift(Class ~ KNN$N+KNN$Y, data = fit_results) #is working 
plot(fit_lift.objknn,main="Lift Curve - Probability Threshold : % of Hits - KNN", auto.key=list(columns=8,lines=TRUE))

fit_lift.objrf <- lift(Class ~ RF$N+RF$Y, data = fit_results) #is working 
plot(fit_lift.objrf,main="Lift Curve - Probability Threshold : % of Hits - RF", auto.key=list(columns=8,lines=TRUE))

fit_lift.objgbm <- lift(Class ~ GBM$N+GBM$Y, data = fit_results) #is working 
plot(fit_lift.objgbm,main="Lift Curve - Probability Threshold : % of Hits - RF", auto.key=list(columns=8,lines=TRUE))

#Calibration

fit_calib.obj <- calibration(Class ~ LDA$N+LDA$Y+CART$N+CART$Y+KNN$N+KNN$Y+RF$N+RF$Y, data = fit_results, cuts=25) 
plot(fit_calib.obj,main="Calibration Curve - Consistency of Predicted Probability v/s Observed", auto.key=list(columns=8,lines=TRUE)) #is working

fit_calib.objlda <- calibration(Class ~ LDA$N+LDA$Y, data = fit_results, cuts=40) 
plot(fit_calib.objlda,main="Calibration Curve - Consistency of Predicted Probability v/s Observed - LDA", auto.key=list(columns=8,lines=TRUE)) #is working

fit_calib.objcart <- calibration(Class ~ CART$N+CART$Y, data = fit_results, cuts=40) 
plot(fit_calib.objcart,main="Calibration Curve - Consistency of Predicted Probability v/s Observed - CART", auto.key=list(columns=8,lines=TRUE)) #is working

fit_calib.objknn <- calibration(Class ~ KNN$N+KNN$Y, data = fit_results, cuts=40) 
plot(fit_calib.objknn,main="Calibration Curve - Consistency of Predicted Probability v/s Observed - KNN", auto.key=list(columns=8,lines=TRUE)) #is working

fit_calib.objrf <- calibration(Class ~ RF$N+RF$Y, data = fit_results, cuts=40) 
plot(fit_calib.objrf,main="Calibration Curve - Consistency of Predicted Probability v/s Observed - RF", auto.key=list(columns=8,lines=TRUE)) #is working

fit_calib.objgbm <- calibration(Class ~ GBM$N+GBM$Y, data = fit_results, cuts=40) 
plot(fit_calib.objgbm,main="Calibration Curve - Consistency of Predicted Probability v/s Observed - GBM", auto.key=list(columns=8,lines=TRUE)) #is working

g <- ggplot(fit_calib.obj)
g + ggtitle("Calibration Curve - Consistency of Predicted Probability v/s Observed")
