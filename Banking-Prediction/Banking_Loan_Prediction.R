#####================================================================================================================#####
##   Application      : Banking and Finance Loan Prediction                                                             ##
##   ML Problem       : Binary Classification Model                                                                     ##
##   Model Version    : 1.0                                                                                             ##
##   Model Build Date : April 29, 2017                                                                                  ##
##   Team             : Data Diggers                                                                                    ##
##   Organization     : UPX Academy                                                                                     ##
#####================================================================================================================#####
#    Datasets given includes customer transactions with a request for loans and contains customers employment, family    #
#    and income details. Training set has additional information of loan status which test set does not contain.         #
#    The goal is to build a predictive analysis model using the training set and be able to classify loan request as     #
#    either loan approved or loan declined. Based on this classification model, predict the test set loans requests.     #
###--------------------------------------------------------------------------------------------------------------------###
#    Load Required Packages                                                                                              #
###--------------------------------------------------------------------------------------------------------------------###

library(gtools)     # Load Package gtools
library(caret)      # Load Classification And REgression Training package
library(Amelia)     # Load Package for Missingness Map Test
library(rattle)     # Load R Analytic Tool To Learn Easily - GUI Capability
library(rpart.plot) # Load Package for Plotting Recursive Partitions
library(rpart)      # Load Package for Recursive Partitioning
library(car)        # Load Package for Scatterplot Matrix
library(dplyr)      # Load Package dplyr
library(ROCR)       # Load Package ROCR

###--------------------------------------------------------------------------------------------------------------------###
#  Read in the input .csv datasets - Test Loans and Train Loans                                                          #
#  Perform preliminary housekeeping on both input datasets                                                               #
#  - Understand the dataset, Review the Structure, Peek into the data, Summarize the data                                #
#  - Get a snapshot of all the features in the dataset with SAPPLY CLASS                                                 #
###--------------------------------------------------------------------------------------------------------------------###

test_loans <- read.csv("test_loan.csv")
train_loans <- read.csv("train_loan.csv")

dim(train_loans)
str(train_loans)
head(train_loans)
View(head(train_loans))
summary(train_loans)
sapply(train_loans, class)

dim(test_loans)
str(test_loans)
head(test_loans)
View(head(test_loans))
summary(test_loans)
sapply(test_loans, class)

#####================================================================================================================#####
##   Get ahead working with the data                                                                                    ##
##   - Clean the data                                                                                                   ##
##     - Analyse features influencing predictiveness                                                                    ##
##     - Identify if there are any NA's/Missing Data in the features and remove such observations                       ##
##   - Perform EDA                                                                                                      ##
#####================================================================================================================#####
#  Analyse features influencing predictiveness                                                                          ##
#  - Understand how many loans were accepted and how many were declined                                                 ##
#  - Understand loan acceptances/declines by property area                                                              ##
###--------------------------------------------------------------------------------------------------------------------###

train_loans_declined <- filter(train_loans, Loan_Status=="N") #Filter to select loans declined
train_loans_accepted <- filter(train_loans, Loan_Status=="Y") #Filter to select loans accepted

table(train_loans$Loan_Status) %>% barplot(col = "light green", xlab="Loan Status", ylab="No. of Loans", main="Overall Loan Status")
                                   #Seems like 1/3rd loans declined and 2/3rd loans #are accepted 

summary(train_loans_accepted)
summary(train_loans_declined)
dim(train_loans_accepted)
dim(train_loans_declined)
dim(train_loans)

plot(train_loans$Property_Area, col="wheat", main="Loans by Property Area")          
prop.table(table(train_loans$Property_Area))          #Semiurban has higher loans, followed by Urban / Rural in order

plot(train_loans_accepted$Property_Area, col="light pink", main="Accepted Loans by Property Area")  
prop.table(table(train_loans_accepted$Property_Area)) #Semiurban has high acceptance, followed by Urban / Rural in order

plot(train_loans_declined$Property_Area, col="light blue", main="Loans Declined by Property Area") 
prop.table(table(train_loans_declined$Property_Area)) #Semiurban has lower declines, followed by Urban / Rural in order

###--------------------------------------------------------------------------------------------------------------------###
#  Analyse features influencing predictiveness (feature engineering)                                                     #
#  - Understand presence of missing values and effects with the features                                                 #
#  - Understand which features influence loan issuance decisioning                                                       #
###--------------------------------------------------------------------------------------------------------------------###

missmap(test_loans,main="Missingness MAP Test")  #Shows that there are missing values noticed with Credit History, 
                                                 #Amount Term, Loan Amount
missmap(train_loans,main="Missingness MAP Test") #Shows that there are missing values noticed with Credit History, 
                                                 #Loan Amount, Amount Term

sapply(train_loans, levels) #Shows that some features are missing data
sapply(test_loans, levels) #Shows that some features are missing data

###--------------------------------------------------------------------------------------------------------------------###
#  Understand the missing data with the other features, how many are there, how would it influence loan status           #
###--------------------------------------------------------------------------------------------------------------------###

gender_blanks <- filter(train_loans, Gender=="")
head(gender_blanks)
dim(gender_blanks) # 13 Rows exist; Even when Gender is missing, loans are Approved & Declined

married_blanks <- filter(train_loans, Married=="")
head(married_blanks)
dim(married_blanks) # 3 rows exist; Even when Married is missing, loans are Approved

dependents_blanks <- filter(train_loans, Dependents=="")
head(dependents_blanks)
dim(dependents_blanks) # 15 rows exist; Even when Dependents is missing, loans are Approved & Declined

selfemployed_blanks <- filter(train_loans, Self_Employed=="")
head(selfemployed_blanks)
dim(selfemployed_blanks) # 32 rows exist; Even when self employed is missing, loans are Approved & Declined

###--------------------------------------------------------------------------------------------------------------------###
#  LoanID does not have any influence on the Loan Prediction, so removing the feature to build model                     #
#  On the similar note, remove features that do not influence on the Loan Status viz. Gender, Married Status,            #                        
#  Dependents, and Self Employed                                                                                         #                        
###--------------------------------------------------------------------------------------------------------------------###

train_loans_DS <- select(train_loans, -(Loan_ID:Self_Employed))
dim(train_loans_DS)
colnames(train_loans_DS)

###--------------------------------------------------------------------------------------------------------------------###
#  From Missingness Map Test and Levels Test, we notice some features are having missing observations, so                #
#  removing such observations.                                                                                           #
###--------------------------------------------------------------------------------------------------------------------###

train_loans_DS <- filter(train_loans_DS, LoanAmount>=0)
dim(train_loans_DS)
train_loans_DS <- filter(train_loans_DS, Loan_Amount_Term>=0)
dim(train_loans_DS)
train_loans_DS <- filter(train_loans_DS, Credit_History>=0)
dim(train_loans_DS)
View(train_loans_DS)
colnames(train_loans_DS)
missmap(train_loans_DS,main="Missingness MAP Test") #Check for missing data

plot(train_loans_DS, col="blue")

###--------------------------------------------------------------------------------------------------------------------###
#  Similarly Drop the columns that do not influence Loan Issuance on the Test Loans Dataset                              #
###--------------------------------------------------------------------------------------------------------------------###

test_loans_DS <- select(test_loans, -(Loan_ID:Self_Employed))
dim(test_loans_DS)
test_loans_DS <- filter(test_loans_DS, LoanAmount>=0)
dim(test_loans_DS)
test_loans_DS <- filter(test_loans_DS, Loan_Amount_Term>=0)
dim(test_loans_DS)
test_loans_DS <- filter(test_loans_DS, Credit_History>=0)
dim(test_loans_DS)

colnames(test_loans_DS)
missmap(test_loans_DS,main="Missingness MAP Test") #Check for missing data
View(is.na(test_loans_DS))

plot(test_loans_DS, col="blue")

###--------------------------------------------------------------------------------------------------------------------###
#  Performing EDA - Investigage and understand data / data relationships                                                 #
#  - How credit history is influencing loan status                                                                       #
#  - How loan amount is influencing loan status                                                                          #
#  - How applicant's income is influencing loan status                                                                   #
#  - How co-applicant's income is influencing loan status                                                                #
#  - Get the overall picture of the features in comparision to each other with Scatter Plot Matrix                       #
###--------------------------------------------------------------------------------------------------------------------###

ggplot(train_loans_DS, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Credit_History)+ggtitle("Loan Status by Applicant's Credit History")
  #Seems like certain applicant's were issued loans even in the absence of credit history. This number is very very less.

ggplot(train_loans_DS, aes(x=Loan_Status,y=LoanAmount))+geom_boxplot()+ggtitle("Loan Status by Loan Amount")

ggplot(train_loans_DS, aes(x=Loan_Status,y=ApplicantIncome))+geom_boxplot()+ggtitle("Loan Status by Aapplicant income")

ggplot(train_loans_DS, aes(x=Loan_Status,y=CoapplicantIncome))+geom_boxplot()+ggtitle("Loan Status by coapplicant income")

scatterplotMatrix(train_loans_DS,diagonal = "density")

scatterplotMatrix(train_loans_DS, diagonal = "qqplot")

#####================================================================================================================#####
##   Get ahead to build the Predictive Models                                                                           ##
##   - Use 80% of the data to train the models and 20% to validate the models                                           ##
##   - To train the model create 10 fold cross validation dataset                                                       ##
##   - Build the predictive models                                                                                      ##
##   - Apply the model to validate the accuracy with the validation data provided                                       ##
##   - Select the Best Model                                                                                            ##
#####================================================================================================================#####

set.seed(999)
cdp_Index <- createDataPartition(train_loans_DS$Loan_Status, p=0.80, list=FALSE) 
train_dp_loans <- train_loans_DS[cdp_Index,]
val_dp_loans <- train_loans_DS[-cdp_Index,]
dim(val_dp_loans)  
dim(train_dp_loans)  
head(train_dp_loans)

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
fit.lda <- train(Loan_Status~., data=train_dp_loans, method="lda", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.lda

set.seed(999)
fit.glm <- train(Loan_Status~., data=train_dp_loans, method="glm", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.glm

set.seed(999)
fit.knn <- train(Loan_Status~., data=train_dp_loans, method="knn", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.knn

set.seed(999)
fit.svm <- train(Loan_Status~., data=train_dp_loans, method="svmRadial", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.svm

set.seed(999)
fit.cart <- train(Loan_Status~., data=train_dp_loans, method="rpart", metric="Accuracy", trControl=tC)
fit.cart

set.seed(999)
fit.rf <- train(Loan_Status~., data=train_dp_loans, method="rf", metric="Accuracy", trControl=tC)
fit.rf

set.seed(999)
fit.gbm <- train(Loan_Status~., data=train_dp_loans, method="gbm", metric="Accuracy", trControl=tC, verbose=FALSE)
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

predict_lda <- predict(fit.lda, val_dp_loans)
confusionMatrix(predict_lda,val_dp_loans$Loan_Status)  # Accuracy=0.819 Kappa=0.518
plot(predict_svm, col="pink", xlab="Predicted Loan Status", ylab="Predicted Loans", main="Linear - LDA Predictions")

predict_glm <- predict(fit.glm, val_dp_loans)
confusionMatrix(predict_glm,val_dp_loans$Loan_Status)  # Accuracy=0.819 Kappa=0.518

predict_knn <- predict(fit.knn, val_dp_loans)
confusionMatrix(predict_knn,val_dp_loans$Loan_Status)  # Accuracy=0.809 Kappa=0.497

predict_svm <- predict(fit.svm, val_dp_loans)
confusionMatrix(predict_svm,val_dp_loans$Loan_Status)  # Accuracy=0.819 Kappa=0.518
plot(predict_svm, col="purple", xlab="Predicted Loan Status", ylab="Predicted Loans", main="Ensemble - SVM Predictions")


predict_cart <- predict(fit.cart, val_dp_loans)
confusionMatrix(predict_cart,val_dp_loans$Loan_Status) # Accuracy=0.819 Kappa=0.518
fancyRpartPlot(fit.cart$finalModel)
plot(varImp(fit.cart),top=20) #Shows Credit History is major criteria to decide the loan status

predict_rf <- predict(fit.rf, val_dp_loans)
confusionMatrix(predict_rf,val_dp_loans$Loan_Status)   # Accuracy=0.819 Kappa=0.518
plot(varImp(fit.rf),top=20) #Shows Credit History and other features are used as criteria to decide the loan status
plot(predict_rf, col="purple", xlab="Predicted Loan Status", ylab="Predicted Loans", main="Ensemble - RF Predictions")

#--> Code for ROC - Sensitivity / Specificity 
plot(predict_rf, col="orange", xlab="Predicted Loan Status", ylab="Predicted Loans", main="Original Dataset - RF Predictions")
predict_rf
predict_rf_val <- prediction(as.numeric(predict_rf)-1, val_dp_loans$Loan_Status)
perf_rf_VAL_ROC <- performance(predict_rf_val, "tpr", "fpr") 
plot(perf_rf_VAL_ROC, colorize=T, lwd=3, main="ROC Curve - Original Dataset") 
                                         # Graph shows there is cutoff at 0.35; High rate of Sensitivity and continues to achieve 100%.
abline(a=0,b=1)                          # Graph closer to the left side and higher to the top side - indicate higher accuracy
perf_rf_VAL_ROC                          # Slot "x.values": 0.0 0.5 1.0 Slot "y.values": 0.0000000 0.9589041 1.0000000
predict_rf_val
#-->

predict_gbm <- predict(fit.gbm, val_dp_loans)
confusionMatrix(predict_gbm,val_dp_loans$Loan_Status)  # Accuracy=0.809 Kappa=0.497
plot(varImp(fit.gbm),top=20) #Shows Credit History and other features are used as criteria to decide the loan status

fit_results <- data.frame(Class = val_dp_loans$Loan_Status)

fit_results$LDA <- predict(fit.lda, val_dp_loans, type="prob")
fit_results$CART <- predict(fit.cart, val_dp_loans, type="prob")
fit_results$KNN <- predict(fit.knn, val_dp_loans, type="prob")
fit_results$RF <- predict(fit.rf, val_dp_loans, type="prob")
fit_results$GBM <- predict(fit.gbm, val_dp_loans, type="prob")

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

###====================================================================================================================###
#  Make Predictions                                                                                                      #
###====================================================================================================================###
#  - Predictions with the Actual Test Data Not Cleaned                                                                   #
###--------------------------------------------------------------------------------------------------------------------###

predict_lda_TD <- predict(fit.lda, test_loans)
predict_lda_TD
plot(predict_lda_TD, col="wheat")
summary(predict_lda_TD) #N-57;Y-271

predict_glm_TD <- predict(fit.glm, test_loans)
predict_glm_TD
plot(predict_glm_TD, col="wheat")
summary(predict_glm_TD) #N-57;Y-271

predict_knn_TD <- predict(fit.knn, test_loans)
predict_knn_TD
plot(predict_knn_TD, col="wheat")
summary(predict_knn_TD) #N-75;Y-253

predict_svm_TD <- predict(fit.svm, test_loans)
predict_svm_TD
plot(predict_svm_TD, col="wheat")
summary(predict_svm_TD) #N-59;Y-269

predict_cart_TD <- predict(fit.cart, test_loans)
predict_cart_TD
plot(predict_cart_TD, col="wheat")
summary(predict_cart_TD) #N-57;Y-271

predict_rf_TD <- predict(fit.rf, test_loans)
predict_rf_TD
plot(predict_rf_TD, col="wheat")
summary(predict_rf_TD) #N-62;Y-266

predict_gbm_TD <- predict(fit.gbm, test_loans)
predict_gbm_TD
plot(predict_gbm_TD, col="wheat")
summary(predict_gbm_TD) #N-60;Y-268

###--------------------------------------------------------------------------------------------------------------------###
#  - Make predictions on actual dataset with cleaned data and check the results                                          #
###--------------------------------------------------------------------------------------------------------------------###

predict_lda_TD <- predict(fit.lda, test_loans_DS)
predict_lda_TD
plot(predict_lda_TD,col="wheat")
summary(predict_lda_TD) #N-57;Y-271

predict_glm_TD <- predict(fit.glm, test_loans_DS)
predict_glm_TD
plot(predict_glm_TD,col="wheat")
summary(predict_glm_TD) #N-57;Y-271

predict_knn_TD <- predict(fit.knn, test_loans_DS)
predict_knn_TD
plot(predict_knn_TD,col="wheat")
summary(predict_knn_TD) #N-75;Y-253

predict_svm_TD <- predict(fit.svm, test_loans_DS)
predict_svm_TD
plot(predict_svm_TD, col="wheat")
summary(predict_svm_TD) #N-59; Y-269

predict_cart_TD <- predict(fit.cart, test_loans_DS)
predict_cart_TD
plot(predict_cart_TD, col="wheat")
summary(predict_cart_TD) #N-57;Y-271

predict_rf_TD <- predict(fit.rf, test_loans_DS)
predict_rf_TD
plot(predict_rf_TD, col="wheat")
summary(predict_rf_TD) #N-62;Y-266

predict_gbm_TD <- predict(fit.gbm, test_loans_DS)
predict_gbm_TD
plot(predict_gbm_TD, col="wheat")
summary(predict_gbm_TD) #N-60;Y-268

#####================================================================================================================#####
##   Build Models by creating new features : Combined Income : Applicant's Income + Co-Applicant's Income               ##
#####================================================================================================================#####
##   Create the dataset with the new features                                                                           ##
##   Train the Models                                                                                                   ##
##   Validate the Models                                                                                                ##
##   Predict with the Models                                                                                            ##
##   Review Accuracy and Performance                                                                                    ##
#####================================================================================================================#####
#  Creating the Train and Test datasets with the new features                                                            #
###--------------------------------------------------------------------------------------------------------------------###

head(train_loans_DS)
head(test_loans_DS)

train_loans_CDS <- train_loans_DS

train_loans_CDS$CombinedIncome <- train_loans_CDS$ApplicantIncome + train_loans_CDS$CoapplicantIncome

train_loans_CDS <- data.frame(train_loans_CDS["Property_Area"],train_loans_CDS["LoanAmount"],train_loans_CDS["Loan_Amount_Term"],
                              train_loans_CDS["CombinedIncome"],train_loans_CDS["Credit_History"],train_loans_CDS["Loan_Status"])

colnames(train_loans_CDS)
head(train_loans_CDS)
View(is.na(train_loans_CDS))

test_loans_CDS <- test_loans_DS

test_loans_CDS$CombinedIncome <- test_loans_CDS$ApplicantIncome + test_loans_CDS$CoapplicantIncome

test_loans_CDS <- data.frame(test_loans_CDS["Property_Area"],test_loans_CDS["LoanAmount"],test_loans_CDS["Loan_Amount_Term"],
                              test_loans_CDS["CombinedIncome"],test_loans_CDS["Credit_History"])

colnames(test_loans_CDS)
head(test_loans_CDS)
View(is.na(test_loans_CDS))

###--------------------------------------------------------------------------------------------------------------------###
#  Split the train dataset for training and validation                                                                   #
###--------------------------------------------------------------------------------------------------------------------###

set.seed(999)
cdp_Index1 <- createDataPartition(train_loans_CDS$Loan_Status, p=0.80, list=FALSE) 
train_cds_dp_loans <- train_loans_CDS[cdp_Index1,]
val_cds_dp_loans <- train_loans_CDS[-cdp_Index1,]
dim(val_cds_dp_loans)  
dim(train_cds_dp_loans)  
head(train_cds_dp_loans)

###--------------------------------------------------------------------------------------------------------------------###
#  Planning to run algorithms using 10-fold cross validation type of resampling                                          #
###--------------------------------------------------------------------------------------------------------------------###

tC <- trainControl(method="cv", number=10)

###--------------------------------------------------------------------------------------------------------------------###
#  Build the Models                                                                                                      #
#   - Linear Methods     - LDA, GLM                                                                                      #
#   - Non Linear Methods - kNN, SVM                                                                                      #
#   - Tree Methods       - CART                                                                                          #
#   - Ensemble Methods   - RF, GBM                                                                                       #
###--------------------------------------------------------------------------------------------------------------------###

set.seed(999)
fit.lda_CDS <- train(Loan_Status~., data=train_loans_CDS, method="lda", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.lda_CDS # Accuracy 0.8147 Kappa 0.4867

set.seed(999)
fit.glm_CDS <- train(Loan_Status~., data=train_loans_CDS, method="glm", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.glm_CDS # Accuracy 0.8147 Kappa 0.4867

set.seed(999)
fit.knn_CDS <- train(Loan_Status~., data=train_loans_CDS, method="knn", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.knn_CDS # Accuracy 0.7977 Kappa 0.4531

set.seed(999)
fit.svm_CDS <- train(Loan_Status~., data=train_loans_CDS, method="svmRadial", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.svm_CDS # Accuracy 0.8071 Kappa 0.4624

set.seed(999)
fit.cart_CDS <- train(Loan_Status~., data=train_loans_CDS, method="rpart", metric="Accuracy", trControl=tC)
fit.cart_CDS # Accuracy 0.8052 Kappa 0.4694

set.seed(999)
fit.rf_CDS <- train(Loan_Status~., data=train_loans_CDS, method="rf", metric="Accuracy", trControl=tC)
fit.rf_CDS # Accuracy 0.8091128 Kappa 0.4800

set.seed(999)
fit.gbm_CDS <- train(Loan_Status~., data=train_loans_CDS, method="gbm", metric="Accuracy", trControl=tC, verbose=FALSE)
fit.gbm_CDS # Accuracy 0.8204349 and Kappa 0.5222773

###--------------------------------------------------------------------------------------------------------------------###
#  Validate the Models                                                                                                   #
###--------------------------------------------------------------------------------------------------------------------###

predict_lda_CDS <- predict(fit.lda_CDS, val_cds_dp_loans)
confusionMatrix(predict_lda_CDS,val_cds_dp_loans$Loan_Status) # Accuracy=0.819 Kappa=0.518 

predict_glm_CDS <- predict(fit.glm_CDS, val_cds_dp_loans)
confusionMatrix(predict_glm_CDS,val_cds_dp_loans$Loan_Status) # Accuracy=0.819 Kappa=0.518 

predict_knn_CDS <- predict(fit.knn_CDS, val_cds_dp_loans)
confusionMatrix(predict_knn_CDS,val_cds_dp_loans$Loan_Status) # Accuracy=0.819 Kappa=0.5274

predict_svm_CDS <- predict(fit.svm_CDS, val_cds_dp_loans)
confusionMatrix(predict_svm_CDS,val_cds_dp_loans$Loan_Status) # Accuracy=0.8286 Kappa=0.5478
plot(predict_svm_CDS, col="purple", xlab="Predicted Loan Status", ylab="Predicted Loans", main="Ensemble - RF Predictions")

predict_cart_CDS <- predict(fit.cart_CDS, val_cds_dp_loans)
confusionMatrix(predict_cart_CDS,val_cds_dp_loans$Loan_Status) # Accuracy=0.819 Kappa=0.518 
fancyRpartPlot(fit.cart_CDS$finalModel)
plot(varImp(fit.cart_CDS),top=20) #Shows Credit History is major criteria to decide the loan status

predict_rf_CDS <- predict(fit.rf_CDS, val_cds_dp_loans)
confusionMatrix(predict_rf_CDS,val_cds_dp_loans$Loan_Status) # Accuracy=0.8762 Kappa=0.6828
plot(varImp(fit.rf_CDS),top=20) #Shows Credit History and other features are used as criteria to decide the loan status
plot(predict_rf_CDS, col="orange", xlab="Predicted Loan Status", ylab="Predicted Loans", main="Ensemble - RF Predictions")

#--> Code for ROC Begins Here
plot(predict_rf_CDS, col="orange", xlab="Predicted Loan Status", ylab="Predicted Loans", main="Derived Dataset - RF Predictions")

predict_rf_val <- prediction(as.numeric(predict_rf_CDS)-1, val_cds_dp_loans$Loan_Status)
perf_rf_val_ROC <- performance(predict_rf_val, "tpr", "fpr") 
plot(perf_rf_val_ROC, colorize=T, lwd=3, main="ROC Curve - Derived Dataset") 
                                         # Graph shows there is cutoff at 0.34; High rate of Sensitivity and continues to achieve 100%.
abline(a=0,b=1)                          # Closer to the left side and higher to the top side show higher accuracy.
perf_rf_val_ROC                          # Slot "x.values": 0.00000 0.34375 1.00000 Slot "y.values": 0.0000000 0.9726027 1.0000000

perf_rf_CDS_ACC <- performance(predict_rf_CDS, "acc") #measure accuracy performance
plot(perf_rf_CDS_ACC,col="red",lwd=3) # Shows high positive rate
plot(perf_rf_CDS_ACC, avg="vertical", col="red", spread.estimate="boxplot", show.spread.at=seq(0.1,0.9,by=0.1))

perf_rf_CDS_senspec <- performance(predict_rf_CDS, measure = "sens", x.measure = "spec")
perf_rf_CDS_senspec <- performance(predict_rf_CDS, "sens", "spec")
plot(perf_rf_CDS_senspec, colorize=T, lwd=3) # Graph shows there is cutoff at 0.68; High rate of Sensitivity;
#--> Code for ROC Ends Here

predict_gbm_CDS <- predict(fit.gbm_CDS, val_cds_dp_loans)
confusionMatrix(predict_gbm_CDS,val_cds_dp_loans$Loan_Status) # Accuracy=0.8381 Kappa=0.5771
plot(varImp(fit.gbm_CDS),top=20) #Shows Credit History and other features are used as criteria to decide the loan status
plot(predict_gbm_CDS, col="light green", xlab="Predicted Loan Status", ylab="Predicted Loans", main="Ensemble - GBM Predictions")

###--------------------------------------------------------------------------------------------------------------------###
#  Predict the Loan Issuance for the Test Loans Dataset                                                                  #
###--------------------------------------------------------------------------------------------------------------------###

predict_lda_CDS_TD <- predict(fit.lda_CDS, test_loans_CDS)
plot(predict_lda_CDS_TD, col="wheat")
summary(predict_lda_CDS_TD) #N-57;Y-271

predict_glm_CDS_TD <- predict(fit.glm_CDS, test_loans_CDS)
plot(predict_glm_CDS_TD, col="wheat")
summary(predict_glm_CDS_TD) #N-57;Y-271

predict_knn_CDS_TD <- predict(fit.knn_CDS, test_loans_CDS)
plot(predict_knn_CDS_TD, col="wheat")
summary(predict_knn_CDS_TD) #N-66;Y-262

predict_svm_CDS_TD <- predict(fit.svm_CDS, test_loans_CDS)
plot(predict_svm_CDS_TD, col="wheat")
summary(predict_svm_CDS_TD) #N-57; Y-271

predict_cart_CDS_TD <- predict(fit.cart_CDS, test_loans_CDS)
plot(predict_cart_CDS_TD, col="wheat")
summary(predict_cart_CDS_TD) #N-57;Y-271

predict_rf_CDS_TD <- predict(fit.rf_CDS, test_loans_CDS)
plot(predict_rf_CDS_TD, col="wheat")
summary(predict_rf_CDS_TD) #N-63;Y-265

predict_gbm_CDS_TD <- predict(fit.gbm_CDS, test_loans_CDS)
plot(predict_gbm_CDS_TD, col="light blue", xlab="Predicted Loan Status", ylab="Predicted Loans", main="Ensemble - GBM Predictions")
summary(predict_gbm_CDS_TD) #N-64;Y-264
#####================================================================================================================#####
##   Conclusion                                                                                                         ##
##                                                                                                                      ##
##   - With the Original features                                                                                       ##
##     RF (Random Forest) model has shown the best accuracy                                                             ##
##     On the training data   : Accuracy 0.8762 and Kappa 0.6828                                                        ##
##                                                                                                                      ##
##     RF (Random Forest) model and SVM (Simple Vector Machines) has shown the best accuracy                            ##
##     On the validation data : Accuracy 0.819 and Kappa 0.518                                                          ##
##                                                                                                                      ##
##   - With the addition new feature by applying the banking principle to consider combined income for Loan Issuance    ##
##     GBM (Generalized Boosting Method) model has shown the improved accuracy                                          ##
##     On the test data       : Accuracy 0.8204 and Kappa 0.5222                                                        ##
##     On the validation data : Accuracy 0.8381 and Kappa 0.5771                                                        ##
#####================================================================================================================#####
