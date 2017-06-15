###====================================================================================================================###
#  Application      : Nike Revenue Forecast                                                                              #
#  ML Problem       : Timeseries Forecasting Model                                                                       #
#  Model Version    : 1.0                                                                                                #
#  Model Build Date : April 29, 2017                                                                                     #
#  Team             : Data Diggers                                                                                       #
#  Organization     : UPX Academy                                                                                        #
###====================================================================================================================###
#  Load Required Packages                                                                                                #
###--------------------------------------------------------------------------------------------------------------------###

library(dplyr)
library(caret)
library(readxl)
library(forecast)
library(TTR)
library(Metrics)
library(tseries)
library("astsa")

###--------------------------------------------------------------------------------------------------------------------###
#  Read in the input dataset                                                                                             #
#  - Review for its correctness                                                                                          #
#  - If not found OK, create proper dataset and format it as required                                                    #
#  - Clean the data, so that it can used as an Time Series Object                                                        #
###--------------------------------------------------------------------------------------------------------------------###

nike_revenue <- read_excel("Nike_revenue.xlsx",sheet=1,col_names = TRUE, col_types = NULL)
dim(nike_revenue)
View(nike_revenue) #Seems like both train / validation data is provided in the same dataset and some columns name are missing 
nike_data_colnames <- c("Year","Aug-31","Nov-30","Feb-28","May-31","Year","Aug-31","Nov-30","Feb-28","May-31")
colnames(nike_revenue) <- nike_data_colnames #Apply Column names to the dataset
View(nike_revenue)
str(nike_revenue)

train_nike_revenue <- nike_revenue[,1:5] #Splitting source data into train dataset
View(train_nike_revenue)
str(train_nike_revenue)
train_nike_revenue

train_nike_revenue_RC <- train_nike_revenue

val_nike_revenue <- nike_revenue[,6:10] #Splitting source data into validation dataset
View(val_nike_revenue)
str(val_nike_revenue) #Seems like it contains missing data in the rows; Also only first row seems to be valid
val_nike_revenue <- head(val_nike_revenue, 1) #selecting rows containing valid data
val_nike_revenue$Year <- as.numeric(val_nike_revenue$Year) #converting to valid column type
val_nike_revenue[,2] <- as.numeric(val_nike_revenue[,2]) #converting to valid column type
View(val_nike_revenue)
str(val_nike_revenue)

val_nike_revenue_RC <- val_nike_revenue

###====================================================================================================================###
#  Reorganize the quarterly data in Dataframe to vectorize, so that a Time Series Object can be created                  #
#  - Convert every row of a dataframe into row of vector                                                                 #
#  - Concatenate all the individual row vectors into a single continuous vector                                          #
#  - Create a Time Series Object with the vectorized row data as a quarterly frequency                                   #
###====================================================================================================================###
#  Convert every row of a dataframe into row of vector                                                                   #
###--------------------------------------------------------------------------------------------------------------------###

tsrdr1 <- as.numeric(train_nike_revenue_RC[1,2:5])
tsrdr2 <- as.numeric(train_nike_revenue_RC[2,2:5])    
tsrdr3 <- as.numeric(train_nike_revenue_RC[3,2:5])
tsrdr4 <- as.numeric(train_nike_revenue_RC[4,2:5])
tsrdr5 <- as.numeric(train_nike_revenue_RC[5,2:5])
tsrdr6 <- as.numeric(train_nike_revenue_RC[6,2:5])
tsrdr7 <- as.numeric(train_nike_revenue_RC[7,2:5])
tsrdr8 <- as.numeric(train_nike_revenue_RC[8,2:5])
tsrdr9 <- as.numeric(train_nike_revenue_RC[9,2:5])
tsrdr10 <- as.numeric(train_nike_revenue_RC[10,2:5])


###--------------------------------------------------------------------------------------------------------------------###
#  Concatenate all the individual row vectors into a single continuous vector                                            #
###--------------------------------------------------------------------------------------------------------------------###

tsrd <- as.numeric(c(tsrdr1,tsrdr2,tsrdr3,tsrdr4,tsrdr5,tsrdr6,tsrdr7,tsrdr8,tsrdr9,tsrdr10))
class(tsrd)
dim(tsrd)
tsrd  

###--------------------------------------------------------------------------------------------------------------------###
#  Create a Time Series Object with the vectorized row data as a quarterly frequency                                     #
###--------------------------------------------------------------------------------------------------------------------###

tsrd_ts <- ts(tsrd, start=1998+2/4, frequency = 4)
tsrd_ts
plot(tsrd_ts)
summary(tsrd_ts)
boxplot(tsrd_ts)

class(tsrd_ts)  # Test to ensure if object created properly as a Time Series Object
length(tsrd_ts) # Test to ensure if object created properly as a Time Series Object
tsrd_ts[40]     # Test to ensure if object created properly as a Time Series Object
dim(tsrd_ts)    # Test to ensure if object created properly as a Time Series Object
nrow(tsrd_ts)   # Test to ensure if object created properly as a Time Series Object
ncol(tsrd_ts)   # Test to ensure if object created properly as a Time Series Object
cycle(tsrd_ts)  # Test to ensure if object created properly as a Time Series Object
time(tsrd_ts)   # Test to ensure if object created properly as a Time Series Object

###--------------------------------------------------------------------------------------------------------------------###
#  Similarly Create Validation Time Series Object with the vectorized row data as a quarterly frequency                  #
###--------------------------------------------------------------------------------------------------------------------###

tsrdrv <- as.numeric(val_nike_revenue_RC[1,2:5])
tsrd_val_ts <- ts(tsrdrv, start=2008+2/4, frequency = 4)
tsrd_val_ts
val_nike_revenue <- tsrd_val_ts

#####================================================================================================================#####
##  Address the case questions                                                                                          ##
##  - Plot the Time Series                                                                                              ##
##  - Time Series Components and Interpretation                                                                         ##
##    - Decompose the Time Series to identify its elements                                                              ##
##    - Interprets the Time Series Data                                                                                 ##
#####================================================================================================================#####
#   - Plot the Time Series                                                                                               #
###--------------------------------------------------------------------------------------------------------------------###

train_nike_revenue <- tsrd_ts

plot.ts(train_nike_revenue, ylab="Quarterly Revenue",main="Nike's Revenue", col="blue", lwd=2) 
                                                                            #plot the time series
                                                                            #Notice increasing revenue shows trend
                                                                            #Cyclical patterns gets marginally wider 
                                                                            #shows marginal variance

###--------------------------------------------------------------------------------------------------------------------###
#   - Decompose the Time Series to identify its elements                                                                 #
###--------------------------------------------------------------------------------------------------------------------###

dts_nr <- decompose(train_nike_revenue) # Decompose time series to identify the elements of time series

plot(dts_nr,col="red", lwd=2)
dts_nr$seasonal   # shows constant in seasonality
dts_nr$trend      # shows variance with trend
dts_nr$random     # shows presence of randomness

boxplot(train_nike_revenue~cycle(train_nike_revenue),border=c(1:4),ylab="Quarterly Revenue", xlab="Cycle", main="Nike Revenue-Box Plot") 
                                                      #-> Variance and Mean higher during November end than the other months

sdc <- stl(train_nike_revenue, s.window="periodic")   # Seasonal Trend Decomposition 
sdc
plot(sdc, col="red", lwd=2) #Plot revenue time series into three components - seasonal, trend and remainder
monthplot(sdc, col="red")

###--------------------------------------------------------------------------------------------------------------------###
#   - Interpret Time Series                                                                                              #
###--------------------------------------------------------------------------------------------------------------------###
#     From the plots, we notice that the time series has Seasonality, Trend and Randomness. Also we notice that the      #
#     random fluctuations are roughly constant in size over time. So this data can be describes as Additive Model.       #
#     Seasonality - We notice a repetition pattern periodically over each fiscal year with high revenues in the          #
#                   beginning of fiscal year and then revenues fall in the next two quarter and rises up until 1st       #
#                   quarter of the next fiscal year.                                                                     #
#     Trend       - We notice a constant pattern of gradual rise an upward pattern in revenues                           #
#     Random      - We notice traces of Randomness with significant high and low spikes along the time period            #
#                                                                                                                        #
#     a. The mean of the time series does not seem to be constant with time. The mean is changing along the time.        #
#     b. The variance of the series seems roughly constant along the time.                                               #
#     c. The covariance of the series seems marginally increases along the time.                                         #
#                                                                                                                        #
#     From the above (a, b, c)  it looks the series is non-stationary time series. Need to validate this.                #
###--------------------------------------------------------------------------------------------------------------------###

acf2(train_nike_revenue)  # Validation for Non-Stationary Time Series, ACF is above threshold and has not decayed. 

adf.test(train_nike_revenue, alternative = "stationary")  # Dicky Fuller Test. 

###--------------------------------------------------------------------------------------------------------------------###
#  ACF plots shows significant lags above threshold. Clearly the series is non-stationary as a high number of previous   #
#  observations are correlated with future values. There are many lags above the significant threshold line.             # 
#                                                                                                                        #     
#  Non-Stationary Time Series can be transformed into a Stationary Time Series by Differencing. Applying diff function.  #
#  Consider 1St Level Differencing if not does satisfy to be Stationary Series, then consider 2nd Level Differencing.    #
###--------------------------------------------------------------------------------------------------------------------###
#  Note1: Since the Nike Revenue TS is non-stationary; we will make it stationarised before fitting ARIMA model, as      # 
#         stationarity is the requirement for ARIMA model.                                                               #
#  Note2: Exponential Smoothing models require that the forecast errors (residuals) are uncorrelated and are normally    #
#         distributed with mean zero and constant variance; and it makes no assumptions about the correlations between   #
#         successive values of the time series (i.e. auto-correlation)                                                   #
###--------------------------------------------------------------------------------------------------------------------###


#####================================================================================================================#####
##   Part - I : Regression : y=c+mx+e. Where Intercept is c; Slope is m; and Random Error is e                          ##
#####================================================================================================================#####
##           - Perform Time Series Linear Regression - tslm                                                             ##
##           - Review Time Series Linear Regression                                                                     ##
##           - Calculate the RMSE for the Training Set                                                                  ##
##           - Predict / Forecast for the Validation Set                                                                ##
##           - Discuss Forecast - Reasonability                                                                         ##
##           - Calculate the RMSE for the Validation Set                                                                ##
##           - Predict / Forecast for Nike Revenues in Year 2010                                                        ##
#####================================================================================================================#####

plot(train_nike_revenue,col="blue",lwd=2) # Plot time series
abline(reg=lm(train_nike_revenue~time(train_nike_revenue)), col="red",lwd=2)

nr_ts_lm <- tslm(train_nike_revenue ~ trend + season) # Perform the Linear Regression of the Time Series
class(nr_ts_lm)                                       # Check the class of the model

###--------------------------------------------------------------------------------------------------------------------###
#  Review Time Series Linear Regression                                                                                  #
###--------------------------------------------------------------------------------------------------------------------###

par(mfrow=c(2,2))                                     # Set canvas to graph by 2 Rows and 2 Columns
plot(nr_ts_lm)                                        # Plot the model for its results
par(mfrow=c(1,1))

summary(nr_ts_lm)                                     # Review the summary the time series model
train_nike_revenue                                    # This is actual values
nr_ts_lm$fitted.values                                # This is predicted values
nr_ts_lm$residuals                                    # This is residuals => Actual Values - Predicted Values

par(mfrow=c(3,1))                                     # Canvas to accommodate graphs in 3 Rows of 1 Column
plot(train_nike_revenue)                              # Plot of Actual Values
plot(nr_ts_lm$fitted.values)                          # Plot of Predicted Values
plot(nr_ts_lm$residuals)                              # Plot of Residuals and varies between +400 to -400 over period

par(mfrow=c(2,1))                                     # Plot for residual analysis
acf(nr_ts_lm$residuals)                               # Seems there several significant lags above threshold
                                                      # Also zeroes down at lag 2.25
pacf(nr_ts_lm$residuals)                              # Seems few significant lags above threshold
                                                      # Also zeroes down at multiple lag 1.75 and 3.75
par(mfrow=c(1,1))
acf2(nr_ts_lm$residuals)                              # From the correlogram it is quite evident that there is significant 
                                                      # evidence of non-zero correlations at various lags

###--------------------------------------------------------------------------------------------------------------------###
#  Calculate RMSE on the Training Set                                                                                    #
###--------------------------------------------------------------------------------------------------------------------###

rmse(train_nike_revenue,nr_ts_lm$fitted.values)       # RMSE of Training Set with Linear Regression Model = 222.1858

###--------------------------------------------------------------------------------------------------------------------###
#  Predict / Forecast for the Validation Set                                                                             #
###--------------------------------------------------------------------------------------------------------------------###

nr_fclm_2009 <- forecast.lm(nr_ts_lm, h=4, level=c(75,80,85,90,95), biasadj = TRUE, ts=TRUE)
summary(nr_fclm_2009)

#->  Predicted results for validation period - 2009
#->          Point Forecast    Lo 75    Hi 75    Lo 80    Hi 80    Lo 85    Hi 85    Lo 90    Hi 90    Lo 95    Hi 95
#->  2008 Q3         4738.3 4434.989 5041.611 4399.609 5076.991 4356.664 5119.936 4300.207 5176.393 4211.909 5264.691
#->  2008 Q4         4367.8 4064.489 4671.111 4029.109 4706.491 3986.164 4749.436 3929.707 4805.893 3841.409 4894.191
#->  2009 Q1         4449.8 4146.489 4753.111 4111.109 4788.491 4068.164 4831.436 4011.707 4887.893 3923.409 4976.191
#->  2009 Q2         4832.1 4528.789 5135.411 4493.409 5170.791 4450.464 5213.736 4394.007 5270.193 4305.709 5358.491

#->  Review Results with the Validation Data
#->  Year            Actual
#->  2008 Q3           5432
#->  2008 Q4           4590
#->  2009 Q1           4440 
#->  2009 Q2           4713     

#->  From the above forecast seems to reasonable at Hi 85 Confidence Intervals and forecast seems satisfactory

plot(nr_fclm_2009)                                # Plot the forecast for the validation period
lines(nr_fclm_2009$fitted, lwd=2, col="green")    # Plot the fitted values of the forecast 2009
lines(val_nike_revenue, col="red")                # Plot to see actual revenues for year 2009

par(mfrow=c(2,1))
plot(nr_fclm_2009, val_nike_revenue, type = "l")
lines(val_nike_revenue, col="red")
plot(nr_fclm_2009, val_nike_revenue, type = "h")
lines(val_nike_revenue, col="red")
par(mfrow=c(1,1))

nr_fclm_2009$mean
nr_fclm_2009$x
nr_fclm_2009$residuals
nr_fclm_2009$fitted

###--------------------------------------------------------------------------------------------------------------------###
#  Calculate RMSE on the Validation Set                                                                                  #
###--------------------------------------------------------------------------------------------------------------------###

rmse(val_nike_revenue, nr_fclm_2009$mean)             # RMSE of Validation Set = 369.0777

# Note: Since RMSE for training dataset (222.1858) is much lower compare to RMSE for Validation dataset (369.0777)
#       Relatively, this shows the predictive power of linear regression model is not as satisfactory

###--------------------------------------------------------------------------------------------------------------------###
#  Predict / Forecast for Nike Revenues in Year 2010                                                                     #
###--------------------------------------------------------------------------------------------------------------------###

nr_fclm_2010 <- forecast.lm(nr_ts_lm, h=8, level=c(75,80,85,90,95), biasadj = TRUE, ts=TRUE)
nr_fclm_2010
summary(nr_fclm_2010)

par(mfrow=c(2,1))
plot(nr_fclm_2010, type = "l")
plot(nr_fclm_2010, type = "h")
par(mfrow=c(1,1))

###--------------------------------------------------------------------------------------------------------------------###
#    Inference: From the above forecast results including ACF measures and considering residuals behaviour on this       #
#               time series data, Regression would not be appropriate best model to forecast as data needs further       #
#               normalization i.e. much info is left in the residuals.                                                   #
###--------------------------------------------------------------------------------------------------------------------###


#####================================================================================================================#####
##   Part - II : Smoothing Methods                                                                                      ##
##   - Identify Smoothing Method for Nike's Revenue Forecasting                                                         ##
##   - Discuss over the selection of Smoothing Method                                                                   ##
#####================================================================================================================#####

#-> From the above Time Series Decompose Plot and STL Plot we have noticed and concluded that there is Seasonal Component, 
#-> Trend Component and Randomness Component in the Nike Revenue Data. Since all the three components are present in the
#-> dataset, we feel Triple Exponential Smoothing Method (TES) is appropriate to consider. TES gives lot of flexibility
#-> in controlling the Smoothing, Trend and Seasonality. 
#-> Double Exponential Smoothing Method (DES) does not perform well when Trend and Seasonality are present in the dataset.
#-> However will validate the assumptions by fitting with SES and DES. We will estimate smoothing parameter alpha and beta
#-> respectively using the given data.

###--------------------------------------------------------------------------------------------------------------------###
#  Simple Exponential Smoothing (SES)                                                                                    #
###--------------------------------------------------------------------------------------------------------------------###

nr_ses <- HoltWinters(train_nike_revenue, beta=FALSE, gamma=FALSE)
nr_ses

#-> Note from above results that alpha is over 0.6 (close to 1) implies that forecasts are largely based on recent 
#-> than historical (less recent) values. Thus more recent values have more weight than older values.

plot(nr_ses, main = "Holt-Winters - Single Exponential Smoothing", lwd=2)

#-> Let's check the accuracy of forecast

nr_ses$SSE  #-> SSE is 3652701

#-> Let's forecast for next 8 quarters
nr_ses_f <- forecast.HoltWinters(nr_ses, h=8)
nr_ses_f
plot.forecast(nr_ses_f, main = "Forecast from Holt-Winters SES")

#-> In above SES forecast, revenue values forecasted for fiscal year 2009 and 2010 i.e. Q3-2009 to Q2-2010. Note in above
#-> plot forecasted value are blue linewithout any trend and seasonality in it; dark shaded are denotes 80% prediction 
#-> interval and light shaded area extended to 95% of prediction interval.

#-> Let's do residuals analysis 
nr_ses_f$residuals

#-> Residuals are only for insample i.e. actual time period of data. It is the difference of actuals and predicted values
nr_ses_f$residuals<-na.omit(nr_ses_f$residuals) # Remove NA from residuals
acf(nr_ses_f$residuals)
acf(nr_ses_f$residuals, lag.max = 16) # Correlogram of insample errors for 1-4 lags
                                      # From the correlogram it is quite clear that there is significant evidence of 
                                      # non-zero correlations at various lags

Box.test(nr_ses_f$residuals, lag=16, type="Ljung-Box") # To test and confirm that data has significant non-zero 
                                                       # autocorrelations with a Ljung-Box test. 

#->                  Box-Ljung test
#-> data:  nr_ses_f$residuals
#-> X-squared = 148.79, df = 16, p-value < 2.2e-16

#-> From Ljung-Box test resuts above, the p value much lower than 0.05, indicates evidence of non-zero correlations 
#-> It is evident from ACF functions and Ljung-box test that residuals have lots of information left, hence it can be
#-> concluded that SES could likely be improved upon by another forecasting technique. Lets verify fitting DES.

###--------------------------------------------------------------------------------------------------------------------###
#  Double Exponential Smoothing (DES)                                                                                    #
###--------------------------------------------------------------------------------------------------------------------###

nr_des <- HoltWinters(train_nike_revenue, gamma=FALSE)
nr_des

#-> The high estimated value of alpha (1) implies that forecast is dependent on most recent values of time series; however 
#-> small value of beta (0.22) implies that slope of the trend component is less dependent on recent values and instead is  
#-> much dependent on historical values

plot(nr_des)

#-> Let's check the acuracy of forecast

nr_des$SSE # SSE is 5137920

#-> Let's forecast for next 8 quarters
nr_des_f <- forecast.HoltWinters(nr_des, h=8)
nr_des_f
plot.forecast(nr_des_f, main = "Forecast from Holt-Winters DES")

#-> From the plot of DES Forecast, notice that the forecasted value (blue line) is clearly seen with increasing trend
#-> Dark shaded area denotes 80% prediction interval and light shaded area extends to 95% of prediction interval.

#-> Let's do residuals analysis
nr_des_f$residuals

#-> Residuals are only for insample i.e. actual time period of data. It is the difference of actuals and predicted values
nr_des_f$residuals<-na.omit(nr_des_f$residuals) # Remove NA from residuals
acf2(nr_des_f$residuals, max.lag=16, lwd=2, col="red")  # Correlogram of insample errors for 1-4 lags
                                                        # From the correlogram it is quite clear that there is significant 
                                                        # evidence of non-zero correlations at various lags

Box.test(nr_des_f$residuals, lag=16, type="Ljung-Box") # To test and confirm if the data has significant non-zero 
                                                       # autocorrelation with a Ljung-Box test. 

#->                 Box-Ljung test
#-> data:  nr_des_f$residuals
#-> X-squared = 116.01, df = 16, p-value < 2.2e-16

#-> With Ljung-Box test above, the p value much lower than 0.05 indicates the evidence of non-zero autocorrelations
#-> It is evident from ACF Function and Ljung-box test that residuals have lots of information left, hence it can be 
#-> concluded that DES could likely be improved upon by another forecasting technique. Lets verify fitting TES.

###--------------------------------------------------------------------------------------------------------------------###
#  Triple Exponential Smoothing                                                                                          #
###--------------------------------------------------------------------------------------------------------------------###

nr_TES <- HoltWinters(train_nike_revenue)
nr_TES

#->                 HoltWinters(x = train_nike_revenue)
#-> Smoothing parameters:
#-> alpha: 0.3775294
#-> beta : 0.6180435
#-> gamma: 1

#-> The estimated values of alpha (0.38) implies the foreast is based upon both recent observations and more distant past
#-> values, however non-recent values has more weight; beta (0.62) implies that the slope b of the trend component, are 
#-> based largely upon very recent observations; while the value of gamma (1) indicating that the estimate of the seasonal
#-> component at the current time point is just based upon very recent observations.

nr_TES$SSE   # SSE is 310892.9; This value is relatively smaller as compared to SES$SSE (3652701) and DES$SSE (5137920)

plot(nr_TES) # Notice the fitted (forecasted) values are close to the actual data

#-> Forecast for the next 8 quarters i.e. Fiscal Year 2009 and Fiscal Year 2010

nr_TES_F <- forecast.HoltWinters(nr_TES, h=8) 

plot.forecast(nr_TES_F, col="red",lwd=2, main = "HoltWinter Forecast FY 2009-2010") 
#-> In above TES forecast, Note the plot of forecasted values in the blue line and clear evidence of trend and seasonality
#-> are incorporated in the forecast;

nr_TES_F # Review results

#->  Predicted results for validation period - 2009
#->    Point       Forecast Lo 80    Hi 80    Lo 95    Hi 95
#->  2008 Q3       5357.145 5239.162 5475.128 5176.706 5537.584
#->  2008 Q4       5074.670 4936.416 5212.924 4863.229 5286.111
#->  2009 Q1       5284.931 5114.537 5455.325 5024.335 5545.527
#->  2009 Q2       5815.746 5603.152 6028.339 5490.612 6140.879
#->  2009 Q3       6084.891 5773.082 6396.699 5608.020 6561.761
#->  2009 Q4       5802.416 5441.284 6163.548 5250.112 6354.720
#->  2010 Q1       6012.677 5595.068 6430.285 5373.999 6651.354
#->  2010 Q2       6543.492 6063.198 7023.785 5808.946 7278.037

#->  Review Results with the Validation Data
#->  Year          Actual
#->  2008 Q3       5432
#->  2008 Q4       4590
#->  2009 Q1       4440 
#->  2009 Q2       4713     

#-> Let's do residuals analysis
nr_TES_F$residuals

#-> Residuals are only for insample i.e. actual time period of data. It is the difference of actuals and predicted values
nr_TES_F$residuals <- na.omit(nr_TES_F$residuals) # Remove NA from residuals
acf2(nr_TES_F$residuals,max.lag = 16) # Correlogram of insample errors for 1-4 lags
                                      # From the correlogram it is quite clear that there is no significant evidence of 
                                      # non-zero correlations at various lags


Box.test(nr_TES_F$residuals, lag=16, type="Ljung-Box") # To test and confirm if the data has significant non-zero 
                                                       # autocorrelation with a Ljung-Box test. 

#->                 Box-Ljung test
#-> data:  nr_TES_F$residuals
#-> X-squared = 25.725, df = 16, p-value = 0.05802

#-> With Ljung-Box test above, the p value is higher than 0.05 indicates the evidence of no non-zero autocorrelations
#-> It is evident from ACF Function and Ljung-box test that residuals not much of information left, hence it can be 
#-> concluded that TES is likely the better forecasting technique. 

#-> Further Residuals Analysis: In order to check whether the forecast errors have constant variance over time, and are          #
#-> normally distributed with mean zero and constant variance, make time series plot of forecast errors and a histogram                    # 
#-> (with overlaid normal curve):                                                                                         #

#-> Plot residual plot to check if residuals have constant variance over time

plot.ts(nr_TES_F$residuals)  # Examining the residuals charts it appears plausible that the forecast errors is constant
                             # at mean zero and has constant variance over time except the abnormal drop in 2006
abline(reg=lm(nr_TES_F$residuals~time(nr_TES_F$residuals)), col="red",lwd=2)

#-> Now let's check if residubals are normally distributed with mean zero; following function will create a histogram of  #
#-> forcast error with over laid normal curve                                                                             #


plotForecastErrors <- function(forecasterrors)
{
  # make a histogram of the forecast errors:
  mybinsize <- IQR(forecasterrors)/4
  mysd <- sd(forecasterrors)
  mymin <- min(forecasterrors) - mysd*5
  mymax <- max(forecasterrors) + mysd*3
  # generate normally distributed data with mean 0 and standard deviation mysd
  mynorm <- rnorm(10000, mean=0, sd=mysd)
  mymin2 <- min(mynorm)
  mymax2 <- max(mynorm)
  if (mymin2 < mymin) { mymin <- mymin2 }
  if (mymax2 > mymax) { mymax <- mymax2 }
  # make a red histogram of the forecast errors, with the normally distributed data overlaid:
  mybins <- seq(mymin, mymax, mybinsize)
  hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
  # freq=FALSE ensures the area under the histogram = 1
  # generate normally distributed data with mean 0 and standard deviation mysd
  myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
  # plot the normal curve as a blue line on top of the histogram of forecast errors:
  points(myhist$mids, myhist$density, type="l", col="blue", lwd=2)
}

plotForecastErrors(nr_TES_F$residuals)  # Histogram of errors with overlaid normal curve.

#-> From the histogram of forecast errors, it seems plausible that the forecast errors are normally distributed with 
#-> mean zero. Thus there is little evidence of autocorrelation at various lags for the forecast errors, and the forecast 
#-> errors appear to be normally distributed over mean zero and has constant variance over time. This suggests that the 
#-> Holt-Winters exponential smoothing provides adequate predictiveness and likely be not required to improve further. 

###====================================================================================================================###
#  Part - III : Classical Time Series Decomposition                                                                      #
#  - Two forms - Additive Decomposition and Multiplicative Decomposition                                                 #
#  - Seasonal Indices m = 4 (quarterly); m = 12 (Monthly); m = 7 (daily)                                                 #
###====================================================================================================================###

ctsd <- decompose(train_nike_revenue, type="additive")
ctsd

plot(ctsd$seasonal)
plot(ctsd$trend)
plot(ctsd$random)
plot(ctsd$figure, type="l", col="red")
plot(ctsd$x)

###====================================================================================================================###
#  Part - IV : ARIMA Models                                                                                              #
#  - A. Is the Data Stationary? Explain if Data is Stationary? Apply ideas to Nike's Revenue Data                        #
#  - B. Can Non-Stationary Data be made Stationary Data?                                                                 #
###====================================================================================================================###
# Part - IV : A - Stationary Series                                                                                      #
###--------------------------------------------------------------------------------------------------------------------###

#-> As seen before Nike Revenue TS is non-stationary. The time series has Trend, Seasonal and Random Components.
#-> Stationarity is important in Time Series. Unless data is stationary, it cannot be used to build times series models. 
#-> For a Time Series to be Stationary 
#-> - Mean of the Time Series should be steady, it should not be changing over time.
#-> - Variance of the Time Series should be steady, it should not be changing over time.
#-> - Covariance of the Time Series should be steady, it  should not be changing over time.

###--------------------------------------------------------------------------------------------------------------------###
# Part - IV : B - Non-Stationary / Stationary Series and Conversion                                                      #
###--------------------------------------------------------------------------------------------------------------------###

#-> To make a non-stationary time series stationary:
#-> - The two most common ways to make a non-stationary time series curve stationary are:
#->   - Differencing (Most commenly used technique; can be of first order or 2nd order or 3rd order). 
#->     When TREND is noticed, differencing will detrend the time series. This is achieved with the diff function.
#->     It computes differences between consecutive observations.
#->   - Seasonality presence can be differenced by the diff function. (Quarterly at 4, Monthly at 12)
#->   - Transforming (Most common transformation is log transformation. When variance is noticed log transformaton will
#->     stabilize the variance.

###--------------------------------------------------------------------------------------------------------------------###
# Tests to check Stationarity                                                                                            #
# - 1. Stabilize the time series to Stationary                                                                           #
# - 2. ACF and PACF graphs and check significant lags                                                                    #
# - 3. Augmented Dickey-Fuller (ADF) t-statistic test suggest smaller p-value for object to be stationary                #
###--------------------------------------------------------------------------------------------------------------------###

###--------------------------------------------------------------------------------------------------------------------###
# 1. Starionarize by Differencing (Stabilize the time series): Difference for Trend and as well for Seasonal             #  
###--------------------------------------------------------------------------------------------------------------------###

train_nike_revenue_diff1 <- diff(train_nike_revenue, 1)         # Differencing the TS for Trend
plot(train_nike_revenue_diff1, col="blue", lwd=2, main="Stationarized @ Trend")
abline(reg=lm(train_nike_revenue_diff1~time(train_nike_revenue_diff1)), col="red", lwd=2) 
acf2(train_nike_revenue_diff1, max.lag = 24)

train_nike_revenue_diff1n4 <- diff(train_nike_revenue_diff1, 4) # Differencing the TS for Quarterly Seasonality

#  - Plot the differenced time series to see if seasonality and trend is removed

plot.ts(train_nike_revenue_diff1n4, col="blue", lwd=2, main="Stationary Plot - Differenced Seasonality & Trend", xlab="Year", ylab="Revenue") 
abline(reg=lm(train_nike_revenue_diff1n4~time(train_nike_revenue_diff1n4)), col="red", lwd=2) 
                                                                           #-> Trend line seems varying around mean zero
plot(decompose(train_nike_revenue_diff1n4), col="red", lwd=2) 

#-> Thus concluding Differenced Time Series at Quarterly Seasonality and 1st Order Detrend leads to Stationary TS

###--------------------------------------------------------------------------------------------------------------------###
# 2. ACF and PACF graphs and check significant lags - Decaying rate signifies Stationarity                               #
###--------------------------------------------------------------------------------------------------------------------###

acf2(train_nike_revenue_diff1n4, max.lag = 24) # ACF - Significant at 1st lag & latter decays gradually; 
                                               #       May likely that MA(0) and MA(1) are the possible models
                                               # PACF- Significant at 1st lag & decays faster.  
                                               #       May likely that AR(0) and AR(1) are the possible models
                                               # Seasonal Terms     : There is no evidence of autocorrelations and  
                                               #                      quarterly spikes are within significant bounds 
                                               #                      for ACF and PACF, so considering AR(0) and MA(0)
                                               #                      Therefore P=0 and Q=0
                                               # Non Seasonal Terms : Early spikes in ACF and PACF are at the first 
                                               #                      lag, so considering AR(1) and MA(1)
                                               #                      Therefore p=1 and q=1

#-> Thus concluding the possible model as ARIMA(1,1,1)x(0,1,0)[4] for a time series with Trend differenced at first order
#-> and Seasonality differenced at quarterly 
#-> This is because ar(1), 1st diff, ma(1) for non-seasonal terms and AR(0), Seaonal Diff, and MA(0) for seasonal terms 
#-> Time Span Differenced at 4 - Quarterly Data.

###--------------------------------------------------------------------------------------------------------------------###
# 3. Augmented Dickey-Fuller (ADF) t-statistic test - Check for Smaller p-Value                                          #  
###--------------------------------------------------------------------------------------------------------------------###

adf.test(train_nike_revenue_diff1n4, alternative = "stationary") # p-Value is statistically significant and is 0.5049;

#->     Augmented Dickey-Fuller Test
#-> data:  train_nike_revenue_diff1
#-> Dickey-Fuller = -2.1757, Lag order = 3, p-value = 0.5049
#-> alternative hypothesis: stationary

###--------------------------------------------------------------------------------------------------------------------###
# Similarly difference the validation set; So as prepare the validation dataset for use during modelling                 #
###--------------------------------------------------------------------------------------------------------------------###

val_nike_revenue_diff4 <- diff(val_nike_revenue, 4)
val_nike_revenue_diff4
val_nike_revenue_diff1n4 <- diff(val_nike_revenue_diff4, 1)
val_nike_revenue_diff1n4

#####================================================================================================================#####
##  Build Forecasting Models                                                                                            ##
#####================================================================================================================#####
##  - 1. Build Forecast Models with ARIMA                                                                               ##
##  - 2. Predict with ARIMA - For Revenue Year 2010                                                                     ##
###--------------------------------------------------------------------------------------------------------------------###

###--------------------------------------------------------------------------------------------------------------------###
# 1. Build Model to Forecast for ARIMA(p,d,q)x(P,D,Q)  Model - In this case ARIMA(1,1,1)x(0,1,0)[4]                      #  
###--------------------------------------------------------------------------------------------------------------------###

nr_fit_sarima <- sarima(train_nike_revenue,1,1,1,0,1,0,4) #fit model with sarima; 

nr_fit_sarima

#-> Residuals variate on mean zero
#-> ACF shows residuals within the threshold boundaries. No significant correlation noticed.
#-> Q-Q Plot show normal probability with good fit along the blue line
#-> The p-values from the Ljung-Box Statistic are all statistically significant, above the blue line.
#-> AIC Measure is 423.01
#-> Can conclude as a Valid Best Fit Model.

###--------------------------------------------------------------------------------------------------------------------###
#    Build Model to Forecast using auto ARIMA Model                                                                      #  
###--------------------------------------------------------------------------------------------------------------------###

nr_fit_arima <- auto.arima(train_nike_revenue)

nr_fit_arima

#->         Series: train_nike_revenue 
#-> Auto ARIMA select ARIMA(1,1,0)(0,1,0)[4]                    
#-> Coefficients:
#-> ar1
#-> -0.5066
#-> s.e.   0.1480
#-> sigma^2 estimated as 9647:  log likelihood=-209.11
#-> AIC=422.23   AICc=422.6   BIC=425.34

nr_fit_arima$fitted       #check the fitted values
nr_fit_arima$residuals    #check the residuals 

###--------------------------------------------------------------------------------------------------------------------###
# 2. Ljung-Box Test - Check Non-Zero Correlations; Smaller p-Value                                                       #  
###--------------------------------------------------------------------------------------------------------------------###

Box.test(nr_fit_arima$residuals, type="Ljung-Box") # Shows much smaller p-Value i.e. = 0.00117; close to 0 
                                                   # Shows very little evidence of non-zero correlations

# So we conculde our Nike Revenue Time Series is Stationary 
#->         Box-Ljung test
#->  data:  train_nike_revenue
#->  X-squared = 0.062938, df = 1, p-value = 0.8019

###--------------------------------------------------------------------------------------------------------------------###
# 3. Residual Analysis for ARIMA Models - ARIMA(1,1,0)x(0,1,0)[4] and Auto ARIMA                                         #
###--------------------------------------------------------------------------------------------------------------------###

tsdisplay(residuals(nr_fit_arima), lag.max=36, main='Auto ARIMA Residuals', col="red", lwd=2)

#-> Residuals variate around mean zero
#-> The ACF/PACF correlogram does not show autocorrelations
#-> And in-sample forecast errors do not exceed the threshold boundaries

#-> Now to double confirm whether the data has any evidence of autocorrelations, carry out a Ljung-Box test: 

Box.test(nr_fit_arima$residuals, lag=36, type="Ljung-Box")

#->           Box-Ljung test
#-> data:  nr_fit_arima$residuals
#-> X-squared = 23.993, df = 36, p-value = 0.9372

#-> From Ljung-Box test resuts above with the p value is 0.9372, it is preety clear that there is no significant 
#-> evidence of non-zero autocorrelations in  residuals.

#-> Now in order to check whether the forecast errors have constant variance over time, and are normally distributed 
#-> with mean zero, by making a time plot of the forecast errors and a histogram (with overlaid normal curve):

plotForecastErrors(nr_fit_arima$residuals)

#-> It appears from the plot that residuals are aproximatly normally distributed with mean zero and seems slighly right skewed

#-> By comparision ARIMA (1,1,1)x(0,1,0) and Auto ARIMA, the later was better model. Concluding Auro ARIMA as Best Model.

###--------------------------------------------------------------------------------------------------------------------###
##  - Forecasts with SARIMA(1,1,1)x(0,1,0)[4] and Auto ARIMA for Fiscal Year 2009 and 2010                              ##
###--------------------------------------------------------------------------------------------------------------------###

nr_fc_sa_2009_2010 <- sarima.for(train_nike_revenue, n.ahead=8, 1,1,1,0,1,0,4)
nr_fc_sa_2009_2010

#->  Predicted results for validation period - 2009/2010
#->               Forecast    se
#->  2008 Q3      5336.331    93.03291
#->  2008 Q4      5039.392    108.00468
#->  2009 Q1      5229.610    136.81786
#->  2009 Q2      5784.126    150.11288
#->  2009 Q3      6024.433    228.10936
#->  2009 Q4      5733.617    258.04836
#->  2010 Q1      5919.163    303.43960
#->  2010 Q2      6477.244    330.06936

nr_fc_aa_2009_2010 <- forecast(nr_fit_arima, h=8)
plot(nr_fc_aa_2009_2010, main = "ARIMA - Forecast Fiscal Year 2009/2010")
nr_fc_aa_2009_2010

#->  Predicted results for Year - 2010
#->               Forecast    Lo 80    Hi 80    Lo 95    Hi 95
#->  2008 Q3      5315.415 5192.179 5438.650 5126.942 5503.887
#->  2008 Q4      5023.004 4885.587 5160.421 4812.843 5233.165
#->  2009 Q1      5215.559 5049.948 5381.170 4962.278 5468.840
#->  2009 Q2      5765.358 5582.971 5947.744 5486.422 6044.294
#->  2009 Q3      5989.835 5713.376 6266.293 5567.028 6412.641
#->  2009 Q4      5698.912 5388.478 6009.346 5224.144 6173.680
#->  2010 Q1      5890.713 5534.386 6247.040 5345.758 6435.668
#->  2010 Q2      6440.894 6051.171 6830.616 5844.865 7036.923

#####================================================================================================================#####
##  Summary of the Report                                                                                               ##
###--------------------------------------------------------------------------------------------------------------------###
#   Forecasting Approach : Prediction with HoltWinters Triple Exponential Smoothing Method and ARIMA Method              #
#                                                                                                                        #
#   Nike's Revenue data collected for fiscal years 1999 thru 2008 show that there is gradual rise in trend and           #
#   as well seasonal repetitive pattern periodically over each financial year. Repetitive rise and fall pattern          #
#   over the fiscal year is observed.                                                                                    #
#                                                                                                                        #
#   Minimum revenue was recorded at 1913 during 2nd Quarter of 1999 Fiscal Year, while Maximum revenue was               #
#   recorded at 5088 4th Quarter of 2008 Fiscal Year. Average revenues for the period is 3094.                           #
#                                                                                                                        #
#   A simple prediction / forecasting called Regression Analysis was not satifactory due to Seaonal and Trend Variances  #
#                                                                                                                        #
#   Moving ahead with other techniques including HoltWinter TES and ARIMA Methods, lead to signicant improvement in the  #
#   forecasting accuracy. From the plots we see the results aligning the Trend and along the seasonality.                #                                                                                                                    #
#                                                                                                                        #
#   Auto ARIMA Model is considered the final model for Forecasting the Nike's Revenue for Fiscal Year 2010               #
#####================================================================================================================#####

