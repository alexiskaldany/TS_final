# TS_final

# Final Project

- [data source](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)

## Plan

1. Create python file, save all images to folder

- Use Ipython and comments to keep track of work

2. Write markdown file instead of word, convert to pdf at end

- To link to images: use <https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/(name> of file)

## Requirements

- Description of the dataset. Describe the independent variable(s) and dependent variable:
  - Pre-processing dataset: Dataset cleaning for missing observation. You must follow the data cleaning techniques for time series dataset. (done)
  - Plot of the dependent variable versus time. (done)
  - ACF/PACF of the dependent variable. (done)
  - Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient. (done)
  - Split the dataset into train set (80%) and test set (20%). (done)

- Stationarity: Check for a need to make the dependent variable stationary. If the dependent variable is not stationary, you need to use the techniques discussed in class to make it stationary. (done)

    1. It appears the original data has the only rolling mean and rolling variance which is stationary after observing many combinations of differencing and log-differencing
    2. ADF/KPSS confirmed

- Perform ACF/PACF analysis for stationarity. You need to perform ADF-test & kpss-test and plot the rolling mean and variance for the raw data and the transformed data. (done)

- Time series Decomposition: Approximate the trend and the seasonality and plot the detrended and the seasonally adjusted data set. Find the out the strength of the trend and seasonality. Refer to the lecture notes for different type of time series decomposition techniques. (done)
    1. Already found differenced/seasonal, use 150 lag
- Holt-Winters method: Using the Holt-Winters method try to find the best fit using the train dataset and make a prediction using the test set. (done)

- Feature selection: You need to have a section in your report that explains how the feature selection was performed and whether the collinearity exits not. Backward stepwise regressionalong with SVD and condition number is needed. You must explain that which feature(s) need to be eliminated and why. You are welcome to use other methods like PCA or random forest for feature elimination. (done)
- Base-models: average, naïve, drift, simple and exponential smoothing. You need to perform an h-step prediction based on the base models and compare the SARIMA model performance with the base model predication. (done)
- Develop the multiple linear regression model that represent the dataset. Check the accuracy of
the developed model. (done)
  - You need to include the complete regression analysis into your report. - Perform one-step ahead prediction and compare the performance versus the test set. (done)
  - Hypothesis tests analysis: F-test, t-test. (done)
  - AIC, BIC, RMSE, R-squared and Adjusted R-squared (done)
  - ACF of residuals. (done)
  - Q-value (done)
  - Variance and mean of the residuals. (done)

- ARMA and ARIMA and SARIMA model order determination: Develop an ARMA, ARIMA and SARIMA model that represent the dataset.
  - Preliminary model development procedures and results. (ARMA model order
determination). Pick at least two orders using GPAC table. (Done)
  - Should include discussion of the autocorrelation function and the GPAC. Include a plot of the autocorrelation function and the GPAC table within this section. (done)
  - Include the GPAC table in your report and highlight the estimated order.
- Estimate ARMA model parameters using the Levenberg Marquardt algorithm. Display the parameter estimates, the standard deviation of the parameter estimates and confidence intervals. (skip)
- Diagnostic Analysis: Make sure to include the followings:
  - Diagnostic tests (confidence intervals, zero/pole cancellation, chi-square test).
  - Display the estimated variance of the error and the estimated covariance of the estimated parameters.
  - Is the derived model biased or this is an unbiased estimator?
  - Check the variance of the residual errors versus the variance of the forecast errors.
  - If you find out that the ARIMA or SARIMA model may better represents the dataset, then you can find the model accordingly. You are not constraint only to use of ARMA model. Finding an ARMA model is a minimum requirement and making the model better is always welcomed.
- Deep Learning Model: Fit the dataset into multivariate LSTM model. You also need to perform hstep prediction using LSTM model. You can use tensorflow package in python for this section. (done)
- Final Model selection: There should be a complete description of why your final model was picked over base-models ARMA, ARIMA, SARIMA and LSTM. You need to compare the performance of various models developed for your dataset and come up with the best model that represent the dataset the best.
- Forecast function: Once the final mode is picked (SARIMA), the forecast function needs to be
developed and included in your report.
- h-step ahead Predictions: You need to make a multiple step ahead prediction for the duration of the test data set. Then plot the predicted values versus the true value (test set) and write down your observations.
