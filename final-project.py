#%%
from matplotlib.cbook import flatten
from toolbox import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import warnings
import seaborn as sns
import tensorflow as tf
from statsmodels.tsa.api import SARIMAX, AutoReg
from keras.layers import Dense
from keras.layers import LSTM
from keras import Sequential
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.preprocessing.timeseries import timeseries_dataset_from_array
import statsmodels.tsa.holtwinters as ets
from numpy import linalg as LA
warnings.filterwarnings('ignore')
image_folder = 'final-images/'

#%%
### Reading csv
df_raw = pd.read_csv('final-data.csv',parse_dates=["date"])
df = df_raw.copy()
df['Date']=pd.to_datetime(df_raw['date'])
del df['date']
del df['rv1']
del df['rv2']
df['Date']=pd.to_datetime(df_raw['date'])
df.set_index('Date', inplace = True)

### Dataset Cleaning
df_na = df[df.isnull().any(axis=1)]
print(f"There are {len(df_na)} many rows with NaN values")
### Checking column types
df_types = df.dtypes
print(df_types)
# All columns are either int, float, or datetime, looks good

########## Description Section
# 1a. Plot of the dependent variable versus time.
plt.figure(figsize=(12, 8), layout='constrained')
plt.plot(df.index,df['Appliances'])
plt.title('Appliance Energy Use')
plt.xlabel('10 Minute Intervals')
plt.ylabel('Energy Use in Wh')
# plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
plt.grid()
plt.savefig(image_folder+'1a-dependent.png',dpi=1000)
plt.show()

# 1c. ACF/PACF of the dependent variable
lags = 300
acf = sm.tsa.stattools.acf(df.Appliances, nlags=lags)
pacf = sm.tsa.stattools.pacf(df.Appliances, nlags=lags)
fig = plt.figure()
plt.subplot(211)
plt.title('ACF/PACF of the raw data')
plot_acf(df.Appliances, ax=plt.gca(), lags=lags)
plt.subplot(212)
plot_pacf(df.Appliances, ax=plt.gca(), lags=lags)
fig.tight_layout(pad=3)
plt.savefig(image_folder+'1c-ACF-PACF-Original.png', dpi=1000)
plt.show()  
# Stem plot TODO :make symmetric
stem_acf('Appliances',acf_df(df.Appliances,300),19735)
###################
# 1d. Correlation Matrix + Heatmap
corr = df.corr()
ax = plt.axes()
ax = sns.heatmap(data=corr)
ax.set_title("Heatmap of Pearson's Correlation Matrix")
plt.savefig(image_folder+'1d-heatmap-corr.png', dpi=1000)
plt.show()
##################
# 1e. Train/Test (80/20)
index_80 = int(len(df)*0.8)
index_20 = int(len(df)-index_80)
df_train = df[:index_80]
df_test = df[index_80:]

######################### 
#Ok, sticking with original dependent variable
adf_original, kpss_original = adf_kpss_statistic(df.Appliances) 
print(f" The ADF p-value for the original data is: {adf_original}")
print(f" The KPSS p-value for the original data is: {kpss_original}")
original_mean_var = cal_rolling_mean_var(df.Appliances)
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Original Data')
ax1.plot(original_mean_var.index, original_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(original_mean_var.index, original_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_original.png', dpi=1000)
plt.show()

# ############ Difference Dataframes
log_df = log_transform(df.Appliances,df.index)
diff_df = pd.DataFrame()
diff_df['Original'] = df.Appliances
diff_df_list = [diff(df.Appliances,x,df.index) for x in range(1,301)]
diff_combined_df = pd.concat(diff_df_list,axis=1)
diff_combined_df.to_csv('diff_combined_df.csv')
##
log_diff_list = [diff(log_df['log_transform'],x,log_df.index) for x in range(1,301)]
log_diff_df = pd.concat(log_diff_list,axis=1)
log_diff_df.to_csv('log_diff_df.csv')
log_diff_df = pd.read_csv('log_diff_df.csv')
diff_combined_df= pd.read_csv('diff_combined_df.csv')
## Log Transform Dataframes

## Cannot take log after difference because of negatives
log_diff_list = [log_transform(diff_combined_df[f"{x}_diff"],diff_combined_df.index) for x in range(1,301)]
diff_log_df = pd.concat(log_diff_list,axis=1)
# Testing log transform on original Appliances
original_transform = adf_kpss_statistic(log_df.log_transform)
print(original_transform)
#################
# Testing for Stationarity
adf_list = []
kpss_list = []
for x in range(1,301):
    output = adf_kpss_statistic(diff_combined_df[f"{x}_diff"])
    adf_list.append(output[0])
    kpss_list.append(output[1])
diff_stats = pd.DataFrame(index=range(1,101))
diff_stats['ADF'] = adf_list
diff_stats['KPSS'] = kpss_list
diff_stats.to_csv('diff_stats.csv')
## ADF Test statistic high == low p value == alternative hypothesis == stationarity
## KPSS Test statistic high == low p value == alternative hypothesis == non-stationarity
# ADF
plt.figure(figsize=(12, 8), layout='constrained')
plt.plot(diff_stats.index,diff_stats['ADF'])
plt.title('ADF Statistics vs Diff Intervals')
plt.xlabel('Diff Intervals')
plt.ylabel('ADF Statistics')
plt.grid()
plt.savefig(image_folder+'1a-ADF-Stats.png',dpi=1000)
plt.show()
## KPSS
plt.figure(figsize=(12, 8), layout='constrained')
plt.plot(diff_stats.index,diff_stats['KPSS'])
plt.title('KPSS Statistics vs Diff Intervals')
plt.xlabel('Diff Intervals')
plt.ylabel('KPSS Statistics')
plt.grid()
plt.savefig(image_folder+'1a-KPSS-Stats.png',dpi=1000)
plt.show()

#########
# Rolling mean/var
log_mean_var = cal_rolling_mean_var(log_df.log_transform)
diff_10_mean_var = cal_rolling_mean_var(diff_combined_df['10_diff'])
diff_25_mean_var = cal_rolling_mean_var(diff_combined_df['25_diff'])
diff_50_mean_var = cal_rolling_mean_var(diff_combined_df['50_diff'])
diff_150_mean_var = cal_rolling_mean_var(diff_combined_df['150_diff'])
diff_300_mean_var = cal_rolling_mean_var(diff_combined_df['300_diff'])
log_diff_1_mean_var = cal_rolling_mean_var(log_diff_df['1_diff'])
log_diff_2_mean_var = cal_rolling_mean_var(log_diff_df['2_diff'])
log_diff_3_mean_var = cal_rolling_mean_var(log_diff_df['3_diff'])
log_diff_4_mean_var = cal_rolling_mean_var(log_diff_df['4_diff'])
log_diff_5_mean_var = cal_rolling_mean_var(log_diff_df['5_diff'])
log_diff_10_mean_var = cal_rolling_mean_var(log_diff_df['10_diff'])
log_diff_25_mean_var = cal_rolling_mean_var(log_diff_df['25_diff'])
log_diff_50_mean_var = cal_rolling_mean_var(log_diff_df['50_diff'])
log_diff_150_mean_var = cal_rolling_mean_var(log_diff_df['150_diff'])
log_diff_300_mean_var = cal_rolling_mean_var(log_diff_df['300_diff'])
## Plotting
# Log rolling
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Log Data')
ax1.plot(log_mean_var.index, log_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(log_mean_var.index, log_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_log.png', dpi=1000)
plt.show()
# 10 diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('10 Diff')
ax1.plot(diff_10_mean_var.index, diff_10_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(diff_10_mean_var.index, diff_10_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_diff_10.png', dpi=1000)
plt.show()
# 25 diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('25 Diff')
ax1.plot(diff_25_mean_var.index, diff_25_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(diff_25_mean_var.index, diff_25_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_diff_25.png', dpi=1000)
plt.show()
# 50 dif
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('50 Diff ')
ax1.plot(diff_50_mean_var.index, diff_50_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(diff_50_mean_var.index, diff_50_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_diff_50.png', dpi=1000)
plt.show()
#150 dif
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('150 Diff ')
ax1.plot(diff_150_mean_var.index, diff_150_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(diff_150_mean_var.index, diff_150_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_diff_150.png', dpi=1000)
plt.show()
#300 dif
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('300 Diff ')
ax1.plot(diff_300_mean_var.index, diff_300_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(diff_300_mean_var.index, diff_300_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_diff_300.png', dpi=1000)
plt.show()
# 1 log diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('1 Log Diff ')
ax1.plot(log_diff_1_mean_var.index, log_diff_1_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(log_diff_1_mean_var.index, log_diff_1_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_log_diff_1.png', dpi=1000)
plt.show()
# 2 Log Diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('2 Log Diff ')
ax1.plot(log_diff_2_mean_var.index, log_diff_2_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(log_diff_2_mean_var.index, log_diff_2_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_log_diff_2.png', dpi=1000)
plt.show()
# 3 Log Diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('3 Log Diff ')
ax1.plot(log_diff_3_mean_var.index, log_diff_3_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(log_diff_3_mean_var.index, log_diff_3_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_log_diff_3.png', dpi=1000)
plt.show()
# 4 Log Diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('4 Log Diff ')
ax1.plot(log_diff_4_mean_var.index, log_diff_4_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(log_diff_4_mean_var.index, log_diff_4_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_log_diff_4.png', dpi=1000)
plt.show()
# 5 Log Diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('5 Log Diff ')
ax1.plot(log_diff_5_mean_var.index, log_diff_5_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(log_diff_5_mean_var.index, log_diff_5_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_log_diff_5.png', dpi=1000)
plt.show()
# 10 Log Diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('10 Log Diff ')
ax1.plot(log_diff_10_mean_var.index, log_diff_10_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(log_diff_10_mean_var.index, log_diff_10_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_log_diff_10.png', dpi=1000)
plt.show()
# 25 log diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('25 Log Diff ')
ax1.plot(log_diff_25_mean_var.index, log_diff_25_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(log_diff_25_mean_var.index, log_diff_25_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_log_diff_25.png', dpi=1000)
plt.show()
# 50 log diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('50 Log Diff ')
ax1.plot(log_diff_50_mean_var.index, log_diff_50_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(log_diff_50_mean_var.index, log_diff_50_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_log_diff_50.png', dpi=1000)
plt.show()
#150 log diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('150 Log Diff ')
ax1.plot(log_diff_150_mean_var.index, log_diff_150_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(log_diff_150_mean_var.index, log_diff_150_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_log_diff_150.png', dpi=1000)
plt.show()
#300 log diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('300 Log Diff ')
ax1.plot(log_diff_300_mean_var.index, log_diff_300_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(log_diff_300_mean_var.index, log_diff_300_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_log_diff_300.png', dpi=1000)
plt.show()
#%%
# Time Series Decomposition
acf = sm.tsa.stattools.acf(diff_combined_df['150_diff'], nlags=lags)
pacf = sm.tsa.stattools.pacf(diff_combined_df['150_diff'], nlags=lags)
fig = plt.figure()
plt.subplot(211)
plt.title('ACF/PACF of the raw data')
plot_acf(diff_combined_df['150_diff'], ax=plt.gca(), lags=lags)
plt.subplot(212)
plot_pacf(diff_combined_df['150_diff'], ax=plt.gca(), lags=lags)
fig.tight_layout(pad=3)
plt.savefig(image_folder+'Decomposition-150-ACF-PACF-Original.png', dpi=1000)
plt.show()  
stem_acf('Appliances',acf_df(diff_combined_df['150_diff'],300),19735)
###################
res = STL(df.Appliances).fit()
fig = res.plot()
plt.title('Original')
plt.ylabel('Residual')
plt.xlabel('Iterations')
plt.tight_layout()
plt.savefig(image_folder+'Original-Decomposition.png', dpi=1000)
plt.show()

res = STL(diff_combined_df['150_diff']).fit()
fig = res.plot()
plt.title('150 Diff')
plt.ylabel('Residual')
plt.xlabel('Iterations')
plt.tight_layout()
plt.savefig(image_folder+'150Diff-Decomposition.png', dpi=1000)
plt.show()

origin_stl = STL(df.Appliance)
res = origin_stl.fit()
#res.plot()



origin_SOT=strength_of_trend(res.resid,res.trend)
origin_season = strength_of_seasonal(res.resid,res.season)

diff_stl = STL(diff_combined_df['150_diff'])
res = diff_stl.fit()
#res.plot()

diff_SOT=strength_of_trend(res.resid,res.trend)
diff_season = strength_of_seasonal(res.resid,res.seasonal)

plt.figure()
plt.plot(df.index,res.trend, label= 'Trend')
plt.plot(df.index,res.resid, label= 'Residual')
plt.plot(df.index,res.season, label= 'Seasonal')
plt.title('Trend, Residual, and Seasonal Plot')
plt.xticks(df.index[::4500], fontsize= 10)
plt.ylabel('Electricity (Wh)')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.savefig(image_folder+'Cleaner-150-Decomposition.png', dpi=1000)
plt.show()

adjusted_seasonal = np.array(df.Appliance.values - res.season)
detrended = np.array(df.Appliances - res.trend)
residual = np.array(res.resid)
adjust_seas = np.array(adjusted_seasonal)

plt.figure()
plt.plot(df.index,df.Appliances, label= 'Original Data', color = 'black')
plt.plot(df.index,adjusted_seasonal, label= 'Adjusted Seasonal', color = 'yellow')
plt.xticks(df.index[::4500], fontsize= 10)
plt.title('Seasonaly Adjusted Data vs. Differenced')
plt.xlabel('Date')
plt.ylabel('Electricity (Wh)')
plt.legend()
plt.tight_layout()
plt.savefig(image_folder+'Seasonal-Adjusted-Decomposition.png', dpi=1000)
plt.show()

plt.figure()
plt.plot(df.index,df.Appliances, label= 'Original Data')
plt.plot(df.index,detrended, label= 'Detrended')
plt.xticks(df.index[::4500], fontsize= 10)
plt.title('Detrended vs. Original')
plt.xlabel('Date')
plt.ylabel('Electricity (Wh)')
plt.legend()
plt.tight_layout()
plt.savefig(image_folder+'Detrended-Decomposition.png', dpi=1000)
plt.show()

####
# ####Holt-Winters
model = ets.ExponentialSmoothing(df['Appliances'], damped_trend= True, seasonal_periods=150, trend='add', seasonal='add').fit()

# prediction on train set
hw = model.forecast(steps=len(df_train['Appliances']))
train_hw = pd.DataFrame(hw, columns=['Appliances']).set_index(df_train.index)
print(hw.summary())
# prediction on test set
hw = model.forecast(steps=len(df_test['Appliances']))
test_hw = pd.DataFrame(hw, columns=['Appliances']).set_index(df_test.index)
print(hw.summary())

# model assessment
hw_train_error = np.array(df_train['Appliances'] - train_hw['Appliances'])
print(f"Train MSE: {mse(hw_train_error).round(4)}")
print(sm.stats.acorr_ljungbox(hw_train_error, lags=[5], boxpierce=True, return_df=True))

print('HW train error mean is', np.mean(hw_train_error))
print('the variance of the Holt-winter model prediction error is', np.var(hw_train_error))

# test data
hw_test_error = np.array(df_test['Appliances'] - test_hw['Appliances'])
print(f"Test MSE is: {mse(hw_test_error)}")
print(sm.stats.acorr_ljungbox(hw_test_error, lags=[5], boxpierce=True, return_df=True))
print('HW test error mean is', np.mean(hw_test_error))
print('the variance of the Holt-winter model error is', np.var(hw_test_error))


# plot Holt-Winter model

# plot of full model
plt.figure()
plt.xlabel('Time')
plt.ylabel('Electricity (Wh)')
plt.title('Holt-Winter Method on Data')
plt.plot(train_hw.index,train_hw.Appliances,label= "Train Data", color = 'green')
plt.plot(test_hw.index,test_hw.Appliances,label= "Test Data", color = 'blue')
plt.plot(test_hw.set_index(test_hw.index), label = 'Forecasting Data', color = 'red')
plt.xticks(df.index[::4500], fontsize= 10)
plt.legend()
plt.tight_layout()
plt.savefig(image_folder+'HW-Train-Test-Predict.png', dpi=1000)
plt.show()


# plot of test data
plt.figure()
plt.plot(test_hw.index,test_hw.Appliances,label= "Test Data", color = 'green')
plt.plot(test_hw.set_index(test_hw.index), label = 'Forecasting Data', color = 'blue')
plt.xlabel('Time')
plt.ylabel('Electricity (Wh)')
plt.title(f'Holt-Winter Method on Data with MSE = {mse(hw_test_error).round(4)}')
plt.xticks(test_hw.index[::725], fontsize= 10)
plt.legend()
plt.tight_layout()
plt.savefig(image_folder+'HW-Test-Predict.png', dpi=1000)
plt.show()

# holt-winter train data
plt.figure()
pred_f = 1.96/np.sqrt(len(train_hw.Appliances))
acf = sm.tsa.stattools.acf(train_hw.Appliances, nlags=lags)
pacf = sm.tsa.stattools.pacf(train_hw.Appliances, nlags=lags)
fig = plt.figure()
plt.subplot(211)
plt.title('ACF/PACF of the H-W Train data')
plot_acf(train_hw.Appliances, ax=plt.gca(), lags=lags)
plt.subplot(212)
plot_pacf(train_hw.Appliances, ax=plt.gca(), lags=lags)
fig.tight_layout(pad=3)
plt.savefig(image_folder+'HW-Train-PACF.png', dpi=1000)
plt.show() 


# holt winter test data
acf = sm.tsa.stattools.acf(test_hw.Appliances, nlags=lags)
pacf = sm.tsa.stattools.pacf(test_hw.Appliances, nlags=lags)
fig = plt.figure()
plt.subplot(211)
plt.title('ACF/PACF of the HW Test data')
plot_acf(test_hw.Appliances, ax=plt.gca(), lags=lags)
plt.subplot(212)
plot_pacf(test_hw.Appliances, ax=plt.gca(), lags=lags)
fig.tight_layout(pad=3)
plt.savefig(image_folder+'HW-Test_PACF.png', dpi=1000)
plt.show()  


###### Backwords Selection
features = []
x_train = df_train.drop(columns=['Appliances'])
y_train = df_train.Appliances
x_train_ols = sm.add_constant(x_train)
OLS_model = sm.OLS(y_train, x_train_ols)
OLS_fit = OLS_model.fit()
print(OLS_fit.summary())
OLS_coefficients = OLS_fit.params
initial_aic_bic_rsquared = aic_bic_rsquared_df(OLS_fit)


newer_df, newer_x = loop_backwards(x_train,y_train,initial_aic_bic_rsquared)
newer_df, newer_x = loop_backwards(newer_x,y_train,newer_df)
newer_df, newer_x = loop_backwards(newer_x,y_train,newer_df)
newer_df, newer_x = loop_backwards(newer_x,y_train,newer_df)
newer_df, newer_x = loop_backwards(newer_x,y_train,newer_df)
newer_df, newer_x = loop_backwards(newer_x,y_train,newer_df)
newer_df, newer_x = loop_backwards(newer_x,y_train,newer_df)
newer_df, newer_x = loop_backwards(newer_x,y_train,newer_df)
newer_df, newer_x = loop_backwards(newer_x,y_train,newer_df)
newer_df, newer_x = loop_backwards(newer_x,y_train,newer_df)
newer_df, newer_x = loop_backwards(newer_x,y_train,newer_df)
newer_df, newer_x = loop_backwards(newer_x,y_train,newer_df)
newer_df, newer_x = loop_backwards(newer_x,y_train,newer_df)
features.append(newer_x.columns[0])
newer_df['feature_to_drop'] = features


x_trainer = sm.add_constant(x_train)
H = np.matmul(x_trainer.T,x_trainer)
print('This is H dim', H.shape)
s,d,v = np.linalg.svd(H)
print('SingularValues = ', d)
#Condition number
print(" the condition number for X is = ", LA.cond(x_trainer))































#%%
# LSTM
# 1e. Train/Test (80/20)
n_input = 300
n_features= 25

df_lstm = pd.read_csv('final-data.csv')
df_lstm.set_index('date', inplace = True)
del df_lstm['rv1']
del df_lstm['rv2']
timestep = [x for x in range(len(df_lstm))]
index_train = int(len(df)*0.8)
index_test = int(len(df)-index_80)
index_val = int(index_train*0.8)
length_val = index_train-index_val
df_train = df_lstm.iloc[:index_val,:]
df_val = df_lstm.iloc[index_val:index_train,:]
df_test = df_lstm.iloc[index_train:,:]


# x_train, x_test, y_train, y_test = np.array(df_train.iloc[:,1:]),np.array(df_train.iloc[:,1]),np.array(df_test.iloc[:,1:]),np.array(df_test.iloc[:,1])

tr_x = np.array(df_train.drop(columns=['Appliances'])).reshape(len(df_train),25)

tr_y =np.array(df_train[['Appliances']]).reshape(len(df_train),1)

val_x = np.array(df_val.drop(columns=['Appliances'])).reshape(len(df_val),n_features,1).reshape(len(df_val),25,1)

val_y =np.array(df_val[['Appliances']]).reshape(len(df_val),1,1)

test_x = np.array(df_test.drop(columns=['Appliances'])).reshape(len(df_test),n_features,1).reshape(len(df_test),25,1)

test_y = np.array(df_test[['Appliances']]).reshape(df_test.shape[0],1,1)

# #####
# x_train = np.array(df_train.drop(columns=['Appliances','date']))
# x_train_shaped = np.reshape(x_train,(len(x_train),25,1))
# y_train = np.array(df_train.iloc[:,[1]])
# y_train_shaped = np.reshape(y_train,(len(y_train),1,1))
# x_val = np.array(df_val.drop(columns=['Appliances','date']))
# x_val_shaped = np.reshape(x_val,(len(x_val),25,1))
# y_val = np.array(df_val.iloc[:,[1]])
# y_val_shaped = np.reshape(y_val,(len(y_val),1,1))
# x_test = np.array(df_test.drop(columns=['Appliances','date']))
# x_test_shaped = np.reshape(x_test,(len(x_test),25,1))
# y_test = np.array(df_test.iloc[:,[1]])
# y_test_shaped = np.reshape(y_test,(len(y_test),1,1))



# train_loaded = TimeseriesGenerator(
#     # x_train_shaped,
#     # y_train_shaped,
#     data=tr_x,
#     targets=tr_y,
#     length=n_input,
    
# )
# val_loaded = TimeseriesGenerator(
#     # x_val_shaped,
#     # y_val_shaped,
#     data = val_x,
#     targets = val_y,
#     length=n_input,
    
    
# )
# test_loaded = TimeseriesGenerator(
#     # x_test_shaped,
#     # y_test_shaped,
#     data=test_x,
#     targets = test_y,
#     length=n_input,
    
# )

# model = Sequential()
# model.add(LSTM(50,activation = 'relu',return_sequences = True,input_shape=(50,1)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer="adam", metrics=['mean_squared_error'])


# model.fit(train_loaded,validation_data=val_loaded, epochs=1000)

inputs = timeseries_dataset_from_array(data=tr_x,targets=None,sequence_length=50)
targets= timeseries_dataset_from_array(data=tr_y,targets=None,sequence_length=50)

dataset = tf.data.Dataset.zip((inputs, targets))

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()





# %%
x = np.reshape(tr_x,(df_train.drop(columns=['Appliances']).shape[0],df_train.drop(columns=['Appliances']).shape[1],1))
print(x.shape)
y = np.reshape(tr_y,(df_train.shape[0],1,1))
print(y.shape)
#%%
model = Sequential()
model.add(LSTM(50,activation = 'relu',return_sequences = True,input_shape=(None,25)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer="adam",metrics=['mean_squared_error'])
history = model.fit(dataset)
# %%
visualize_loss(model.history, "Training and Validation Loss")