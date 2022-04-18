#%%
from toolbox import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import warnings
import seaborn as sns
import tensorflow
from tensorflow import keras
from statsmodels.tsa.api import SARIMAX, AutoReg
warnings.filterwarnings('ignore')
#%%
image_folder = '/final-images/'

#%%
### Reading csv
df_raw = pd.read_csv('final-data.csv',parse_dates=["date"])
df_raw['Date']=pd.to_datetime(df_raw['date'])
del df_raw['date']
df_raw.set_index('Date', inplace = True)
#%%
### Dataset Cleaning
df_na = df_raw[df_raw.isnull().any(axis=1)]
print(f"There are {len(df_na)} many rows with NaN values")
df = df_raw.copy()
### Checking column types
df_raw_types = df_raw.dtypes
print(df_raw_types)
# All columns are either int, float, or datetime, looks good
#%%
### Description Section
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
lags = 50
acf = sm.tsa.stattools.acf(df.Appliances, nlags=lags)
pacf = sm.tsa.stattools.pacf(df.Appliances, nlags=lags)
fig = plt.figure()
plt.subplot(211)
plt.title('ACF/PACF of the raw data')
plot_acf(df.Appliances, ax=plt.gca(), lags=lags)
plt.subplot(212)
plot_pacf(df.Appliances, ax=plt.gca(), lags=lags)
fig.tight_layout(pad=3)
plt.savefig(image_folder+'1c-ACF-PACF.png', dpi=1000)
plt.show()  
# Stem plot TODO :make symmetric
stem_acf('Appliances',acf_df(df.Appliances,50),19735)
# %%
# 1d. Correlation Matrix + Heatmap
corr = df.corr()
ax = plt.axes()
ax = sns.heatmap(data=corr)
ax.set_title("Heatmap of Pearson's Correlation Matrix")
plt.savefig(image_folder+'1d-heatmap-corr.png', dpi=1000)
plt.show()

#%%
# 1e. Train/Test (80/20)
index_80 = int(len(df)*0.8)
df_train = df[:index_80]
df_test = df[index_80:]
# %%
# 2 
# Difference Dataframes
log_df = log_transform(df.Appliances,df.index)
diff_df = pd.DataFrame()
diff_df['Original'] = df.Appliances
diff_df_list = [diff(df.Appliances,x,df.index) for x in range(1,101)]
diff_combined_df = pd.concat(diff_df_list,axis=1)
diff_combined_df.to_csv('diff_combined_df.csv')
##
log_diff_list = [diff(log_df['log_transform'],x,log_df.index) for x in range(1,101)]
log_diff_df = pd.concat(log_diff_list,axis=1)
log_diff_df.to_csv('log_diff_df.csv')
log_diff_df = pd.read_csv('log_diff_df.csv')
diff_combined_df= pd.read_csv('diff_combined_df.csv')
## Log Transform Dataframes

## Cannot take log after difference because of negatives
log_diff_list = [log_transform(diff_combined_df[f"{x}_diff"],diff_combined_df.index) for x in range(1,100)]
diff_log_df = pd.concat(log_diff_list,axis=1)
# Testing log transform on original Appliances
original_transform = adf_kpss_statistic(log_df.log_transform)
print(original_transform)
#%%
## Testing for Stationarity
adf_list = []
kpss_list = []
test = adf_kpss_statistic(diff_combined_df['10_diff'])
for x in range(1,101):
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

# Rolling mean/var
#%%
original_mean_var = cal_rolling_mean_var(df.Appliances)
log_mean_var = cal_rolling_mean_var(log_df.log_transform)
diff_10_mean_var = cal_rolling_mean_var(diff_combined_df['10_diff'])
diff_25_mean_var = cal_rolling_mean_var(diff_combined_df['25_diff'])
diff_50_mean_var = cal_rolling_mean_var(diff_combined_df['50_diff'])
log_diff_10_mean_var = cal_rolling_mean_var(log_diff_df['10_diff'])
log_diff_25_mean_var = cal_rolling_mean_var(log_diff_df['25_diff'])
log_diff_50_mean_var = cal_rolling_mean_var(log_diff_df['50_diff'])
## Plotting
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Original Data')
ax1.plot(original_mean_var.index, original_mean_var['Rolling Mean'])
ax1.set_ylabel('Rolling Mean')
ax2.plot(original_mean_var.index, original_mean_var['Rolling Variance'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Rolling Variance')
plt.savefig(image_folder+'rolling_original.png', dpi=1000)
plt.show()
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
# %%
# 10 log diff
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('50 Diff ')
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
#%%
