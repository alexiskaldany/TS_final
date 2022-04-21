# All functions take pd series as "series" input
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import seaborn as sns

############
"""
All prediction functions return an array equal to the imputed array + h_steps
For graphing, create extended index array equal to original array + h_steps

For split training - testing data:
option 1: return df with

"""
def diff(series,interval,index):
    l = len(series)
    diff = [0] * interval
    [diff.append(series[x+interval] - series[x]) for x in range(0,l-interval)]    
    df = pd.DataFrame(index=index)
    df[f"{interval}_diff"] = diff
    return df

def log_transform(series,index):
    log_series = np.log(series)
    df = pd.DataFrame(index=index)
    df['log_transform'] = log_series
    return df

def simple_time_series(y_column, time_column, df):
    """

    :param y_column:
    :param time_column:
    :param df:
    :return:
    """
    plt.figure(figsize=(8, 4), layout='constrained')
    plt.plot(time_column, y_column, data=df)
    plt.title(f'{y_column} by {time_column}')
    plt.xlabel(f'{time_column}')
    plt.ylabel(f'{y_column}')
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.grid()
    print(f"{y_column} saved!")
    return plt.show()

def plotting_rolling(name, df):
    """

    :param name:
    :param df:
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Rolling mean and var vs time')

    ax1.plot(df.index, df['Rolling Mean'])
    ax1.set_ylabel('Rolling Mean')

    ax2.plot(df.index, df['Rolling Variance'])
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rolling Variance')
    plt.show()
    return

def cal_rolling_mean_var(series):
    length = len(series)
    # date = []
    mean = []
    var = []
    for y in range(0, length):
        row_slice = slice(0, y)
        if y != 1:
            mean.append(series[row_slice].mean())
            var.append(series[row_slice].var())
        else:
            mean.append(series[row_slice].mean())
            var.append(0)
        # date.append(df.iloc[y, date_index])
    dict_rolling = {'Rolling Mean': mean, 'Rolling Variance': var}
    new_df = pd.DataFrame(dict_rolling, columns=['Rolling Mean', 'Rolling Variance'])
    
    return new_df


def correlation_coefficent_cal(x, y):
    x_mean = np.nanmean(np.array(x))
    y_mean = np.nanmean(np.array(y))
    x_r = np.subtract(x, x_mean)
    y_r = np.subtract(y, y_mean)
    numerator = np.dot(x_r, y_r)
    denom_x = np.nansum((x_r) ** 2)
    denom_y = np.nansum((y_r) ** 2)
    denominator = (denom_x * denom_y) ** (1 / 2)
    if denominator != 0:
        return round((numerator / denominator), 2)
    else:
        return print('Divide by zero??')


def kpss_test(timeseries):
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = [x for x in kpsstest[0:3]]
    crit_dict = kpsstest[3]
    crit_values = list(crit_dict.keys())
    for x in crit_values:
        kpss_output.append(crit_dict.get(x))
    kpss_cols = ['Test Statistic', 'p-value', 'Lags', '10%', '5%', '2.5%', '1%']
    kpss_dict = {x: y for x, y in zip(kpss_cols, kpss_output)}
    df = pd.DataFrame.from_dict([kpss_dict])
    print(kpss_dict)
    return df


def adf_cal(x, df):
    df = df.dropna()
    result = adfuller(df[x])
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    output = [x, result[0], result[1]]
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        output.append(value)

    cols = ['Column', 'ADF Statistic', 'p-value', '1% CV', '5% CV', '10% CV']
    dicta = {x: y for x, y in zip(cols, output)}
    df = pd.DataFrame(dicta, columns=['Column', 'ADF Statistic', 'p-value', '1% CV', '5% CV', '10% CV'],
                      index=['Column'])
    return df

def adf_kpss_statistic(timeseries):
    adf = adfuller(timeseries)[0]
    kpss_ = kpss(timeseries, regression='c', nlags="auto")[0]
    stats = [adf,kpss_]
    return stats

def acfunc(series, lag):
    if lag == len(series):
        return 0
    if lag == 0:
        return 1
    series = np.array(series)
    mean = series.mean()
    series_sub_mean = np.subtract(series, mean)
    shifted_right = series_sub_mean[:-lag]
    shifted_left = series_sub_mean[lag:]
    denominator = np.sum(np.square(series_sub_mean))
    numerator = np.sum(np.dot(shifted_right, shifted_left))
    r = numerator / denominator
    return round(r, 3)


# %%'
def acf_df(series, lag):
    lag_list = [x for x in range(-lag, lag + 1, 1)]
    acf_value = [1]
    for l in [x for x in range(1, lag + 1, 1)]:
        x = acfunc(series, l)
        acf_value.insert(0, x)
        acf_value.append(x)
    df = pd.DataFrame()
    df['lags'] = lag_list
    df['acf'] = acf_value
    return df


def stem_acf(name, df,n):
    plt.stem(df['lags'], df['acf'])
    plt.title(f'ACF plot of {name}')
    plt.xlabel('lags')
    plt.ylabel('Autocorrelation value')
    #markers = 1.96/((len(df)**(1/2)))
    #plt.setp(markers, color='red',markers='0')
    #plt.axhspan(-markers,markers,alpha=0.2,color='blue')
    plt.fill_between(df['lags'], (1.96 / ((n) ** (1/2))), ((-1.96) / ((n) ** (1 / 2))), color='thistle')
    return plt.show()


def rolling_average_1(series, h=0):
    prediction = []
    length = list(range(1, (len(series) + 1), 1))
    series = np.array(series)
    for index in length:
        prediction.append(series[:index].mean())
    if h > 0:
        h = list(range(1, (h + 1), 1))
        for step in h:
            prediction.append(series.mean())
    else:
        print('no steps')

    return prediction


def error_variance(array_1, array_2):
    return round(np.nanvar(array_2 - array_1), 2)


# def array_length_index(array):
#     length = np.shape(array)
#
#     return


def average_prediction(train_series, h_steps):
    prediction = [np.nan]
    average = np.array(train_series).mean()
    for index, x in np.ndenumerate(train_series):
        prediction.append(train_series[:index[0]].mean())
    forecast = np.full(fill_value=average, shape=(h_steps,))
    return np.array(prediction[:-1]), np.array(forecast[:-1])


def error(observation, prediction):
    try:
        residual = np.subtract(np.array(observation), np.array(prediction))
        squared_error = np.square(residual)
        return residual, squared_error
    except:
        residual = np.subtract(np.array(observation[1:]), np.array(prediction))
        squared_error = np.square(residual)
        return residual, squared_error


def plot_train_test_predict(train_index, train, test_index, test, predict_index, predict, title):
    fig, ax = plt.subplots()
    ax.plot(train_index, train, color='navy')
    ax.plot(test_index, test, color='red')
    ax.plot(predict_index, predict, color='green')
    plt.title(f'{title}')
    plt.xlabel('Time steps')
    plt.ylabel('Value of Data')
    plt.legend(['train', 'test', 'prediction'])
    return plt.show()


def q_value(residuals, lag):
    ACF_df = acf_df(residuals, lag)
    T = len(residuals)
    squared_acf = np.sum(np.square(ACF_df['acf']))
    return T * squared_acf


def naive_prediction(train_series, h_steps):
    train_series = np.array(train_series)
    prediction = []
    for index, x in np.ndenumerate(train_series):
        prediction.append(train_series[index[0] - 1])
    forecast = np.full(fill_value=train_series[-1], shape=(h_steps,))
    return np.array(prediction), np.array(forecast)


def naive_rolling(series, h=0):
    prediction = [np.nan]
    length = list(range(1, (len(series) + 1), 1))
    series = np.array(series)
    for index in length:
        prediction.append(series[(index - 1)])
    if h > 0:
        h = list(range(1, (h + 1), 1))
        for step in h:
            prediction.append(series[-1])
    else:
        print('no steps')

    return prediction[:-1]


def drift_rolling(series, h=0):
    series = np.array(series)
    length = list(range(2, (len(series) + 1), 1))
    # series = np.append(series, [np.nan] * 1)
    prediction = [np.nan,np.nan]
    for index in length:
        drift = series[index - 1] + (h * ((series[index - 1] - series[0]) / (index - 1)))
        prediction.append(drift)
    if h > 0:
        h = list(range(1, (h + 1), 1))
        for step in h:
            prediction.append(prediction[-1])
    else:
        print('no steps')

    return prediction[:-2]


def drift_prediction(series, h_steps):
    prediction = []
    series = np.array(series)
    # series = np.insert(series, [np.nan] * 1)
    steps = list(range(1, (h_steps + 1), 1))
    for h in steps:
        drift = series[-1] + (h * ((series[-1] - series[0]) / (len(series) - 2)))
        prediction.append(drift)
    return prediction


def ses_rolling(series, extra_periods=1, alpha=0.5):
    series = np.array(series)  # Transform the input into a numpy array
    cols = len(series)  # Historical period length
    series = np.append(series, [np.nan] * extra_periods)  # Append np.nan into the demand array to cover future periods
    f = np.full(cols + extra_periods, np.nan)  # Forecast array
    f[1] = series[0]  # initialization of first forecast
    # Create all the t+1 forecasts until end of historical period
    for t in range(2, cols + 1):
        f[t] = alpha * series[t - 1] + (1 - alpha) * f[t - 1]
    f[cols + 1:] = f[t]  # Forecast for all extra periods
    return f[:-extra_periods]


def ses_prediction(series, h_steps=1, alpha=0.5):
    series = np.array(series)  # Transform the input into a numpy array
    cols = len(series)  # Historical period length
    series = np.append(series, [np.nan] * h_steps)  # Append np.nan into the demand array to cover future periods
    f = np.full(cols + h_steps, np.nan)  # Forecast array
    f[1] = series[0]  # initialization of first forecast
    # Create all the t+1 forecasts until end of historical period
    for t in range(2, cols + 1):
        f[t] = alpha * series[t - 1] + (1 - alpha) * f[t - 1]
    f[cols + 1:] = f[t]  # Forecast for all extra periods
    return f[-h_steps:]


def mse_calc(obs_series, pred_series):
    return round((np.sum(np.square(np.subtract(np.array(pred_series), np.array(obs_series))) / len(obs_series))), 2)


def residuals(array_1, array_2):
    return array_2 - array_1

def lse(x_matrix,y_array):
    invert = np.linalg.inv(np.dot(np.transpose(x_matrix),x_matrix))
    transpose_y_array = np.dot(np.transpose(x_matrix),y_array)
    return np.dot(invert,transpose_y_array)

def coeff(X, Y):
    '''The function returns the values of the multiple regression model coefficients'''
    coef= np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)),X.transpose()), Y)
    return coef

def worst_feature(p_value_series):
    return p_value_series.idxmax()

def new_x_train(worst_feature,old_x_train):
    return old_x_train.drop(columns=[worst_feature])

def aic_bic_rsquared_df(fitted_model):
    return pd.DataFrame.from_dict({'index':[0],'aic':fitted_model.aic,'bic':fitted_model.bic,'adj_rsquared':fitted_model.rsquared_adj})




def recursive_selection(x_train,y_train,df):
    fit = sm.OLS(y_train, x_train).fit()
    remove_this_feature = worst_feature(fit.pvalues)
    new_x = new_x_train(remove_this_feature, x_train)
    new_x_df = aic_bic_rsquared_df(sm.OLS(y_train, new_x).fit())
    new_df = pd.concat([df, new_x_df])

    #fit = OLS_fit(x_train,y_train)
    # remove_this_feature = worst_feature(fit.pvalues)
    # new_x = new_x_train(remove_this_feature,x_train)
    # new_df = pd.concat([df, aic_bic_rsquared_df(OLS_fit(new_x,y_train))])
    if len(list(new_x.columns))>1:
        recursive_selection(new_x,y_train,new_df)
    else:
        return new_df

def generalized_AR():
    samples = int(input('Sample number:'))
    order = int(input('Order number:'))
    parameters = []
    for value in range(0,order):
        parameters.append(float(input(f'Parameter {value}:')))
    np.random.seed(2)
    e=np.random.randn(samples)
    y=np.zeros(len(e))
    for i in range(len(e)):
        if i==0:
            y[i]=e[i]
        elif i==1:
            y[i]=parameters[0]*y[i-1]+e[i]
        else:
            y[i]+= e[i]
            for k in range(order):
                y[i]+=parameters[k]*y[i-(k+1)]
        z1 = y[1:len(y) - 1]
        z2 = y[:len(y) - 2]
        x_2 = np.array([z1,z2]).T
    coef = coeff(x_2,y[2:])
    print('The estimated parameters are:',coef)
    print('The true parameters are:', parameters)
    return print('Done')

def rolling_average_non_weighted(array,n):
    """Rolling average
    Must be greater than 2

    Args:
        a (array): np.array use to create moving average
        n (int): number of lags, must be greater than 2
    Returns: np.array of moving averages
    """
    n = int(n)
    odd = True if n%2 ==1 else False 
    if n <=2:
        return print("Order must be greater than 2")
    elif odd == True:
        start= np.array([np.nan] * int((n-1)/2))
        average = np.array(np.lib.stride_tricks.sliding_window_view(array,n).mean(axis=1))
        end  = np.array([np.nan] * int((n-1)/2))
        full =np.append(np.append(start,average),end)
        return full
    else:
        start = np.array([np.nan] * int(n/2))
        average = np.array(np.lib.stride_tricks.sliding_window_view(array,n).mean(axis=1))
        end =  np.array(([np.nan] * int((n-1)/2)))
        full =np.append(np.append(start,average),end)
        return full
def even_rolling_average(array,n):
        start = np.array([np.nan] * int(n/2))
        average = np.array(np.lib.stride_tricks.sliding_window_view(array,n).mean(axis=1))
        end =  np.array(([np.nan] * int((n-1)/2)))
        full =np.append(np.append(start,average),end)
        return full
    
def odd_rolling_average(array,n):
        start= np.array([np.nan] * int((n-1)/2))
        average = np.array(np.lib.stride_tricks.sliding_window_view(array,n).mean(axis=1))
        end  = np.array([np.nan] * int((n-1)/2))
        full =np.append(np.append(start,average),end)
        return full
def odd_or_even_rolling_average(array):
    length = len(array)
    order_1 = int(input("Enter the order of the moving average:"))
    if order_1 <=2:
        return print('Order must be greater than 2')
    elif order_1%2 == 0:
        order_2 = int(input('Enter an even valued folding order, must be greater than 1'))
        if order_2 <2 or order_2%2 !=0:
            print("Invalid folding order")
            pass
        else:
            output = even_rolling_average(even_rolling_average(array,order_1),order_2)
            return output
    elif order_1%2 ==1:
        return odd_rolling_average(array,order_1)
            
def mse(errors):
       return  np.sum(np.power(errors,2))/len(errors)
        
        
def strength_of_trend(residual,trend):
    var_resid = np.nanvar(residual)
    var_resid_trend = np.nanvar(np.add(residual,trend))
    return 1-(var_resid/var_resid_trend)

def strength_of_seasonal(residual,seasonal):
    var_resid = np.nanvar(residual)
    var_resid_seasonal = np.nanvar(np.add(residual,seasonal))
    return 1-(var_resid/var_resid_seasonal)
    
    

def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()  
    return 

def arma_input_process_and_sample():
    samples = int(input('Enter the number of samples'))
    mean_wn = int(input('Enter mean of white noise'))
    var_wn = int(input('Enter variance of white noise'))
    ar_order = int(input('Enter AR order'))
    ma_order = int(input('Enter MA order'))
    ar_coeff = []
    [ar_coeff.append(int(input(f'Enter #{x} AR coefficient, hint: range is (0,1)'))) for x in ar_order]
    ma_coeff = []
    [ma_coeff.append(int(input(f'Enter #{x} MA coefficient, hint: range is (0,1)'))) for x in ma_order]
    arparams = np.array(ar_coeff*(-1))
    maparams = np.array(ma_coeff)
    ar = np.insert(arparams,1)
    ma = np.insert(maparams,1)
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    arma_sample = np.array(arma_process.generate_sample(nsample=samples,scale=var_wn)) + mean_wn
    return arma_process,arma_sample

# Q3
def arma_process(ar_param,ma_param,samples):
    np.random.seed(2)
    arparams = np.array(ar_param)
    maparams = np.array(ma_param)
    # print(arparams)
    # print(maparams)
    # ar = np.insert(arparams,0,[1])
    
    # ma = np.insert(maparams,0,[1])
    ar = np.r_[1, -arparams] 
    ma = np.r_[1, maparams]
    arma_process = sm.tsa.ArmaProcess(ar, ma,nobs=samples)
    return arma_process

def gpac_matrix(ry, j, k):
    # generates left most column
    col = []
    for i in range(j):
        col.append(ry[i + 1] / ry[i])
    # convert values to dataframe
    # number of rows equal to [0,j)
    table = pd.DataFrame(col, index=np.arange(0, j).tolist())
    # list for values in the GPAC matrix
    # this list compiles the various pacf values
    val = []
    # for K in GPAC, you do not want to include column 1 as it will all be 1's
    # first loop goes through the (k,k) matrices 
    for a in range(2, k + 1):
        # second loop goes through the (k,k) matrices as j increases
        for f in range(j): # f is j
            b_val = [] # numerator
            t_val = [] # denominator
            for d in range(a): #
                den = []
                for h in range(a):
                    den.append(ry[abs(f - h + d)])
                b_val.append(den.copy())
                t_val.append(den.copy())
            mes = a
            for l in range(a):
                t_val[l][mes - 1] = ry[l + 1 + f]

            pac = np.linalg.det(t_val) / np.linalg.det(b_val)
            val.append(pac)
    # reshape the GPAC value so there is no 0 row in k
    GPAC = np.array(val).reshape(k -1, j)
    #correctly transpose the data
    GPAC_T = pd.DataFrame(GPAC.T)
    GPAC_F = pd.concat([table, GPAC_T], axis=1)
    GPAC_F.columns = list(range(1, k + 1))
    return GPAC_F

def gpac_plot(gpac_df):
    plt.figure()
    sns.heatmap(gpac_df,  annot=True)
    plt.title('GPAC Table')
    plt.xlabel('k values')
    plt.ylabel('j values')
    plt.show()

def gpac_with_input():
    samples = int(input('Enter the number of samples'))
    mean_wn = int(input('Enter mean of white noise'))
    var_wn = int(input('Enter variance of white noise'))
    ar_order = int(input('Enter AR order'))
    ma_order = int(input('Enter MA order'))
    ar_coeff = []
    [ar_coeff.append(int(input(f'Enter #{x} AR coefficient, hint: range is (0,1)'))) for x in ar_order]
    ma_coeff = []
    [ma_coeff.append(int(input(f'Enter #{x} MA coefficient, hint: range is (0,1)'))) for x in ma_order]
    arparams = np.array(ar_coeff*(-1))
    maparams = np.array(ma_coeff)
    ar = np.insert(arparams,1)
    ma = np.insert(maparams,1)
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    arma_sample = np.array(arma_process.generate_sample(nsample=samples,scale=var_wn)) + mean_wn
    k = input("What K?")
    j = input("What J?")
    lags = input("What lags?")
    arma_acf = arma_process.acf(lags=lags)
    gpac_df = gpac_matrix(arma_acf,k,j)
    gpac_plot(gpac_df)
    return

def gpac_from_arma_process(ar,ma,lags,samples,k,j):
    process = arma_process(ar,ma,samples)
    y = process.generate_sample(nsample=samples)
    ry = process.acf(lags=lags)
    gpac_df = gpac_matrix(ry,k,j)
    gpac_plot(gpac_df)
    ACF_PACF_Plot(y,lags)
    return gpac_df

def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()  
    
def train_test(series,train_share):
    slicer = int((len(series)*train_share))
    train = series[:slicer+1]
    test = series[slicer:]
    return train, test

def one_step_prediction_from_ar_ma(ar,ma,series,h):
    from_y =[ma[0]-ar[0],ma[1]-ar[1]]
    from_predictions = [-ma[0],-ma[1]]
    if h==1:
        predictions = [0]
        for i in range(1,len(series)):
            sum = 0
            sum += from_y[0]*series[i] + from_y[1]*series[i-1]
            sum += from_predictions[0]*predictions[i-1] + from_predictions[1]*predictions[i-2]
            predictions.append(sum)
    elif h>1:
        predictions = [series[-1]]
        for i in range(0,h):
            sum = 0 
            sum += ar[0]*predictions[i]
            sum += ar[1]*predictions[i-1]
            predictions.append(sum)
        
    return predictions


def train_test_predict_from_array(train,test,predict,title):
    fig, ax = plt.subplots()
    index = len(train) + len(test)
    ax.plot([x for x in range(0,len(train+1))], train, color='navy')
    ax.plot([x for x in range(len(train+1),len(train+1)+len(test+1))], test, color='red')
    ax.plot([x for x in range(0,len(train+1))], predict, color='green')
    plt.title(f'{title}')
    plt.xlabel('Time steps')
    plt.ylabel('Value of Data')
    plt.legend(['train', 'test', 'prediction'])
    return plt.show()

def parameter_CI(ar_list,ma_list):
    array = np.array(ar_list+ma_list)
    matrix = np.reshape(array,(2,2))
    print(np.cov(matrix))
    cov_flat = np.sqrt(np.cov(matrix).flatten())*2
    ci_list = [[array[i]+cov_flat[i],array[i]-cov_flat[i]] for i in range(len(array))]
    
    return ci_list

def Autocorr(array, lag):
    mean_array = np.mean(array)
    length_df = len(array)
    num = []
    for i in range(lag, length_df):
        numer = (array[i] - mean_array) * (array[i - lag] - mean_array)
        num.append(numer)
    num_sum = sum(num)
    den= []
    for j in range(0, length_df):
        denom = (array[j] - mean_array) ** 2
        den.append(denom)
    den_sum = sum(den)
    result_array = round(num_sum / den_sum, 4)
    return result_array