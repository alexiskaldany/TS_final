# Appliance Energy Use Prediction

                Author: Alexis Kaldany
                Spring 2022
                DATS_6313_10
                Overseen by Professor Reza Jafari

-----------------------

## Abstract

-----------------------

## Introduction

-----------------------

### Description of Data

- The data set is at 10 min for about 4.5 months. The house temperature and humidity conditions were monitored with a ZigBee wireless sensor network. Each wireless node transmitted the temperature and humidity conditions around 3.3 min. Then, the wireless data was averaged for 10 minutes periods. The energy data was logged every 10 minutes with m-bus energy meters.

- Weather from the nearest airport weather station (Chievres Airport, Belgium) was downloaded from a public data set from Reliable Prognosis (rp5.ru), and merged together with the experimental data sets using the date and time column.

- Two random variables have been included in the data set for testing the regression models and to filter out non predictive attributes (parameters).

-----------------------

### Visualization of Variable Locations

![House Variable Visualization](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/variable_location_within_house.png)

-----------------------

### Table of Variables

- All temperatures in Celsius
- All humidity is in % terms
- All energy use in watts per hour

| Variable  | Description  | Variable  | Description  |
|--- |--- |--- |--- |
| date | 10 minute intervals  |  Appliances | energy use of appliances  |
| lights  | energy use of light fixtures  |  Press_mm_hg | Pressure  |
| RH_1  | Humidity in kitchen  |   T1| Temp in kitchen |
|RH_2   | Humidity in living room  | T2  | Temp in living room |
| RH_3  | Humidity in laundry area |T3   | Temp in laundry area |
|RH_4   | Humidity in office room | T4  | Temp in office room |
|RH_5   | Humidity in bathroom | T5  | Temp in bathroom|
| RH_6  | Humidity in north area | T6  | Temp in north area |
| RH_7  | Humidity in ironing room | T7  | Temp in ironing room |
| RH_8  | Humidity in teenager room |  T8 | Temp in teenager room |
| RH_9  | Humidity in parents room |  T9 | Temp in parents room |
|  RH_out | Humidity outside  |  T_out | Temp outside  |
|  Windspeed | in m/s  |Visibility   |  kilometers |
| rv1  | random variable 1  | rv2  | andom variable 2  |
| Tdewpoint  |  A*C |   |   |
-----------------------

### 1a

![Dependent Variable vs Time](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/1a-dependent.png)

![Stem Original](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/Original-Appliances)

![Heatmap](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/1d-heatmap-corr.png)

![ACF-PACF original](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/1c-ACF-PACF-Original.png)

## Stationarity

![rolling_original](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/rolling_original.png)

![adf_stats](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/1a-ADF-Stats.png)

![kpss_stats](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/1a-KPSS-Stats.png)

![rolling_diff_150](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/rolling_diff_150.png)

![rolling_log_diff_150](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/rolling_log_diff_150.png)

## Decomposition

![Decomposition-144-ACF-PACF-Original](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/Decomposition-144-ACF-PACF-Original.png)

![Appliances-144-diff-ACF](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/Appliances-144-diff-ACF.png)

![Original-Decomposition](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/Original-Decomposition.png)

![144-Diff-Decomposition](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/144-Diff-Decomposition.png)

![Cleaner-144-Decomposition](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/Cleaner-144-Decomposition.png)

![Seasonal-Adjusted-Decomposition](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/Seasonal-Adjusted-Decomposition.png)

![Detrended-Decomposition](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/Detrended-Decomposition.png)

## Base Models

### Holt-Winters

![rolling-HW-prediction](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/rolling-HW-prediction.png)

![HW-Train-Test-Predict](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/HW-Train-Test-Predict.png)

![HW-Test-Predict](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/HW-Test-Predict.png)

![H-W-Train](H-W-Train.png)

![HW-Test_PACF](HW-Test_PACF.png)

### Average Method
![Average-train-test-predict](Average-train-test-predict.png)

![Average-test-predict](Average-test-predict.png)

![Average-Error-ACF](Average-Error-ACF.png)

### Naive Method

![Naive-train-test-predict.png](Naive-train-test-predict.png)

![Naive-test-predict.png](Naive-test-predict.png)

![Stem-ACF-Naive-Err](Stem-ACF-Naive-Err.png)

### Drift Method

![Drift-Train-Test-Predict.png](Drift-Train-Test-Predict.png)

![]()


![]()
![]()
![]()



## Backwards Selection and Linear Model
![]()
![]()
![]()
![]()
![]()
![]()
![]()
## ARMA (3,0)
![]()
![]()
![]()
![]()
![]()
## ARMA(3,3)
![]()
![]()
![]()
![]()
![]()
![]()
![]()

## ARIMA(3,0,0) x (0,3,0,12)
![]()
![]()
![]()
![]()
![]()
![]()
![]()
![]()
## LSTM
![]()
![]()
![]()

## Model Summary
![]()
## Conclusion
![]()
## Citations

## Appendix + Code
