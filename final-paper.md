# Appliance Energy Use Prediction

                Author: Alexis Kaldany
                Spring 2022
                DATS_6313_10
                Overseen by Professor Reza Jafari


## Abstract

## Introduction

### Description of Data

- The data set is at 10 min for about 4.5 months. The house temperature and humidity conditions were monitored with a ZigBee wireless sensor network. Each wireless node transmitted the temperature and humidity conditions around 3.3 min. Then, the wireless data was averaged for 10 minutes periods. The energy data was logged every 10 minutes with m-bus energy meters.

- Weather from the nearest airport weather station (Chievres Airport, Belgium) was downloaded from a public data set from Reliable Prognosis (rp5.ru), and merged together with the experimental data sets using the date and time column. 

- Two random variables have been included in the data set for testing the regression models and to filter out non predictive attributes (parameters).

### Visualization of Variable Locations

![House Variable Visualization](https://raw.githubusercontent.com/alexiskaldany/TS_final/main/final-images/variable_location_within_house.png)

### Table of Variables

- All temperatures in Celsius
- All humidity is in % terms
- All energy use in watts per hour

| Variable  | Description  | Variable  | Description  |
|--- |--- |--- |--- |
| date | 10 minute intervals  |  Appliances | energy use of appliances  |
| lights  | energy use of light fixtures  |  Press_mm_hg |   |
| RH_1  | Humidity in kitchen  |   T1| Temp in kitchen |
|RH_2   | Humidity in living room  | T2  | Temp in living room |
| RH_3  | Humidity in laundry area |T3   | Temp in laundry area |
|RH_4   | Humidity in office room | T4  | Temp in office room |
|RH_5   | Humidity in bathroom | T5  | Temp in bathroom|
| RH_6  | Humidity in north area | T6  | Temp in north area |
| RH_7  | Humidity in ironing room | T7  | Temp in ironing room |
| RH_8  | Humidity in teenager room |  T8 | Temp in teenager room |
| RH_9  | Humidity in parents room |  T9 | Temp in parents room |
|  RH_out |   |  T_out |   |
|  Windspeed |   |Visibility   |   |
| rv1  |   | rv2  |   |
| Tdewpoint  |   |   |   |

### 1a
! [Dependent Variable vs Time]("")

