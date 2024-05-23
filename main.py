import random

import pytz
import numpy as np
import pandas as pd
import holidays as hd
from datetime import datetime
from meteostat import Point, Daily
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import pacf
from fastai.tabular.model import emb_sz_rule


def read_in():

    df = pd.read_csv('electric-chargepoint-analysis-2017-raw-domestics-data.csv', delimiter=',', nrows=70000)
    print("Number of rows read:", len(df))

    columns_of_interest = ['ChargingEvent', 'StartDate', 'StartTime', 'EndDate', 'EndTime', 'Energy',
                           'PluginDuration']

    df_subset = df[columns_of_interest]

    df_subset['UTCTransactionStart'] = pd.to_datetime(df_subset['StartDate'] + ' ' + df_subset['StartTime'])
    df_subset['UTCTransactionStop'] = pd.to_datetime(df_subset['EndDate'] + ' ' + df_subset['EndTime'])


    # print(df_subset.head(10))
    # Asif
    df_subset.sort_values(by=['UTCTransactionStart'], inplace=True)
    # No NA's in my dataset but still use it
    df_subset.dropna(subset=['PluginDuration'], inplace=True)

    return df_subset

# New methods from Asif
def energy_segmentation(input_df, interval_minutes=60):
    new_rows = []
    interval = timedelta(minutes=interval_minutes)

    # Convert 'UTCTransactionStart' to datetime format

    for index, row in input_df.iterrows():
        current_time = row['UTCTransactionStart']
        done_charging_time = row['UTCTransactionStop']
        kwh_delivered = row['Energy']
        total_duration = float(row['PluginDuration'])


        while current_time < done_charging_time:
            new_row = row.copy()
            new_row['connectionTime'] = current_time
            new_row['doneChargingTime'] = min(current_time + interval, done_charging_time)
            interval_duration = (new_row['doneChargingTime'] - current_time).total_seconds()
            new_row['chargingTime'] = interval_duration / 60
            new_row['kWhDelivered'] = (interval_duration / total_duration) * kwh_delivered
            new_rows.append(new_row)
            current_time += interval

    result_df = pd.DataFrame(new_rows)
    result_df.reset_index(drop=True, inplace=True)
    return result_df

'''
def charging_time_and_idle_features(df_input):
    df_input['UTCTransactionStart'] = pd.to_datetime(df_input['UTCTransactionStart'])
    df_input['UTCTransactionStop'] = pd.to_datetime(df_input['UTCTransactionStop'])

    #modified Asif version caus dt.total_seconds() dont work

    df_input['StartInSeconds'] = df_input['UTCTransactionStart'].apply(lambda x: int(x.timestamp()))
    df_input['StopInSeconds'] = df_input['UTCTransactionStop'].apply(lambda x: int(x.timestamp()))

    df_input['totalChargingTime'] = (df_input['PluginDuration'] * 3600  - (df_input['StartInSeconds']) / 60)
    df_input['totalIdleTime'] = ((df_input['StopInSeconds'] - (df_input['PluginDuration'] * 3600)) / 60)
    df_input['totalParkingTime'] = df_input['totalIdleTime'] + df_input['totalChargingTime']
    return df_input
'''

def timestamp_format(dataframe):
    columns_tobe_converted = ['UTCTransactionStart', 'UTCTransactionStop']
    for column_name in columns_tobe_converted:
        dataframe[column_name] = pd.to_datetime(dataframe[column_name])
        dataframe[column_name] = dataframe[column_name].dt.strftime('%a, %d %b %Y %H:%M:%S GMT')
    return dataframe


def timezonecorrection(dataframe):
    columns_tobe_converted = ['UTCTransactionStart', 'UTCTransactionStop']

    for column in columns_tobe_converted:
        dataframe[column] = pd.to_datetime(dataframe[column], errors='coerce')
        print(f"Vor Hinzufügen einer Stunde: {dataframe[column]} \n")
        dataframe[column] = dataframe[column] + pd.Timedelta(hours=1)
        dataframe[column] = dataframe[column].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        dataframe[column] = pd.to_datetime(dataframe[column], format='%Y-%m-%d %H:%M:%S.%f')
        print(f"Nach Hinzufügen einer Stunde: {dataframe[column]} \n")

    return dataframe
'''
def process_data(duplicated_df, unique_hours, n_samples=1, SEED=None):
    duplicated_df = duplicated_df.assign(hour=duplicated_df['doneChargingTime'].dt.hour)
    df_filtered = duplicated_df[duplicated_df['hour'].isin(unique_hours)]
    df_final = None

    for hour in unique_hours:
        df_hour = df_filtered[df_filtered['hour'] == hour]
        grouped = df_hour.groupby('stationID').size().reset_index(name='count')
        grouped.sort_values('count', ascending=False, inplace=True)

        selected_stationID = None
        for stationID in grouped['stationID']:
            if not df_filtered[(df_filtered['hour'] == hour) & (df_filtered['stationID'] == stationID)].empty:
                selected_stationID = stationID
                break

        if selected_stationID is None:
            df_selected = df_filtered[df_filtered['hour'] == hour].sample(n=min(n_samples, len(df_filtered[df_filtered['hour'] == hour])), random_state=SEED)
        else:
            df_selected = df_filtered[(df_filtered['hour'] != hour) | (df_filtered['stationID'] == selected_stationID)]

        if df_final is None:
            df_final = df_selected
        else:
            df_final = pd.concat([df_final, df_selected])

    df_final.drop_duplicates(subset='doneChargingTime', inplace=True)
    return df_final
'''


def create_temporal_features(df):
    gb_holidays = hd.GB(state='ENG', observed=False)

    df['UTCTransactionStop'] = pd.to_datetime(df['UTCTransactionStop'])
    df['Hour_of_Day'] = df['UTCTransactionStop'].dt.hour
    df['Day_Of_Week'] = df['UTCTransactionStop'].dt.dayofweek
    df['Day_Of_year'] = df['UTCTransactionStop'].dt.dayofyear
    df['Month_Of_Year'] = df['UTCTransactionStop'].dt.month

    conditions = [
    (df['Hour_of_Day'] >= 0) & (df['Hour_of_Day'] < 4) & (df['Energy'] != 0),
    (df['Hour_of_Day'] >= 4) & (df['Hour_of_Day'] < 8) & (df['Energy'] != 0),
    (df['Hour_of_Day'] >= 8) & (df['Hour_of_Day'] < 12) & (df['Energy'] != 0),
    (df['Hour_of_Day'] >= 12) & (df['Hour_of_Day'] < 16) & (df['Energy'] != 0),
    (df['Hour_of_Day'] >= 16) & (df['Hour_of_Day'] < 20) & (df['Energy'] != 0),
    (df['Hour_of_Day'] >= 20) & (df['Hour_of_Day'] < 24) & (df['Energy'] != 0)
    ]

    categories = ['lateNight', 'earlyMorning', 'morning', 'midDay', 'evening', 'night']

    df['sessionOfDay'] = pd.Categorical(np.select(conditions, categories), categories=categories)
    df['sessionOfDay'] = df['sessionOfDay'].cat.codes

    df['Time_of_day_0_4'] = ((df['Hour_of_Day'] >= 0) & (df['Hour_of_Day'] < 4) & (df['Energy'] != 0)).astype(int)
    df['Time_of_day_4_8'] = ((df['Hour_of_Day'] >= 4) & (df['Hour_of_Day'] < 8) & (df['Energy'] != 0)).astype(int)
    df['Time_of_day_8_12'] = ((df['Hour_of_Day'] >= 8) & (df['Hour_of_Day'] < 12) & (df['Energy'] != 0)).astype(int)
    df['Time_of_day_12_16'] = ((df['Hour_of_Day'] >= 12) & (df['Hour_of_Day'] < 16) & (df['Energy'] != 0)).astype(int)
    df['Time_of_day_16_20'] = ((df['Hour_of_Day'] >= 16) & (df['Hour_of_Day'] < 20) & (df['Energy'] != 0)).astype(int)
    df['Time_of_day_20_24'] = ((df['Hour_of_Day'] >= 20) & (df['Hour_of_Day'] < 24) & (df['Energy'] != 0)).astype(int)

    def categorize_day(timestamp):
        date = timestamp.date()
        if date in gb_holidays:
            return 'Holiday: ' + gb_holidays[date]
        elif timestamp.weekday() == 5:
            return 'WeekendSaturday'
        elif timestamp.weekday() == 6:
            return 'WeeekendSunday'
        else:
            return 'Weekday'

    df['DayCategory'] = df['UTCTransactionStop'].apply(categorize_day)

    df['Season'] = (df['Month_Of_Year'] % 12 + 3) // 3

    london_tz = pytz.timezone('Europe/London')
    df['daylightSaving'] = df['UTCTransactionStop'].dt.tz_localize('UTC').dt.tz_convert(london_tz).apply(lambda x: int(x.dst().total_seconds() != 0))

    return df
def day_categories(df, source_column):
    df['Weekday'] = df[source_column].apply(lambda x: 1 if 'Weekday' in x else 0)
    df['Weekend'] = df[source_column].apply(lambda x: 1 if 'WeekendSaturday' in x or 'WeekendSunday' in x else 0)
    df['holiday'] = df[source_column].apply(lambda x: 1 if x.startswith('Holiday:') else 0)
    return df

def intmt_demand(ts, method='cro', alpha=0.1, beta=0.1):
    """
    Perform smoothing on an intermittent time series, ts, and return
    a forecast array

    Parameters
    ----------
    ts : (N,) array_like
        1-D input array
    method : {'cro', 'sba', 'tsb'}
        Forecasting method: Croston, Syntetos-Boylan Approximation
        and Teunter-Syntetos-Babai method
    alpha : float
        Demand smoothing factor, `0 < alpha < 1`, default = 0.1
    beta : float
        Interval smoothing factor, `0 < beta < 1`, default = 0.1

    Returns
    -------
    forecast : (N+1,) ndarray
        1-D array of forecasted values
    """
    ts_trim = np.trim_zeros(ts, 'f')
    n = len(ts_trim)
    z = np.zeros(n)
    p = np.zeros(n)
    p_idx = np.flatnonzero(ts)
    p_diff = np.diff(p_idx, prepend=-1)
    p[0] = np.mean(p_diff)
    z[0] = ts[p_idx[0]]
    if method in {'cro', 'sba'}:
        q = 1
        for i in range(1,n):
            if ts_trim[i] > 0:
                z[i] = alpha*ts_trim[i] + (1-alpha)*z[i-1]
                p[i] = beta*q + (1-beta)*p[i-1]
                q = 1
            else:
                z[i] = z[i-1]
                p[i] = p[i-1]
                q += 1
        f = z / p
        if method == 'sba':
            f *= (1 - beta/2)
    elif method == 'tsb':
        p[0] = 1 / p[0]
        for i in range(1,n):
            if ts_trim[i] > 0:
                z[i] = alpha*ts_trim[i] + (1-alpha)*z[i-1]
                p[i] = beta + (1-beta)*p[i-1]
            else:
                z[i] = z[i-1]
                p[i] = (1 - beta)*p[i-1]
        f = p * z
    nan_arr = [np.nan] * (len(ts)-n)
    return np.concatenate((nan_arr, f))

def smoothing_with_best_params(dataframe, column_names, method='cro', alpha_range=None, beta_range=None):

    df = dataframe.copy()

    for column_name in column_names:
        # print(f"Analysis for column: {column_name}")

        df_column = df[['UTCTransactionStop', column_name]]
        total_rows = len(df_column)
        zero_count = len(df_column[df_column[column_name] == 0])
        non_zero_count = len(df_column[df_column[column_name] != 0])
        zero_percentage = (zero_count / total_rows) * 100
        non_zero_percentage = (non_zero_count / total_rows) * 100
        # print(f"Percentage of zero values in {column_name}: {zero_percentage:.2f}%")
        # print(f"Percentage of non-zero values in {column_name}: {non_zero_percentage:.2f}%")

        if alpha_range is None:
            alpha_range = np.linspace(0.01, 0.99, 10)
        if beta_range is None:
            beta_range = np.linspace(0.01, 0.99, 10)

        best_alpha = None
        best_beta = None
        lowest_mse = float('inf')

        for alpha in alpha_range:
            for beta in beta_range:
                forecasting = intmt_demand(df_column[column_name].values, method=method, alpha=alpha, beta=beta)
                mse = mean_squared_error(df_column[column_name], forecasting)
                if mse < lowest_mse:
                    lowest_mse = mse
                    best_alpha = alpha
                    best_beta = beta

        # print("Best Alpha:", best_alpha)
        # print("Best Beta:", best_beta)
        # print("Lowest MSE:", lowest_mse)

        forecasting = intmt_demand(df_column[column_name].values, method=method, alpha=best_alpha, beta=best_beta)
        df.loc[:, f'Smoothed_{column_name}'] = forecasting

        data_column1 = df_column[column_name].values
        data_column2 = df[f'Smoothed_{column_name}'].values

        num_bins = 10
        hist_column1, _ = np.histogram(data_column1, bins=num_bins, density=True)
        hist_column2, _ = np.histogram(data_column2, bins=num_bins, density=True)
        kl_divergence = entropy(hist_column1, hist_column2)
        print(f"KL Divergence for {column_name}: {kl_divergence}")

    return df

def expanding_mean_std_weighted_avg(dataframe, window_size):

    dataframe['expanding_mean'] = dataframe['Energy'].expanding(window_size).mean()
    dataframe['expanding_std'] = dataframe['Energy'].expanding(window_size).std()

    # weights have to be the same size as window_size
    weights = [0.4, 0.2] + [0.4 / (window_size - 2)] * (window_size - 2)
    weights = pd.Series(weights) / pd.Series(weights).sum()  # Normalising of the weights
    dataframe['weighted_avg'] = dataframe['Energy'].rolling(window=window_size).apply(lambda x: (x * weights).sum(), raw=True)

    dataframe.dropna(inplace=True)

    return dataframe
def create_lag_features(dataframe, target, lags=None, thres=0.2):

    scaler = StandardScaler()
    features = pd.DataFrame()

    if lags is None:
        partial = pd.Series(data=pacf(target, nlags=2))
        lags = list(partial[np.abs(partial) >= thres].index)

    df = pd.DataFrame()
    if 0 in lags:
        lags.remove(0)
    for l in lags:
        df[f"lag_{l}"] = target.shift(l)

    # features = pd.DataFrame(scaler.fit_transform(df[df.columns]),columns=df.columns)

    features = df
    features.index = target.index

    final_df = pd.concat([dataframe, features], axis=1)
    final_df.dropna(inplace=True)

    return final_df

def unique_value_count(df, df_name):
    columns_to_count_unique = ['ChargingEvent', 'StartDate']
    unique_count_list = []

    for column in columns_to_count_unique:
        unique_count = df[column].nunique()
        unique_count_list.append(unique_count)
        print(f"For {df_name}: Number of unique values in '{column}': {unique_count}")
    return unique_count_list



def get_emb_sz_list(dims: list):
    """
    For all elements in the given list, find a size for the respective embedding through trial and error
    Each element denotes the amount of unique values for one categorical feature
    Parameters
    ----------
    dims : list
        a list containing a number of integers.
    Returns
    -------
    list of tupels
        a list containing the amount of unique values and respective embedding size for all elements.
    """
    return [(d, emb_sz_rule(d)) for d in dims]


def column_order(dataframe):
    desired_columns = ['doneChargingTime', 'clusterID', 'CA 91125', 'CA 91106',	'CA 91109',	'CA 95136', 'UniqueEVSEType', 'EVSEcount', 'EVSE_AV', 'EVSE_CC32', 'EVSE_CC64',
                       'EVSE_DX', 'EVSE_TWC', 'UniquechargingReqType', 'claimedCount', 'unclaimedCount', 'NotAvailable', 'UniqueFee', 'freeStationCount', 'paidStationCount',
                       'embedding_feature', 'Hour_of_Day','Day_Of_Week', 'Day_Of_year', 'Month_Of_Year', 'Time_of_day_0_4', 'Time_of_day_4_8', 'Time_of_day_8_12',
                       'Time_of_day_12_16', 'Time_of_day_16_20', 'Time_of_day_20_24', 'DayCategory', 'Season', 'daylightSaving', 'Smoothed_chargingTimeTotal',
                       'Smoothed_energyPriceTotal', 'lag_1', 'lag_2', 'Smoothed_kWhDeliveredTotal']
    dataframe = dataframe[desired_columns]
    return dataframe

'''
OWN FEATURES
Method to extract all features for the model
Features to inspect:
Power rolling week mean: Rolling mean considering a lag of 1 week, using a rolling window size of 1 week
Power rolling day mean: Rolling mean considering a lag of 1 day using a rolling window size of 1 day
Power week lag 3 h mean: Rolling means considering a 1-week lag, using a rolling window size of 3 h
Power day lag 3 h mean: Rolling mean considering a lag of 1 day, using a rolling window size of 3 h

Features implemented: 
Temperature: EV's Range is up to 30% less when its cold
Number of same Days: Count the number of same Days in the dataset
'''

def extract_dayCounts(df_subset):
    df_subset['UTCTransactionStart'] = pd.to_datetime(df_subset['UTCTransactionStart'])

    # Zählen, wie oft jeder Tag vorkommt und dem DataFrame hinzufügen
    day_counts = df_subset['UTCTransactionStart'].dt.date.value_counts()
    df_subset['DayCounts'] = df_subset['UTCTransactionStart'].dt.date.map(day_counts)


    return df_subset['DayCounts']


def extract_temperature(df_subset):
    # API-Key is deactivated because of github public sharing
    # This API can be used for actual Weather dates
    # owm = pyowm.OWM('1b574e901a932b453389e5cc2f1aba6d')
    '''
    weather_mgr = owm.weather_manager()
    place = 'Medebach, DE'
    observation = weather_mgr.weather_at_place(place)
    temperature = observation.weather.temperature("celsius")["temp"]
    humidity = observation.weather.humidity
    wind = observation.weather.wind()
    print(f'Temperature: {temperature}°C')
    print(f'Humidity: {humidity}%')
    print(f'Wind Speed: {wind["speed"]} m/s')
    '''

    # Import Meteostat library and dependencies
    # This API is for historical Weather data


    # Set time period
    start = datetime(2017, 1, 1)
    end = datetime(2017, 12, 31)
    # Coordinates for Amsterdam
    latitude = 51.509865
    longitude = -0.118092

    # Create a Point object for Amsterdam
    amsterdam = Point(latitude, longitude, 10)

    # Get daily data for 2019
    data = Daily(amsterdam, start, end)
    data = data.fetch()

    # Plot line chart including average, minimum and maximum temperature
    # data.plot(y=['tavg', 'tmin', 'tmax'])
    # plt.show()

    # Merge weather data with DataFrame based on date
    df_subset['StartDate'] = pd.to_datetime(df_subset['StartDate'])

    df_merged = pd.merge(df_subset, data, left_on='StartDate', right_on='time')
    #df_merged.to_excel('output.xlsx', index=False)

    return df_merged


def features(df_subset):
    print("here are all my features")
    #df_subset['Months'] = extract_months(df_subset)
    #df_subset['Season'] = extract_seasons(df_subset)
    #df_subset['Weekday'] = extract_weekdays(df_subset)
    #df_subset['Weekend'] = extract_weekend(df_subset)
    #df_subset['Holidays'] = extract_holidays(df_subset)
    df_subset['DayCounts'] = extract_dayCounts(df_subset)
    df_subset = extract_temperature(df_subset)
    #df_subset = drop_helper_columns(df_subset)
    return df_subset


def time_to_quarter_hour(time_str):
    # Split the time string into hours and minutes
    hours, minutes, seconds = map(int, time_str.split(':'))

    # Calculate the total minutes since midnight
    total_minutes = hours * 60 + minutes

    # Calculate the number of quarter hours passed
    quarter_hours = total_minutes // 15

    return quarter_hours


def check_previous_charges(df, datetime_column='StartDate'):
    df['datetime'] = pd.to_datetime(df[datetime_column])
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time


    # Sort the DataFrame by datetime
    df = df.sort_values(by='datetime')

    # Initialize columns to store the results of the check
    df['previous_day_charge'] = 0
    df['previous_week_charge'] = 0

    # Iterate over the DataFrame and check for previous charges
    for i in range(1, len(df)):
        current_row = df.iloc[i]
        previous_rows = df.iloc[:i]

        # Check the day before and the same weekday the week before
        previous_day = current_row['date'] - pd.Timedelta(days=1)
        previous_weekday = current_row['date'] - pd.Timedelta(weeks=1)

        same_quarter_hour_day = previous_rows[
            (previous_rows['quarter_hours'] == current_row['quarter_hours']) &
            (previous_rows['date'] == previous_day)
            ]

        same_quarter_hour_week = previous_rows[
            (previous_rows['quarter_hours'] == current_row['quarter_hours']) &
            (previous_rows['date'] == previous_weekday)
            ]

        if not same_quarter_hour_day.empty:
            df.at[i, 'previous_day_charge'] = 1
        if not same_quarter_hour_week.empty:
            df.at[i, 'previous_week_charge'] = 1

    return df

df_subset = read_in()
df_subset = timestamp_format(dataframe=df_subset)
df_subset = timezonecorrection(dataframe=df_subset)
df_subset = create_temporal_features(df=df_subset)
df_subset = expanding_mean_std_weighted_avg(dataframe=df_subset, window_size=10)
df_subset = features(df_subset=df_subset)
df_subset = smoothing_with_best_params(dataframe=df_subset, column_names=['Energy', 'expanding_mean', 'expanding_std', 'weighted_avg'], method='cro', alpha_range=None, beta_range=None)

# comment this in for energy segmentation (e.g. 18-21 o'clock with 1 hour intervall are 3 more rows)
# df_subset = energy_segmentation(df_subset)
# TODO
# df_subset = create_lag_features(df_subset, target=df_subset['Energy'],  thres=0.3)

unique_value_count(df=df_subset, df_name="UK")



df_subset['quarter_hours'] = df_subset['StartTime'].apply(time_to_quarter_hour)
df_subset = check_previous_charges(df=df_subset)



df_subset.to_excel('output1.xlsx', index=False)


# cant use it because the dataset has no DoneCharging Timestamp
# df_subset = charging_time_and_idle_features(df_subset)

