import scipy
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
import sklearn
import pandas as pd
import holidays as hd
import pyowm
from datetime import datetime
from meteostat import Point, Daily


def read_in():
    df = pd.read_csv('transactions.csv', delimiter=';')
    df_metervalues = pd.read_csv('meter_values.csv', delimiter=';')

    # Adjust columns
    columns_of_interest = ['TransactionId', 'ChargePoint', 'Connector', 'UTCTransactionStart', 'UTCTransactionStop',
                           'StartCard', 'ConnectedTime', 'ChargeTime', 'TotalEnergy', 'MaxPower']
    columns_metervalues = ['TransactionId', 'ChargePoint', 'Connector', 'UTCTime', 'Collectedvalue',
                           'EnergyInterval', 'AveragePower']

    # Subset DataFrame with specific columns
    df_subset_metervalues = df_metervalues[columns_metervalues]
    df_subset = df[columns_of_interest]

    df_subset['StartDate'] = pd.to_datetime(df['UTCTransactionStart']).dt.date
    df_subset['TotalEnergy'] = df_subset['TotalEnergy'].str.replace(',', '.').astype(float)

    return df_subset




'''
Method to extract all features for the model
Features to inspect:
Power rolling week mean: Rolling mean considering a lag of 1 week, using a rolling window size of 1 week
Power rolling day mean: Rolling mean considering a lag of 1 day using a rolling window size of 1 day
Power week lag 3 h mean: Rolling means considering a 1-week lag, using a rolling window size of 3 h
Power day lag 3 h mean: Rolling mean considering a lag of 1 day, using a rolling window size of 3 h
Number of charging points: The number of charging points for the site
Holiday: Categorical encoded; 1 if Netherlands holiday, 0 if not
Weekday: Categorical encoded weekdays
Temperature: EV's Range is up to 30% less when its cold
'''


def extract_seasons(df_subset):
    season_dict = {'January': 'Winter',
                   'February': 'Winter',
                   'March': 'Spring',
                   'April': 'Spring',
                   'May': 'Spring',
                   'June': 'Summer',
                   'July': 'Summer',
                   'August': 'Summer',
                   'September': 'Fall',
                   'October': 'Fall',
                   'November': 'Fall',
                   'December': 'Winter'}
    return df_subset['Months'].apply(lambda x: season_dict[x])


def extract_months(df_subset):
    # Convert 'StartDate' to datetime format
    df_subset['StartDate'] = pd.to_datetime(df_subset['StartDate'])

    # Extract month as numerical digit
    df_subset['MonthDigit'] = df_subset['StartDate'].dt.month

    # Map month numerical digit to month name
    df_subset['Months'] = df_subset['MonthDigit'].apply(
        lambda x: pd.Timestamp(year=2019, month=x, day=1).strftime('%B'))

    return df_subset['Months']


def extract_weekdays(df_subset):
    # Convert 'StartDate' to datetime format
    df_subset['StartDate'] = pd.to_datetime(df_subset['StartDate'])

    # Extract weekday as numerical digit (0 for Monday, 1 for Tuesday, ..., 6 for Sunday)
    df_subset['Weekday'] = df_subset['StartDate'].dt.weekday

    # Map numerical weekday to categorical encoded weekday
    weekday_dict = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df_subset['Weekday'] = df_subset['Weekday'].map(weekday_dict)

    return df_subset['Weekday']


def extract_weekend(df_subset):
    df_subset['StartDate'] = pd.to_datetime(df_subset['StartDate'])

    # Extract weekday as numerical digit (0 for Monday, 1 for Tuesday, ..., 6 for Sunday)
    df_subset['WeekdayToExtract'] = df_subset['StartDate'].dt.weekday
    df_subset['Weekend'] = df_subset['WeekdayToExtract'].apply(lambda x: 1 if x >= 5 else 0)
    return df_subset['Weekend']
    pass
def drop_helper_columns(df_subset):
    df_subset = df_subset.drop('WeekdayToExtract', axis=1)
    return df_subset


def extract_holidays(df_subset):
    df_subset['StartDate'] = pd.to_datetime(df_subset['StartDate'])

    hd_nl_2019 = hd.Netherlands(years=2019)

    holiday_dates = list(hd_nl_2019.keys())

    # Create a new column 'IsHoliday' and set default value to 0
    df_subset['Holidays'] = 0

    # Set 'IsHoliday' to 1 for holiday dates
    df_subset.loc[df_subset['StartDate'].isin(holiday_dates), 'Holidays'] = 1

    '''
    prints out all holidays from dataset
    holidays_subset = df_subset[df_subset['Holidays'] == 1]['StartDate']
    print("Dates marked as holidays:\n")
    for holiday_date in holidays_subset:
        print(holiday_date)
    '''
    return df_subset['Holidays']


def extract_dayCounts(df_subset):
    df_subset['UTCTransactionStart'] = pd.to_datetime(df_subset['UTCTransactionStart'])

    # Zählen, wie oft jeder Tag vorkommt und dem DataFrame hinzufügen
    day_counts = df_subset['UTCTransactionStart'].dt.date.value_counts()
    df_subset['DayCounts'] = df_subset['UTCTransactionStart'].dt.date.map(day_counts)


    return df_subset['DayCounts']


def extract_temperature(df_subset):
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


    # Set time period
    start = datetime(2019, 1, 1)
    end = datetime(2019, 12, 31)
    # Coordinates for Amsterdam
    latitude = 52.3676
    longitude = 4.9041

    # Create a Point object for Amsterdam
    amsterdam = Point(latitude, longitude, 10)

    # Get daily data for 2019
    data = Daily(amsterdam, start, end)
    data = data.fetch()

    # Plot line chart including average, minimum and maximum temperature
    data.plot(y=['tavg', 'tmin', 'tmax'])
    plt.show()

    # Merge weather data with DataFrame based on date
    df_subset['StartDate'] = pd.to_datetime(df_subset['StartDate'])

    df_merged = pd.merge(df_subset, data, left_on='StartDate', right_on='time')
    print(df_merged.head(10))
    #df_merged.to_excel('output.xlsx', index=False)

    pass


def features(df_subset):
    print("here are all features")
    df_subset['Months'] = extract_months(df_subset)
    df_subset['Season'] = extract_seasons(df_subset)
    df_subset['Weekday'] = extract_weekdays(df_subset)
    df_subset['Weekend'] = extract_weekend(df_subset)
    df_subset['Holidays'] = extract_holidays(df_subset)
    df_subset['DayCounts'] = extract_dayCounts(df_subset)
    df_subset['Temperature'] = extract_temperature(df_subset)

    df_subset = drop_helper_columns(df_subset)
    print(list(df_subset))
    pass




def plot(df_subset):

    daily_energy_mean = df_subset.groupby('StartDate')['TotalEnergy'].mean()

    x_values = np.arange(len(daily_energy_mean.index))
    y_values = daily_energy_mean.values

    coefficients = np.polyfit(x_values, y_values, 2)
    quadratic_regression = np.poly1d(coefficients)

    # Creating x values for the regression line
    x_regression = np.linspace(0, len(daily_energy_mean.index) - 1, 100)
    y_regression = quadratic_regression(x_regression)

    # Plotting the data and the regression line
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', label='Mean Total Energy')
    plt.plot(x_regression, y_regression, color='red', linestyle='--', label='Quadratic Regression')
    plt.title('Mean Total Energy by Start Date')
    plt.xlabel('Start Date Index')
    plt.ylabel('Mean Total Energy')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

df_subset = read_in()

features(df_subset)

plot(df_subset)