import scipy
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
import sklearn
import pandas as pd

# Read the CSV file into a DataFrame with the correct delimiter
df = pd.read_csv('transactions.csv', delimiter=';')
df_metervalues = pd.read_csv('meter_values.csv', delimiter=';')


# Adjust column names if needed
columns_of_interest = ['TransactionId', 'ChargePoint', 'Connector', 'UTCTransactionStart', 'UTCTransactionStop', 'StartCard', 'ConnectedTime', 'ChargeTime', 'TotalEnergy', 'MaxPower']
columns_metervalues = ['TransactionId',	'ChargePoint',	'Connector',	'UTCTime',	'Collectedvalue',
                       'EnergyInterval',	'AveragePower']

# Create a subset DataFrame with specific columns
df_subset_metervalues = df_metervalues[columns_metervalues]
df_subset = df[columns_of_interest]


df_subset['StartDate'] = pd.to_datetime(df['UTCTransactionStart']).dt.date
df_subset['TotalEnergy'] = df_subset['TotalEnergy'].str.replace(',', '.').astype(float)
print(df_subset['TotalEnergy'].head(10))
#df_subset['TotalEnergy'] = pd.to_numeric(df_subset['TotalEnergy'], errors='raise')
daily_energy = df_subset.groupby('StartDate')
daily_energy_mean = df_subset.groupby('StartDate')['TotalEnergy'].mean()
# Print the first few rows to verify

print(df_subset.head(10))

print(daily_energy_mean.head(30))



plt.figure(figsize=(10, 6))
plt.plot(daily_energy_mean.index, daily_energy_mean.values, marker='o', linestyle='-')
plt.title('Mean Total Energy by Start Date')
plt.xlabel('Start Date')
plt.ylabel('Mean Total Energy')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df_subset, model='additive')
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)