import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

energy_data = pd.read_csv('data/Retail_sales_of_electricity_monthly.csv', parse_dates=['DATE'])

energy_data.sort_values('DATE', inplace=True)

data = energy_data[['Energy_Consumption']]

model = IsolationForest(contamination=0.05) 
energy_data['Anomaly'] = model.fit_predict(data)

anomalies = energy_data[energy_data['Anomaly'] == -1]

plt.figure(figsize=(12, 6))
plt.plot(energy_data['DATE'], energy_data['Energy_Consumption'], label='Energy Consumption', color='blue')
plt.scatter(anomalies['DATE'], anomalies['Energy_Consumption'], color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (MWh)')
plt.title('Anomaly Detection in Energy Consumption')
plt.legend()

plt.savefig('outputs/anomaly_detection.png')
plt.show()

anomalies.to_csv('outputs/anomalies.csv', index=False)