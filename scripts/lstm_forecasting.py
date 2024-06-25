import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

energy_data = pd.read_csv('data/Retail_sales_of_electricity_monthly.csv', parse_dates=['DATE'])
energy_data.sort_values('DATE', inplace=True)
data = energy_data['Energy_Consumption'].values.astype(float)

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X_train, y_train = create_dataset(train, time_step)
X_test, y_test = create_dataset(test, time_step)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=1, epochs=50)

future_steps = 24
last_train_data = data[-time_step:]
future_predictions = []

for _ in range(future_steps):
    prediction = model.predict(last_train_data.reshape((1, time_step, 1)))
    future_predictions.append(prediction[0, 0])
    last_train_data = np.append(last_train_data[1:], prediction)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

future_dates = pd.date_range(start=energy_data['DATE'].iloc[-1] + pd.DateOffset(months=1), periods=future_steps, freq='M')
future_df = pd.DataFrame({'DATE': future_dates, 'Energy_Consumption': future_predictions.flatten()})

combined_df = pd.concat([energy_data, future_df])

plt.figure(figsize=(12, 6))
plt.plot(combined_df['DATE'], combined_df['Energy_Consumption'], label='Actual Data and Predictions', color='blue')
plt.axvline(x=energy_data['DATE'].iloc[-1], color='red', linestyle='--', label='Prediction Start')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (MWh)')
plt.title('Energy Consumption Forecasting using LSTM')
plt.legend()

plt.savefig('outputs/energy_consumption_forecast.png')
plt.show()

future_df.to_csv('outputs/future_predictions.csv', index=False)
