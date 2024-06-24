import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load merged data
merged_data = pd.read_csv('outputs/merged_data.csv')

# Feature Engineering
# Create new features from the existing data
merged_data['Month'] = pd.to_datetime(merged_data['DATE']).dt.month
merged_data['Year'] = pd.to_datetime(merged_data['DATE']).dt.year
merged_data['DayOfWeek'] = pd.to_datetime(merged_data['DATE']).dt.dayofweek

# Lag features
for lag in range(1, 13):
    merged_data[f'Energy_Consumption_Lag_{lag}'] = merged_data['Energy_Consumption'].shift(lag)

# Rolling mean features
merged_data['Energy_Consumption_Rolling_Mean'] = merged_data['Energy_Consumption'].rolling(window=3).mean()

# Drop NA values created by lag features
merged_data = merged_data.dropna()

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Energy_Consumption', 'TAVG', 'Real GDP (millions of chained 2017 dollars)', 'Total Employment (number of jobs)']
merged_data[numerical_features] = scaler.fit_transform(merged_data[numerical_features])

# Save the feature-engineered data
merged_data.to_csv('outputs/feature_engineered_data.csv', index=False)

print("Feature engineering completed and data saved.")
