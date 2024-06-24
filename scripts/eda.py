import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

energy_data = pd.read_csv('data/Retail_sales_of_electricity_monthly.csv')
weather_data = pd.read_csv('data/weather_data.csv')
gdp_data = pd.read_csv('data/california_GDP.csv')
employment_data = pd.read_csv('data/california_Employment.csv')

required_periods = min(len(weather_data), 11 * 12)  # 11 years of monthly data
weather_data = weather_data.head(required_periods)
weather_data['DATE'] = pd.date_range(start='2013-01-01', periods=required_periods, freq='MS')  # Monthly start frequency

energy_data['DATE'] = pd.to_datetime(energy_data['DATE'], errors='coerce')
gdp_data['Year'] = pd.to_datetime(gdp_data['Year'], format='%Y', errors='coerce')
employment_data['Year'] = pd.to_datetime(employment_data['Year'], format='%Y', errors='coerce')

employment_data['Employment_Year'] = employment_data['Year'].dt.year

print("Weather Data:")
print(weather_data.head())

print("Employment Data:")
print(employment_data.head())

merged_data = pd.merge(energy_data, weather_data, on='DATE', how='left')

merged_data['Year'] = merged_data['DATE'].dt.year.astype(int)

gdp_data.rename(columns={'Year': 'GDP_Year'}, inplace=True)
gdp_data['GDP_Year'] = gdp_data['GDP_Year'].dt.year
merged_data = pd.merge(merged_data, gdp_data, left_on='Year', right_on='GDP_Year', how='left')

merged_data = pd.merge(merged_data, employment_data, left_on='Year', right_on='Employment_Year', how='left')

merged_data.drop(columns=['GDP_Year', 'Employment_Year'], inplace=True)

merged_data.fillna(0, inplace=True)

print("Merged Data:")
print(merged_data.head())

plt.figure(figsize=(10, 6))
sns.lineplot(data=gdp_data, x='GDP_Year', y='Real GDP (millions of chained 2017 dollars)')
plt.title('Real GDP Over Time (Original Data)')
plt.xlabel('Year')
plt.ylabel('Real GDP (millions of chained 2017 dollars)')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=weather_data, x='DATE', y='TAVG')
plt.title('Average Temperature Over Time (Original Data)')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=employment_data, x='Employment_Year', y='Total Employment (number of jobs)')
plt.title('Total Employment Over Time (Original Data)')
plt.xlabel('Year')
plt.ylabel('Total Employment')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=merged_data, x='DATE', y='Energy_Consumption')
plt.title('Energy Consumption Over Time')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (MWh)')
plt.show()

merged_data.to_csv('outputs/merged_data.csv', index=False)
