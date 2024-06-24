import pandas as pd
import os

print("Current Working Directory:", os.getcwd())

print("Files in data directory:", os.listdir('data'))

energy_data = pd.read_csv('data/Retail_sales_of_electricity_monthly.csv')
weather_data = pd.read_csv('data/weather_data.csv')
gdp_data = pd.read_csv('data/california_GDP.csv')
employment_data = pd.read_csv('data/california_Employment.csv')

print("Energy Consumption Data Columns:\n", energy_data.columns)
print("Weather Data Columns:\n", weather_data.columns)
print("GDP Data Columns:\n", gdp_data.columns)
print("Employment Data Columns:\n", employment_data.columns)
