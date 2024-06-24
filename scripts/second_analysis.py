import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

merged_data = pd.read_csv('outputs/merged_data.csv', parse_dates=['DATE'])
merged_data = merged_data[merged_data['DATE'].dt.year <= 2023]
merged_data['Month'] = merged_data['DATE'].dt.month
merged_data['Year'] = merged_data['DATE'].dt.year

monthly_avg_consumption = merged_data.groupby(['Year', 'Month'])['Energy_Consumption'].mean().reset_index()
monthly_avg_pivot = monthly_avg_consumption.pivot(index='Year', columns='Month', values='Energy_Consumption')

plt.figure(figsize=(12, 8))
sns.heatmap(monthly_avg_pivot, cmap='coolwarm', annot=True, fmt=".1f", linewidths=.5)
plt.title('Monthly Average Energy Consumption (MWh) Over the Years')
plt.xlabel('Month')
plt.ylabel('Year')

plt.savefig('outputs/monthly_avg_consumption_heatmap.png')

plt.show()

merged_data['Normalized_GDP'] = (merged_data['Real GDP (millions of chained 2017 dollars)'] - merged_data['Real GDP (millions of chained 2017 dollars)'].mean()) / merged_data['Real GDP (millions of chained 2017 dollars)'].std()
merged_data['Normalized_Employment'] = (merged_data['Total Employment (number of jobs)'] - merged_data['Total Employment (number of jobs)'].mean()) / merged_data['Total Employment (number of jobs)'].std()
merged_data['Normalized_Energy'] = (merged_data['Energy_Consumption'] - merged_data['Energy_Consumption'].mean()) / merged_data['Energy_Consumption'].std()

merged_data['Economic_Activity'] = merged_data['Normalized_GDP'] + merged_data['Normalized_Employment']
merged_data['Energy_Employment'] = merged_data['Normalized_Energy'] + merged_data['Normalized_Employment']

plt.figure(figsize=(12, 8))
plt.plot(merged_data['DATE'], merged_data['Economic_Activity'], label='Economic Activity (GDP + Employment)', color='green')
plt.plot(merged_data['DATE'], merged_data['Energy_Employment'], label='Energy Consumption + Employment', color='blue')
plt.title('Combined Metrics Over Time')
plt.xlabel('Date')
plt.ylabel('Combined Metric')
plt.legend()

plt.savefig('outputs/combined_metrics_over_time.png')

plt.show()
