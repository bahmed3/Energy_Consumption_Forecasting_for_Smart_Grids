import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_csv('outputs/predicted_energy_consumption.csv')

os.makedirs('outputs', exist_ok=True)

data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce')

plt.figure(figsize=(14, 7))
plt.plot(data['DATE'], data['Energy_Consumption'], label='Actual Energy Consumption', color='blue')
plt.plot(data['DATE'], data['Predicted_Energy_Consumption'], label='Predicted Energy Consumption', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (MWh)')
plt.title('Actual vs. Predicted Energy Consumption Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  
plt.tight_layout()

output_path = 'outputs/actual_vs_predicted_energy_consumption.png'
plt.savefig(output_path)
plt.close()

print(f"Visualization saved to {output_path}")
