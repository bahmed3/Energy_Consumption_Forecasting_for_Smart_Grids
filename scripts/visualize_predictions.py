import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the predicted data
data = pd.read_csv('outputs/predicted_energy_consumption.csv')

# Ensure the outputs directory exists
os.makedirs('outputs', exist_ok=True)

# Convert 'DATE' to datetime format for better plotting
data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce')

# Plot actual vs. predicted energy consumption
plt.figure(figsize=(14, 7))
plt.plot(data['DATE'], data['Energy_Consumption'], label='Actual Energy Consumption', color='blue')
plt.plot(data['DATE'], data['Predicted_Energy_Consumption'], label='Predicted Energy Consumption', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (MWh)')
plt.title('Actual vs. Predicted Energy Consumption Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit the number of x-axis labels to avoid clutter
plt.tight_layout()

# Save the plot
output_path = 'outputs/actual_vs_predicted_energy_consumption.png'
plt.savefig(output_path)
plt.close()

print(f"Visualization saved to {output_path}")
