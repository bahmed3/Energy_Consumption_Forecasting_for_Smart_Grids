import pandas as pd
import joblib

# Load the feature-engineered data
data = pd.read_csv('outputs/feature_engineered_data.csv')

# Define features (exclude the target variable and DATE)
X = data.drop(['Energy_Consumption', 'DATE'], axis=1)
X = X.select_dtypes(include=['float64', 'int64'])  # Ensure only numerical columns are used

# Load the best model (Random Forest in this case)
model = joblib.load('models/random_forest.joblib')

# Make predictions
predictions = model.predict(X)

# Save the predictions
data['Predicted_Energy_Consumption'] = predictions
data.to_csv('outputs/predicted_energy_consumption.csv', index=False)

print("Predictions saved to outputs/predicted_energy_consumption.csv.")
