import pandas as pd
import joblib

data = pd.read_csv('outputs/feature_engineered_data.csv')

X = data.drop(['Energy_Consumption', 'DATE'], axis=1)
X = X.select_dtypes(include=['float64', 'int64']) 

model = joblib.load('models/random_forest.joblib')

predictions = model.predict(X)

data['Predicted_Energy_Consumption'] = predictions
data.to_csv('outputs/predicted_energy_consumption.csv', index=False)

print("Predictions saved to outputs/predicted_energy_consumption.csv.")
