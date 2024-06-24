import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

os.makedirs('models', exist_ok=True)

data = pd.read_csv('outputs/feature_engineered_data.csv')

X = data.drop(['Energy_Consumption', 'DATE'], axis=1)
X = X.select_dtypes(include=['float64', 'int64'])  
y = data['Energy_Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} Performance:')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print('-----------------------------------')
    model_path = f'models/{name.replace(" ", "_").lower()}.joblib'
    print(f'Saving model to {model_path}...')
    joblib.dump(model, model_path)
    print(f'{name} saved successfully.')

print("Model training completed and models saved.")
