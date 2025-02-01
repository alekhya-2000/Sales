import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset (assuming a CSV file named 'retail_sales.csv')
data = pd.read_csv('retail_sales.csv')

# Display basic info
data.info()

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Convert date column to datetime format (if applicable)
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

# Summary statistics
print(data.describe())

# Visualize sales trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x=data.index, y='sales', label='Sales Trend')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Retail Sales Trend Over Time')
plt.legend()
plt.show()

# Feature selection
features = ['marketing_spend', 'holiday_flag', 'seasonality_index']  # Modify based on dataset
X = data[features]
y = data['sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}')

# Save the model (optional)
import joblib
joblib.dump(model, 'sales_forecast_model.pkl')
