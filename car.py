import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

file_path = 'car data.csv'
car_data = pd.read_csv(file_path)

car_data = car_data.drop('Car_Name', axis=1)

car_data['Car_Age'] = 2024 - car_data['Year']
car_data = car_data.drop('Year', axis=1)

categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
encoder = OneHotEncoder(sparse_output=False, drop='first') 
encoded_features = encoder.fit_transform(car_data[categorical_features])

encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

car_data = pd.concat([car_data, encoded_features_df], axis=1)
car_data = car_data.drop(categorical_features, axis=1)

X = car_data.drop('Selling_Price', axis=1)
y = car_data['Selling_Price']


scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)  # Using k=5 by default
knn_model.fit(X_train, y_train)

# Making predictions
y_pred = knn_model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Printing evaluation metrics
print(f"Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"KNN accuracy: {r2:.2f}")
