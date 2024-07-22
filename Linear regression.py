import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = {
    'square_footage': [1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
    'bedrooms': [3, 4, 4, 5, 5, 6, 6, 7],
    'bathrooms': [2, 3, 3, 4, 4, 5, 5, 6],
    'price': [300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
}

df = pd.DataFrame(data)


X = df[['square_footage', 'bedrooms', 'bathrooms']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


new_house = np.array([[3500, 4, 3]])  
predicted_price = model.predict(new_house)

print(f"Predicted price for the house: ${predicted_price[0]:.2f}")
