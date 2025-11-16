import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("G:\\tasks\\House Price Prediction\\data.csv")
numeric_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
X = data[numeric_columns]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(mse)
new_sample = pd.DataFrame([[3, 2, 1500, 5000, 2, 0, 1, 3, 1000, 500, 1990, 0]], columns=X.columns)
predicted_price = model.predict(new_sample)
print(predicted_price[0])

#The first line in the output is the MSE
#The second line is the predicted price for the house