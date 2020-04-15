import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# read csv file into a DataFrame
dataset= pd.read_csv('../data/house_price/train.csv')
print(dataset.head())
print(dataset.describe())
print(dataset.corr())
size = dataset["LotArea"]
price = dataset["SalePrice"]

# machine learning handle arrays not DataFrames
x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)
print(x)

# we use linear regression + fit() is training
model = LinearRegression()
model.fit(x,y)

# MSE value and R value
regression_mse_result = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_mse_result))
print("R squared value : ", model.score(x, y))

# we can get b value after fit

# b0 value is
print("B0 value is :",model.coef_[0])
# b1 value is
print("B1 value is : ",model.intercept_[0])

# visualize the dataset with fitted model

plt.scatter(x, y, color="green")
plt.plot(x, model.predict(x), color="black")
plt.title("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

#predicting the prices

print("Prediction by the model : ", model.predict([[2000]]))


