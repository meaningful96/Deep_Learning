"""
Created on Meaningful96

Lucent lux tua
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# Step 1) Prepare the data
np.random.seed(0)

X = np.random.rand(100,1)
y = 2*X + 1 + np.random.rand(100,1)


# Step 2) Create a LinearRegression model and fit the data
model = LinearRegression()
model.fit(X,y)

# Step 3) Make predictions on new data
# Generate some new data to make predictions on

X_new = np.array([[0],[1]])
y_new = model.predict(X_new)

# Step 4) Visualization

import matplotlib.pyplot as plt
plt.close('all')

plt.scatter(X,y, color = 'blue')
plt.plot(X_new, y_new, color = 'red')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

#%% without sklearn library
## Ordinary least squares

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Step 1) Prepare the Data
# Generate some random data

X = np.random.rand(100,1)
y = 2*X + 1 + np.random.rand(100,1)

# Step 2) Implement linear regression using Ordinary Least Squares
# Add a column of ones to X to account for the intercept term

X_b = np.column_stack([np.ones([100,1]), X])

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Step 3) Make predictions on new data
# Generate some new data to make predictions on

X_new = np.array([[0], [1]])
# Add a column of ones to X_new to account for the intercept term
X_new_b = np.c_[np.ones((2, 1)), X_new]
# Use the optimal parameters to make predictions for X_new
y_new = X_new_b.dot(theta_best)

# Step 4: Visualize the results
plt.scatter(X, y, color='blue')
plt.plot(X_new, y_new, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
