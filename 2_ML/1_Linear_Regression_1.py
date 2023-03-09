"""
Created on Meaningful96

Lucent lux tua
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

plt.close("all")


## Step 1) Data Load
dfLoad = pd.read_csv("https://raw.githubusercontent.com/meaningful96/Deep_Learning/main/1_DataSet/1_LinearRegression_dataset.txt"
                     , sep ="\s+")

x = dfLoad["xx"]
y = dfLoad["yy"]

Ndata = len(x)

plt.plot(x,y,'r.')

## Step 2) Ananlytical Way(Ordinary Least Squares, wOLS)
y_np = np.array(y)
X = np.column_stack([np.ones([Ndata,1]), x])
wOLS = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_np)

## Step 3) Prediction 
xPredict = np.linspace(0,2,101)
xPredictPadding = np.column_stack([np.ones([101,1]), xPredict])
yPredict = wOLS.T.dot(xPredictPadding.T)

## Step 4) Numerical way, Gradient Descent
eta = 0.1
niterations = 15
wGD = np.zeros([2,1])

for iteration in range(niterations):
    gradients = -(2/Ndata)*(X.T.dot(y_np.reshape([Ndata,1]) - X.dot(wGD)))
    wGD = wGD - eta*gradients
    yGD = wGD.T.dot(xPredictPadding.T)
    plt.plot(xPredict.reshape(1,101), yGD, 'b.') 
plt.plot(xPredict.reshape(1,101), yGD, 'g.')
    
    
    
