# ML Project
"""
 Written by Youmin Ko 
 
 Live life as if there were no second chance
"""

########################## Linear Regression ##################################


# Step 1) Data Loading
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

plt.close("all")

dfLoad = pd.read_csv("https://raw.githubusercontent.com/meaningful96/DLproject/main/DataSet/1_LinearRegression_dataset.txt"
                     , sep ="\s+")

xxRaw = dfLoad["xx"]
yyRaw = dfLoad["yy"]

xxRawNP = np.array(xxRaw)
yyRawNP = np.array(yyRaw)
Ndata = len(xxRaw)
plt.plot(xxRaw, yyRaw, 'ro')

# Step2) Analytical Way, wOLS
X = np.column_stack([np.ones([Ndata,1 ]), xxRaw])
wOLS = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(yyRawNP.reshape([Ndata,1]))

# Step3) Prediction variable 
xPredict = np.linspace(0,2,101)
xPredictPadding = np.column_stack([np.ones([101,1]),xPredict])
yPredict = wOLS.T.dot(xPredictPadding.T)

# Step4) Numerical way, Gradient Descent
eta = 0.1
niteration = 10
wGD = np.zeros([2,1])

for iteration in range(niteration):
    gradients = -(2/Ndata)*(X.T.dot(yyRawNP.reshape([Ndata,1]) - X.dot(wGD)))
    wGD = wGD - eta*gradients
    yGD = wGD.T.dot(xPredictPadding.T)
    plt.plot(xPredict.reshape(1,101), yGD, 'b.' )
plt.plot(xPredict.reshape(1,101), yGD, 'g*' )
    
