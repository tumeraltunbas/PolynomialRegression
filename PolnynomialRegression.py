#Polynomical Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\hp\\Desktop\\Regression Tekrar\\Polynomical Regression\\Position_Salaries.csv')

y = df.iloc[:,2:].values
x=df.iloc[:,1:2].values

#To the understand between linear regression and polynomic regression, we gonna first use linear regression.
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)
plt.scatter(x,y)
plt.plot(x,lr.predict(x))
plt.show()

#Our LR model is not successful on non-linear dataset.

#Let's build a polynomic model.
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=4)
x_poly = pf.fit_transform(x)
lr2= LinearRegression()
lr2.fit(x_poly,y)
plt.scatter(x,y)
plt.plot(x,lr2.predict(x_poly))
plt.show()