import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

veri = 'csv_file.csv'

data = pd.read_csv(veri)
# print(data.head())
x = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values.reshape(-1, 1)
xtrain, xtest, ytrain, ytest = train_test_split(x,
                                                y,
                                                test_size=0.2,
                                                random_state=42)
# print(xtest)

lr = LinearRegression()
lr.fit(xtrain, ytrain)
yhead = lr.predict(xtest)

plt.scatter(x, y)
plt.plot(xtest, yhead, color="red")
plt.xlabel("Tecrübe")
plt.ylabel("Maaş")
# plt.show()
