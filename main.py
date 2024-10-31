import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import linear_model
csv_data = pd.read_csv("FuelConsumptionCo2.csv")
readable_data = csv_data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(readable_data.head(9))
plt.scatter(readable_data.ENGINESIZE,readable_data.CO2EMISSIONS,color = 'blue')
# readable_data.ENGINESIZE selects the engine size column from the dara
# plt.scatter(x-axix,y-axsis,chart plot color) the parameter passed for x-axis is x-axis value and the parameter passed for y-axis
# is y-axis value it creates a scatter plot whih shows the relationship between engine size and co2Emissions
plt.xlabel("Engine Size")
# labels the x-axis
plt.ylabel("Co2 Emissions")
# labels the y-axis
plt.savefig("./pngFiles/engine-vs-co2.png")
# Saves a png file of the scatter plot 
split = np.random.rand(len(csv_data))< 0.8
# np.random.rand(n) generates an array of n random numbers between 0 and 1 in this case it generates 1 random float for each row in
# csv_data 
# len returns the number of rows in csv_data 
# <0.8 returns true for every number in the np.random.rand() array less then 0.8 and false for greater than 0.8 creating a 
# 20/80 split in the form of a random boolean array 
train = csv_data[split]
# all the values of boolean array split that are true are saved in the train array
test  = csv_data[~split]
# all the values of boolean array split that are false are saved in the test array
model = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# np.asanyarray() converts the data into a numpy array but preserves subclass or special structure for example it will keep a matrix 
# a matrix where as np.asarray() will convert the data into a regular numpy array here we should use np.asarray() as the data is simple
# and does not contain a special sub class
train_y = np.asarray(train[['CO2EMISSIONS']])
model.fit(train_x,train_y)
# fits data to the model and trains the model to learn from the provided data so it can make predictions on new and unseen data
print("Coeffecients : ",model.coef_)