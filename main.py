import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
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

