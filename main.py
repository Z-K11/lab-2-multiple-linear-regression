import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
csv_data = pd.read_csv("FuelConsumptionCo2.csv")
readable_data = csv_data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(readable_data.head(9))
