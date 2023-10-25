import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Maas2.csv",sep=";")

x = df.iloc[:,[0,2]].values
y=df.Maas.values.reshape(-1,1)

multiple_linear_reg = LinearRegression()
multiple_linear_reg.fit(x, y)


print("b0: ", multiple_linear_reg.intercept_)
print("b1,b2: ",multiple_linear_reg.coef_)



multiple_linear_reg.predict(np.array([[10,35],[5,35]]))