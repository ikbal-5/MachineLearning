#import library
import pandas as pd
import matplotlib.pyplot as plt

#import data
df = pd.read_csv("maas.csv",sep=";")
# plot data
plt.scatter(df.Deneyim,df.Maas)
plt.xlabel("Deneyim")
plt.ylabel("Maas")
plt.show()


from sklearn.linear_model import LinearRegression

#linear Regression  model

linear_reg = LinearRegression()
# Grafikta mavi noktaların belirlenmesi
x = df.Deneyim.values.reshape(-1,1)
y=df.Maas.values.reshape(-1,1)

#Grafikte ki kırmızı çizgi
linear_reg.fit(x,y)
import numpy as np
#Prediction
b0 = linear_reg.predict([[0]])
print("b0: ",b0)

b0_ = linear_reg.intercept_
print("b0_: ",b0)

b1 = linear_reg.coef_
print("b1: ",b1)

#Maaş = b0 + b1*deneyim

maas_yeni = 1663 + 1138*11
print(maas_yeni)

print(linear_reg.predict([[11]]))
      
      
      
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)





y_head = linear_reg.predict(array) 


#Grafikleri Alt Alta Çalıştır Yoksa Hata Verir
plt.scatter(x,y)
plt.plot (array, y_head, color="red")
plt.show()

linear_reg.predict([[100]])


#Modelin performansının r^2 skoru ile değerlendirilmesi
from sklearn.metrics import r2_score
print("r_square score ",r2_score(y,y_head))








