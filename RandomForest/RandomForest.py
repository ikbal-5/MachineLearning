import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("DesicionTree.csv",sep=";",header = None)
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)


#randomforest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)
print("7.8 Seviyesinde fiyatın ne kadar olduğu: ",rf.predict([[7.8]]))


#Modelin performans değerlendirmesi
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)
y_head = rf.predict(x)


from sklearn.metrics import r2_score

print("r_score: ", r2_score(y,y_head))
