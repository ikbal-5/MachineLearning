import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/ikbalgencarslan/Spyder/decisiontree/Kitap1.csv",sep=";",header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)


from sklearn.tree import DecisionTreeRegressor
#decisiontree

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

tree_reg.predict([[5.5]])
x_ =np.arange(min(x),max(x),0.01 ).reshape(-1,1)

y_head = tree_reg.predict(x_)

#görselleştirme
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("Tribun Level")
plt.ylabel("Ücret")
plt.show()

#Decision
data =pd.read_csv("/Users/ikbalgencarslan/Spyder/adsız klasör/data.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.diagnosis = [ 1 if each == "M"  else 0 for each in data.diagnosis ]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis =1)
#Normalize
x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state = 42)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
print("score",dt.score(x_test,y_test))
