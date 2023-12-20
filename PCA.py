#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:25:02 2023

@author: ikbalgencarslan
"""
# Sklearn kütüphanesinden veri seti çekmek
from sklearn.datasets import load_iris
import pandas as pd

#%%

iris = load_iris()
data =iris.data
feature_names = iris.feature_names
y = iris.target


df = pd.DataFrame(data, columns = feature_names)
df["Sınıf"] = y
x = data
from sklearn.decomposition import PCA

pca = PCA(n_components=2,whiten=True) # whiten normalize etmek

pca.fit(x)
x_pca = pca.transform(x)

print("variance ratio: ",pca.explained_variance_ratio_)
print("sum:",sum(pca.explained_variance_ratio_))

#%% 2D

df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red","green","blue"]
import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.Sınıf == each], df.p2[df.Sınıf == each],color = color[each],label=iris.target_names[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
