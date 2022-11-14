#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
datos= pd.read_csv("ingreso.csv")
datos


# In[ ]:





# In[8]:


import matplotlib.pyplot as plt
plt.ylabel("Ingreso($)")
plt.xlabel("promedio de horas semanales trabajadas")
plt.scatter(datos["horas"],datos["ingreso"],color="pink")
plt.show()


# In[14]:


#Creacion de modelo de regresion lineal simple
from sklearn import linear_model 
regresion = linear_model.LinearRegression()
horas=datos["horas"].values.reshape((-1,1))
modelo= regresion.fit(horas,datos["ingreso"])
print("Interseccion (b)", modelo.intercept_)
print("Pendiente (m)", modelo.coef_)
entrada =[[39],[40],[43],[44]]
modelo.predict(entrada)

plt.scatter(entrada, modelo.predict(entrada), color= "red")
plt.plot(entrada, modelo.predict(entrada), color="black")

plt.ylabel("Ingreso($)")
plt.xlabel("promedio de horas semanales trabajadas")
plt.scatter(datos["horas"],datos["ingreso"],color="pink")
plt.show()


# In[ ]:




