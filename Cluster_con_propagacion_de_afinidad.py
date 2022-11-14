import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation

datos = {"alan" : [2, 4],
         "alberto" : [3, 3],
         "alex" : [3, 2], 
         "zelda" : [7, 6], 
         "zoila" : [6, 7], 
         "zulema" : [6, 8]}

personas = set(datos.keys())
datos = pd.DataFrame(datos, index=["ahorro", "evaluacion"])
datos

plt.figure(figsize=(7, 7))
plt.title("Análisis crediticio", fontsize=20)

plt.scatter(datos.T[0:3]["ahorro"], 
            datos.T[0:3]["evaluacion"], 
            marker="x", s=150, color="red",
            linewidths=5, label="Nombres con A")

plt.scatter(datos.T[3:]["ahorro"], 
            datos.T[3:]["evaluacion"], 
            marker="o", s=150, color="blue",
            linewidths=5, label="Nombres con Z")

for i in range(len(datos.columns)):
    plt.text(datos[datos.columns[i]][0]+0.3, 
             datos[datos.columns[i]][1]+0.3, 
             datos.columns[i].capitalize())
    
plt.xlabel("Ahorro", fontsize=15)
plt.ylabel("Evaluación Crediticia", fontsize=15)
plt.legend(bbox_to_anchor=(1.35, 0.15))
plt.box(False)
plt.xlim((0, 10.01))
plt.ylim((0, 10.01))
plt.grid()
plt.show()

print("Similitud entre Alan y Alberto", -((datos["alan"] - datos["alberto"])**2).sum())
print("Similitud entre Alan y Zulema", -((datos["alan"] - datos["zulema"])**2).sum())

print("Similitud entre Alan y Alberto", -((datos["alan"] - datos["alberto"])**2).sum())
print("Similitud entre Alan y Zulema", -((datos["alan"] - datos["zulema"])**2).sum())

np.fill_diagonal(s.values, np.min(s.values))

s

# Matriz de Disponibilidad
d = pd.DataFrame(0, columns=datos.columns, index=datos.columns) 

d

# La responsabilidad se actualiza a partir de la similitud y la disponibilidad
s

# Responsabilidad ["alan", "alberto"]

sim = s.loc["alan", "alberto"]

sim_otros = s.loc["alan", ["alan", "alex", "zelda", "zoila", "zulema"]]

dis_otros = d.loc["alan", ["alan", "alex", "zelda", "zoila", "zulema"]]

print("Qué tan adecuado es Alberto como EJEMPLAR para Alan:", sim - max(sim_otros + dis_otros)) 


sim = s.loc["alan", "zulema"]

sim_otros = s.loc["alan", ["alan", "alberto", "alex", "zelda", "zoila"]]

dis_otros = d.loc["alan", ["alan", "alberto", "alex", "zelda", "zoila"]]

# Responsabilidad ["alan", "zulema"]

print("Qué tan adecuada es Zulema como EJEMPLAR para Alan:", sim - max(sim_otros + dis_otros)) 

# Matriz de Responsabilidad
r = pd.DataFrame(0, columns=datos.columns, index=datos.columns) 


for i in range(10):

    factor = 0.5

    # Actualización de Responsabilidades
    
    r_anterior = r.copy()

    for i in personas:
        for k in personas:
            elegibles = list(personas.difference({k}))
            r.loc[i, k] = s.loc[i, k] - max(s.loc[i, elegibles] + d.loc[i, elegibles])  
    
    r = (1 - factor)*r + factor*r_anterior
    

    # Actualización de Disponibilidades
    
    d_anterior = d.copy()

    for i in personas:
        for k in personas:
            if i == k:
                elegibles = list(personas.difference({i}))
                d.loc[k, k] = r.loc[elegibles, k][r.loc[elegibles, k] > 0].sum()
            else:
                elegibles = list(personas.difference({i, k}))
                d.loc[i, k] = min(0, r.loc[k, k] + r.loc[elegibles, k][r.loc[elegibles, k] > 0].sum())

    d = (1 - factor)*d + factor*d_anterior
    
    # Actualización de Asignaciones 
    
    a = r + d

a.round(decimals=2)

clustering = AffinityPropagation(random_state=None).fit(datos.T.values)
print("Etiquetas de Clusters:", clustering.labels_)
print("Índices de Centroides:", clustering.cluster_centers_indices_)