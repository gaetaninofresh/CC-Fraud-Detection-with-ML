from joblib import load 
import pandas 
import numpy as np
import matplotlib.pyplot as plt
import os

local_path = os.path.dirname(__file__)


results = load(f'{local_path}/KNN_results.joblib')
print(results)

distance = results.loc[results['Weights'] == 'distance'].sort_values('Neighbors')
uniform = results.loc[results['Weights'] == 'uniform'].sort_values('Neighbors')


plt.figure(figsize=(10, 6))
plt.plot(
    distance["Neighbors"],
    distance["F1-Score"],
    marker='o',
    label="Distance"
  )

plt.plot(
    uniform["Neighbors"],
    uniform["F1-Score"],
    marker='x',
    label="Linear"
  )

plt.xlabel("Number of Neighbors (K)")
plt.ylabel("F1-Score")

plt.title("KNN F1-Score Comparison")
plt.legend()
plt.grid()
plt.show()

