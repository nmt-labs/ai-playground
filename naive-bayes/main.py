import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

PATH = '../sample-data/weather_nominal.csv'
PATH_PKL = '../sample-data/weather_nominal.pkl'
SEPARATOR = ','

# convert csv to pkl
df = pd.read_csv(PATH, sep=SEPARATOR, encoding='utf-8')

with open(PATH_PKL, 'wb') as file:
  pickle.dump(df, file)

with open(PATH_PKL, 'rb') as f:
  f.seek(0)
  x_treino, x_teste, y_treino, y_teste = pickle.load(f)
  
modelo = GaussianNB()
modelo.fit(x_treino, y_treino)

previsoes = modelo.predict(x_teste)

print(accuracy_score(y_teste,previsoes))
print(confusion_matrix(y_teste,previsoes))

cm = ConfusionMatrix(modelo)
cm.fit(x_treino, y_treino)
cm.score(x_teste, y_teste)
cm.show()

print(classification_report(y_teste, previsoes))