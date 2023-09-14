import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.naive_bayes import GaussianNB
import pickle

PATH = '../sample-data/weather_nominal.csv'
SEPARATOR = ','

# convert csv to pkl
df = pd.read_csv(PATH, sep=SEPARATOR, encoding='utf-8')
df.to_pickle('../sample-data/weather_nominal.pkl')

with open('../sample-data/weather_nominal.pkl', 'rb') as f:
  f.seek(0)
  x_treino, x_teste, y_treino, y_teste = pickle.load(f)

modelo = GaussianNB()
modelo.fit(x_treino, y_treino)

previsoes = modelo.predict(x_teste)
print(previsoes)