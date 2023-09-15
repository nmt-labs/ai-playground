import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
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
  base = pickle.load(f)
  print(base)
x_prev = base.iloc[:, 0:4].values
x_prev_label = base.iloc[:, 0:4]
y_classe = base.iloc[:, 4].values

label_encoder = LabelEncoder()

x_prev[:,0] = label_encoder.fit_transform(x_prev[:,0])
x_prev[:,1] = label_encoder.fit_transform(x_prev[:,1])
x_prev[:,2] = label_encoder.fit_transform(x_prev[:,2])
x_prev[:,3] = label_encoder.fit_transform(x_prev[:,3])

print(x_prev)

x_treino, x_teste, y_treino, y_teste = train_test_split(x_prev, y_classe, test_size = 0.20, random_state = 23)
  
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