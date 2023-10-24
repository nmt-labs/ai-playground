import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn import tree

PATH = '../sample-data/weather_nominal.csv'
SEPARATOR = ','

base = pd.read_csv(PATH, sep=SEPARATOR, encoding='utf-8')

x_prev = base.iloc[:, 0:4].values
x_prev_label = base.iloc[:, 0:4]
y_classe = base.iloc[:, 4].values

label_encoder = LabelEncoder()

x_prev[:,2] = label_encoder.fit_transform(x_prev[:,2])
x_prev[:,3] = label_encoder.fit_transform(x_prev[:,3])

onehotencoder_restaurante = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0,1])], remainder='passthrough')
x_prev = onehotencoder_restaurante.fit_transform(x_prev)

# print(x_prev)

# separe samples for training and testing
x_treino, x_teste, y_treino, y_teste = train_test_split(x_prev, y_classe, test_size = 0.20, random_state = 23)

modelo = DecisionTreeClassifier(criterion='entropy')
# training decision tree
Y = modelo.fit(x_treino, y_treino)

# testinng decision tree
previsoes = modelo.predict(x_teste)

print('accuracy score:')
print(accuracy_score(y_teste,previsoes))

print('confusion matrix:')
print(confusion_matrix(y_teste, previsoes))

cm = ConfusionMatrix(modelo)
cm.fit(x_treino, y_treino)
cm.score(x_teste, y_teste)
cm.show()

print(classification_report(y_teste, previsoes))

# better tree :D
previsores = ['Sunny', 'Overcast', 'Rainy', 'Hot', 'Mild', 'Cool','Humidity', 'Windy']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
tree.plot_tree(modelo, feature_names=previsores, class_names = modelo.classes_.tolist(), filled=True)
plt.show()