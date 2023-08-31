import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# OPEN CSV -------------------------------------------------------
print('\nOPEN CSV')
PATH = '../sample-data/restaurantev2.csv'
SEPARATOR = ';'

base = pd.read_csv(PATH, sep=SEPARATOR, encoding='utf-8')
print('base:')
print(base)

# COUNTING INTANCES -----------------------------------------------
print('\nCOUNTING INSTANCES')
print('np.unique:')
print(np.unique(base['Tipo'], return_counts=True))
# (array(['Francês', 'Hamburger', 'Italiano', 'Tailandês'], dtype=object), array([2, 4, 2, 4], dtype=int64))

sns.countplot(x = base['Preço'])
print('sns.countplot:')
print('open pop-up')
# print(sns.countplot(x = base['Preço']))
# Axes(0.125,0.11;0.775x0.77)
# plt.show() # comented to not open pop-up every time
# open pop-up with column graph

# SEPARATE ATTRIBUTES -------------------------------------------
print('\nSEPARETE ATTRIBUTES')
x_prev = base.iloc[:, 1:11].values
print('x_prev:')
print(x_prev)
# show as array of arrays
# base.iloc[<lines>,<columns>]

x_prev_label = base.iloc[:, 1:11]
print('\nx_prev_label:')
print(x_prev_label)
# show as dataset

y_classe = base.iloc[:, 11].values
print('\ny_classe: ')
print(y_classe)

# TREATING DATA ------------------------------------------------
print('\nTREATING DATA')
label_encoder = LabelEncoder()
# change data with 1, 2, 3...
print('x_prev[:,0]: ')
print(x_prev[:,0])

x_prev[:,0] = label_encoder.fit_transform(x_prev[:,0])
x_prev[:,1] = label_encoder.fit_transform(x_prev[:,1])
x_prev[:,2] = label_encoder.fit_transform(x_prev[:,2])
x_prev[:,3] = label_encoder.fit_transform(x_prev[:,3])
x_prev[:,4] = label_encoder.fit_transform(x_prev[:,4])
x_prev[:,5] = label_encoder.fit_transform(x_prev[:,5])
x_prev[:,6] = label_encoder.fit_transform(x_prev[:,6])
x_prev[:,7] = label_encoder.fit_transform(x_prev[:,7])
x_prev[:,9] = label_encoder.fit_transform(x_prev[:,9])
print('\nx_prev:')
print(x_prev)

# change non ordinals attributes to binary
onehotencoder_restaurante = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [8])], remainder='passthrough')
x_prev= onehotencoder_restaurante.fit_transform(x_prev)
# change Restaurante column into new columns for each atribute
# and set the attributes as 0 or 1
print('\nx_prev:')
print(x_prev)
