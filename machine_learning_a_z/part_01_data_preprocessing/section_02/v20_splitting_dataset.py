import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# See v19categorical_data.py
dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(
  missing_values=np.nan, strategy='mean',
  axis=0
)
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

