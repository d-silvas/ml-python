import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###############################
# v20
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

# Split the dataset into the Trainig set and the #test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, # we have 20% of observations in test set
    random_state=0
)
###############################

# We need to scale because we use euclidean distances in order
# to calculate how well the model does. If we don't scale,
# the "salary" variable will dominate the age

# Feature scaling. Ther are two ways of doing it: standarisation
# and normalisation. StandardScaler uses standarisation
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Do we have to scale dummy variables (countries)? 
# ^ google it: there are several approaches

# Do we need to scale y?
# No, because the dependent variable is categorical

print(x_train)
print()
print(x_test)
