#############################
# Categorical Variables: "country" and "purchased":
#   they contain categories (as opposed to numerical values)
# We need to encode them into numbers
############################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# See v18missing_data.py
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

# Change the country names for 0, 1, 2
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

# We have to prevent the ML equations from thinking
# that "one country is greater than another"
# because of the numeric values (e.g. 2 > 0)
# We are going to transform the country column
# into 3:
# Original  ----->  France  Germany  Spain
# ---------         ------  -------  -----
# France            1       0        0
# Spain             0       0        1
# Germany           0       1        0
# Spain             0       0        1
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

# For dependent variables, the ML algorithm will know
# that it's a category and that there is no order
# between the values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
