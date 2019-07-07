import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Fix if we can't see the full array
# np.set_printoptions(threshold=np.nan)

# Import dataset
dataset = pd.read_csv('Data.csv')

# Independent variables (""" Matrix of features """)
x = dataset.iloc[:, :-1].values

# Dependent variables
y = dataset.iloc[:, -1].values

# Taking care of the missing data:
# We replace missing values with the mean of
# the rest of the values in that column
from sklearn.preprocessing import Imputer
imputer = Imputer(
  missing_values=np.nan, strategy='mean', # Also valid: missing_values='NaN'
  axis=0
)
imputer.fit(x[:, 1:3]) # 1:3 includes indexes 1 and 2
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)