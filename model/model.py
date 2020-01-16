from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from math import sqrt
import numpy as np
import pandas as pd
import pickle

# Read abalone dataset
df = pd.read_csv('data/abalone.csv')

# undo scaling described in https://archive.ics.uci.edu/ml/datasets/abalone
df[df.select_dtypes(include=['float64']).columns] *= 200

df.describe() # use the mean values as defaults in index.html
df['Sex'].value_counts()

target = 'Rings'
y = df[target]
X = df.drop(target, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=668)

# Benchmark model
model = DummyRegressor()
model.fit(X_train, y_train)

model.score(X_train, y_train)
sqrt(mean_squared_error(y_train, model.predict(X_train))) # RMSE is 3.2

# Data Frame Mapper
mapper = DataFrameMapper([
    ('Sex', [CategoricalImputer(), LabelBinarizer()]),
    (['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
            'Viscera weight', 'Shell weight'],
        [SimpleImputer(), StandardScaler()])
])

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

# Linear Regression
model = LinearRegression()
model.fit(Z_train, y_train)

model.score(Z_train, y_train)
model.score(Z_test, y_test)
sqrt(mean_squared_error(y_train, model.predict(Z_train))) # RMSE is 2.2

# Lasso Regression
alphas = np.linspace(0.0001, 0.0050, 100)
model = LassoCV(alphas=alphas, cv=5)
model = model.fit(Z_train, y_train)
model.alpha_

model.score(Z_train, y_train)
model.score(Z_test, y_test)
sqrt(mean_squared_error(y_train, model.predict(Z_train))) # RMSE is 2.2

# Pipeline
pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open('model/pipe.pkl', 'wb')) # https://mollusc.herokuapp.com/
