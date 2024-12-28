# Building_Polynomial_Linear_Regression
## Case Study
>Below is a dataset to predict the salary of an individual based on his job position. You as a data scientist is saddled with the role of predicting how much the person to be hired earn in his previous company based on his position

## Dataset
The position is encoded to numbers in the level column, hence we will not be applying OneHotEncoder method
|Position	|Level	|Salary|
|---------|-------|-------|
|Business Analyst|	1|	45000|
|Junior Consultant|	2	|50000|
|Senior Consultant|	3|	60000|
|Manager|	4	|80000|
|Country Manager	|5	|110000|
|Region Manager|	6	|150000|
|Partner|	7	|200000|
|Senior Partner|	8	|300000|
|C-level|	9	|500000|
|CEO|	10	|1000000|
---
## Python codes for Polynomial Linear Regression
---

## Importing the libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Importing the dataset
```python
dataset = pd.read_csv('Position_Salaries.csv')
#exclude the first column
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
```

## Training the Linear Regression model on the whole dataset
```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
```

## Training the Polynomial Regression model on the whole dataset
The first will be to create the metrics of power features, then create another to fit into the model

[View Polynomial Regression Model Formula here, click to view](https://ibb.co/Xxg5TWs)

```python
# it is kind of a preprocessing phase to create the square metrics, 
# hence we import from sklearn.preprocessing
# the final model will be a combination of the linear model and the metric features
# degree is the parameter for the n in X(power n) in the formula
# fit_transform transform the x1, and x2 square as second features, if degree is 3,
# it will be x1, x1 square, x1 power 3, the fit_transform when executed will give only the
# metric features without the x1, hence we need to combine to have full

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
```

## Visualising the Linear Regression results
```python
print(x)
```

## Visualising the Polynomial Regression results
```python
print(x)
```

## Visualising the Polynomial Regression results (for higher resolution and smoother curve)
```python
print(x)
```

## Predicting a new result with Linear Regression
```python
print(x)
```

## Predicting a new result with Polynomial Regression
```python
print(x)
```


