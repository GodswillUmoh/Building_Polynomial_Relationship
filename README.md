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
# Note the more the nth value, the more the curve will be smooth with better results, try 4
This will be: bo + B1X1 + b2X1(power2) + b3X1(power3) + b4X4(power4)
#poly_reg = PolynomialFeatures(degree= 2)

poly_reg = PolynomialFeatures(degree= 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
```

## Visualising the Linear Regression results
```python
# displaying the real values
plt.scatter(X, y, color = 'red')
# plot the linear line
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear plot')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Result the predicted salary, line is far from most points, hence, linearReg is not the best
```
[See Graph Here, Click to view Graph](https://colab.research.google.com/drive/1A4FvrP7RtiEFlPeRX1MAXRm3pj837x_B#scrollTo=KD-jHmzSIWZq)

## What is Overfitting in Polynomial Regression?
> Overfitting happens when a model becomes too complex and starts capturing noise (random variations) in the training data rather than the actual underlying pattern. In polynomial regression, this often occurs when you use a very high-degree polynomial to fit the data.

## How Overfitting Happens
> + High Degree Polynomial: Adding too many powers of 𝑥 (e.g., 𝑥3,𝑥4,𝑥5) allows the model to fit every small variation in the training data.
> + Poor Generalization: While the model fits the training data perfectly, it fails to make accurate predictions on new, unseen data because it has "memorized" the training data rather than learning the overall trend.

## Example
> Imagine you have 10 data points shaped like a smooth curve, but you fit a 9th-degree polynomial to it. The curve will pass exactly through all the points, but it will oscillate wildly in between, making poor predictions for new data.

## How to Prevent Overfitting
> 1. Reduce Polynomial Degree: Use a lower-degree polynomial that captures the overall trend without fitting every fluctuation.

> 2. Cross-Validation: Split your data into training and testing sets (or use k-fold cross-validation) to ensure the model generalizes well.

> 3. Regularization: Apply techniques like Ridge Regression or Lasso Regression to penalize large coefficients, discouraging overfitting.

> 4. Add More Data: Increasing the size of the dataset can help the model learn more robust patterns.

> 5. Feature Selection: Avoid adding irrelevant or redundant features that can complicate the model unnecessarily.

## Visualization Example
> + Underfitting: A straight line doesn’t capture the curve in the data.
> + Optimal Fit: A moderately curved line follows the general trend.
> + Overfitting: A highly wiggly curve passes through every point but fails to generalize.

## Visualising the Polynomial Regression results
```python
# displaying the real values
plt.scatter(X, y, color = 'red')
# plot the linear line
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Linear plot')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```

## Predicting a new result with Linear Regression
```python
# 2D array is used to display prediction to 6.5
# use [[]], the first dimension [] is for rows, second [] is for column
lin_reg.predict([[6.5]])
```

## Predicting a new result with Polynomial Regression
```python
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
```

[See code results, click Here to view](https://colab.research.google.com/drive/1A4FvrP7RtiEFlPeRX1MAXRm3pj837x_B#scrollTo=uQmtnyTHFRGG)

