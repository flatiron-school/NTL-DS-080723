# Linear Regression Checkpoint

This checkpoint is designed to test your understanding of linear regression.

Specifically, this will cover:

* Creating simple and multiple linear regression models with StatsModels
* Interpreting linear regression model metrics
* Interpreting linear regression model parameters

## Your Task: Build Linear Regression Models to Predict Home Prices

### Data Understanding

You will be using the Ames Housing dataset, modeling the `SalePrice` using these numeric features:

* `GrLivArea`: Above grade living area (square feet)
* `GarageArea`: Size of garage (square feet)
* `LotArea`: Lot size (square feet)
* `LotFrontage`: Length of street connected to property (feet)


```python
# Run this cell without changes

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("ames.csv", index_col=0)
df = df[["SalePrice", "GrLivArea", "GarageArea", "LotArea", "LotFrontage"]].copy()
df.dropna(inplace=True)
df
```

### Modeling

You will apply an inferential modeling process using StatsModels. This means that you are trying to create the best model in terms of variance in `SalePrice` that is explained (i.e. R-Squared).

You will build **two models — one simple linear regression model and one multiple linear regresssion model** — then you will interpret the model summaries.

There are two relevant components of interpreting the model summaries: model **metrics** such as r-squared and p-values, which tell you how well your model is fit to the data, and model **parameters** (intercept and coefficients), which tell you how the model is using the feature(s) to predict the target.

### Requirements

## 1. Build a Simple Linear Regression Using StatsModels

Below, we use the `.corr()` method to find which features are most correlated with `SalePrice`:


```python
# Run this cell without changes
df.corr()["SalePrice"]
```

The `GrLivArea` feature has the highest correlation with `SalePrice`, so we will use it to build a simple linear regression model.

Use the OLS model ([documentation here](https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLS.html)) with:

- `SalePrice` as the endogenous (dependent) variable
- `GrLivArea` as the exogenous (independent) variable

Don't forget to include `sm.add_constant` to ensure that there is an intercept term.

Fill in the appropriate values in the cell below.


```python
import statsmodels.api as sm

# Replace None with appropriate code
simple_model = None
# your code here
raise NotImplementedError
simple_model_results = simple_model.fit()
print(simple_model_results.summary())
```


```python
# simple_model should be an OLS model
assert type(simple_model) == sm.OLS

# simple_model should have 1 feature (other than the constant)
assert simple_model.df_model == 1

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

## 2. Interpret Simple Linear Regression Model Metrics

We want to know:

1. How much of the variance is explained by this model? This is also known as the R-Squared. Fill in `r_squared` with this value — a floating point number between 0 and 1.
2. Is the model statistically significant at $\alpha = 0.05$? This is determined by comparing the probability of the f-statistic to the alpha. Fill in `model_is_significant` with this value — either `True` or `False`.

You can either just look at the print-out above and fill in the values, or you can use attributes of `simple_model_results` ([documentation here](https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.RegressionResults.html)). If you are getting stuck, it's usually easier to type the answer in rather than writing code to do it.


```python
# Replace None with appropriate code
r_squared = None
model_is_significant = None
# your code here
raise NotImplementedError
```


```python
import numpy as np

# r_squared should be a floating point value between 0 and 1
assert 0 <= r_squared and r_squared <= 1
assert type(r_squared) == float or type(r_squared) == np.float64

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```


```python
# model_is_significant should be True or False
assert model_is_significant == True or model_is_significant == False

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

## 3. Interpret Simple Linear Regression Parameters

Now, we want to know what relationship the model has found between the feature and the target. Because this is a simple linear regression, it follows the format of $y = mx + b$ where $y$ is the `SalePrice`, $m$ is the slope of `GrLivArea`, $x$ is `GrLivArea`, and $b$ is the y-intercept (the value of $y$ when $x$ is 0).

In the cell below, fill in appropriate values for `m` and `b`. Again, you can use the print-out above or use attributes of `simple_model_results`.


```python
# Replace None with appropriate code

# Slope (coefficient of GrLivArea)
m = None

# Intercept (coefficient of const)
b = None

# your code here
raise NotImplementedError

print(f"""
Our simple linear regression model found a y-intercept
of ${round(b, 2):,}, then for every increase of 1 square foot
above-ground living area, the price increases by ${round(m, 2)} 
""")
```


```python
from numbers import Number

# m should be a number
assert isinstance(m, Number)

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```


```python
# b should be a number
assert isinstance(b, Number)

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

## 4. Build a Multiple Regression Model Using StatsModels

Now, build an OLS model that contains all of the columns present in `df`.

Specifically, your model should have `SalePrice` as the target, and these columns as features:

* `GrLivArea`
* `GarageArea`
* `LotArea`
* `LotFrontage`


```python
# Replace None with appropriate code

multiple_model = None

# your code here
raise NotImplementedError
multiple_model_results = multiple_model.fit()
print(multiple_model_results.summary())
```


```python
# multiple_model should be an OLS model
assert type(multiple_model) == sm.OLS

# multiple_model should have 4 features (other than the constant)
assert multiple_model.df_model == 4

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

## 5. Interpret Multiple Regression Model Results

Now we want to know: **is our multiple linear regression model a better fit than our simple linear regression model? We'll measure this in terms of percentage of variance explained (r-squared)**, where a higher r-squared indicates a better fit.

Replace `second_model_is_better` with either `True` if this model is better, or `False` if the previous model was better (or the two models are exactly the same).


```python
# Replace None with appropriate code
second_model_is_better = None
# your code here
raise NotImplementedError
```


```python
# second_model_is_better should be True or False
assert second_model_is_better == True or second_model_is_better == False

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

One of the feature coefficients is not statistically significant. Which one is it?

Replace `not_significant` with the name of the feature, which should be one of these four:

* `GrLivArea`
* `GarageArea`
* `LotArea`
* `LotFrontage`


```python
# Replace None with appropriate code
not_significant = None
# your code here
raise NotImplementedError
```


```python
# not_significant should be a string
assert type(not_significant) == str

# It should be one of the features in df
assert not_significant in df.columns

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```
