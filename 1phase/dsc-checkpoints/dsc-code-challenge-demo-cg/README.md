# Demo Code Challenge 

This demo is designed to give you the experience of a Code Challenge before your official Code Challenge.
 
*Read the instructions carefully.* Your code will need to meet detailed specifications to pass automated tests.

## Code Tests

We have provided some code tests for you to run to check that your work meets the item specifications. Passing these tests does not necessarily mean that you have gotten the item correct - there are additional hidden tests. However, if any of the tests do not pass, this tells you that your code is incorrect and needs changes to meet the specification. To determine what the issue is, read the comments in the code test cells, the error message you receive, and the item instructions.

## 1) Read `titanic.csv` into a pandas DataFrame named `df`

Use pandas to create a new DataFrame, called `df`, containing the data from the dataset in the file `titanic.csv` in the folder containing this notebook. 

Hint: Use the string `'./titanic.csv'` as the file reference.


```python
# Run this cell without changes

import pandas as pd
import numpy as np
from numbers import Number
import warnings
warnings.filterwarnings('ignore')
```


```python
# CodeGrade step1
# Replace None with appropriate code

df = None
```


```python
# This test confirms that you have created a DataFrame named df

assert type(df) == pd.DataFrame
```

### 2) Create a variable `dogwood_index` containing the index for the `'dogwood'` element from the list `tree_type_names`

Below is the list `tree_type_names` that you will need - do not change this list.



```python
# Run this cell without changes

tree_type_names = ['linden', 'spruce', 'dogwood', 'hickory', 'willow']
```


```python
# CodeGrade step2
# Replace None with appropriate code

dogwood_index = None
```


```python
# This test confirms that you have created a numeric variable named dogwood_index

assert isinstance(dogwood_index, Number)
```

## 3)  Briefly explain the purpose of code challenges in the Flatiron School Data Science program.

Enter a one sentence response below.


```python
# Your answer here


```


```python

```
