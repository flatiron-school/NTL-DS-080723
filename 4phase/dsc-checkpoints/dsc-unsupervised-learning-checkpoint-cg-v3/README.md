# Unsupervised ML Checkpoint

This checkpoint is designed to test your understanding of unsupervised machine learning.

Specifically, this will cover:

* Performing clustering analysis of data, including interpreting silhouette scores
* Creating visualizations using unsupervised ML

## Your Task: Use Unsupervised ML to Investigate a Sensor Dataset

### Data Understanding

You will be using a dataset ([source](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)) generated from embedded sensors (accelerometer and gyroscope) in a smartphone worn on a person's waist. Additional processing and feature engineering has already been applied to the data, including noise filters, resulting in a 561-feature vector.

In the cell below, we load the data into a `pandas` dataframe:


```python
# Run this cell without changes
import pandas as pd
import numpy as np

df = pd.read_csv("sensor_data.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tBodyAcc-mean()-X</th>
      <th>tBodyAcc-mean()-Y</th>
      <th>tBodyAcc-mean()-Z</th>
      <th>tBodyAcc-std()-X</th>
      <th>tBodyAcc-std()-Y</th>
      <th>tBodyAcc-std()-Z</th>
      <th>tBodyAcc-mad()-X</th>
      <th>tBodyAcc-mad()-Y</th>
      <th>tBodyAcc-mad()-Z</th>
      <th>tBodyAcc-max()-X</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag-meanFreq()</th>
      <th>fBodyBodyGyroJerkMag-skewness()</th>
      <th>fBodyBodyGyroJerkMag-kurtosis()</th>
      <th>angle(tBodyAccMean,gravity)</th>
      <th>angle(tBodyAccJerkMean),gravityMean)</th>
      <th>angle(tBodyGyroMean,gravityMean)</th>
      <th>angle(tBodyGyroJerkMean,gravityMean)</th>
      <th>angle(X,gravityMean)</th>
      <th>angle(Y,gravityMean)</th>
      <th>angle(Z,gravityMean)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.288585</td>
      <td>-0.020294</td>
      <td>-0.132905</td>
      <td>-0.995279</td>
      <td>-0.983111</td>
      <td>-0.913526</td>
      <td>-0.995112</td>
      <td>-0.983185</td>
      <td>-0.923527</td>
      <td>-0.934724</td>
      <td>...</td>
      <td>-0.074323</td>
      <td>-0.298676</td>
      <td>-0.710304</td>
      <td>-0.112754</td>
      <td>0.030400</td>
      <td>-0.464761</td>
      <td>-0.018446</td>
      <td>-0.841247</td>
      <td>0.179941</td>
      <td>-0.058627</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.278419</td>
      <td>-0.016411</td>
      <td>-0.123520</td>
      <td>-0.998245</td>
      <td>-0.975300</td>
      <td>-0.960322</td>
      <td>-0.998807</td>
      <td>-0.974914</td>
      <td>-0.957686</td>
      <td>-0.943068</td>
      <td>...</td>
      <td>0.158075</td>
      <td>-0.595051</td>
      <td>-0.861499</td>
      <td>0.053477</td>
      <td>-0.007435</td>
      <td>-0.732626</td>
      <td>0.703511</td>
      <td>-0.844788</td>
      <td>0.180289</td>
      <td>-0.054317</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.279653</td>
      <td>-0.019467</td>
      <td>-0.113462</td>
      <td>-0.995380</td>
      <td>-0.967187</td>
      <td>-0.978944</td>
      <td>-0.996520</td>
      <td>-0.963668</td>
      <td>-0.977469</td>
      <td>-0.938692</td>
      <td>...</td>
      <td>0.414503</td>
      <td>-0.390748</td>
      <td>-0.760104</td>
      <td>-0.118559</td>
      <td>0.177899</td>
      <td>0.100699</td>
      <td>0.808529</td>
      <td>-0.848933</td>
      <td>0.180637</td>
      <td>-0.049118</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.279174</td>
      <td>-0.026201</td>
      <td>-0.123283</td>
      <td>-0.996091</td>
      <td>-0.983403</td>
      <td>-0.990675</td>
      <td>-0.997099</td>
      <td>-0.982750</td>
      <td>-0.989302</td>
      <td>-0.938692</td>
      <td>...</td>
      <td>0.404573</td>
      <td>-0.117290</td>
      <td>-0.482845</td>
      <td>-0.036788</td>
      <td>-0.012892</td>
      <td>0.640011</td>
      <td>-0.485366</td>
      <td>-0.848649</td>
      <td>0.181935</td>
      <td>-0.047663</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.276629</td>
      <td>-0.016570</td>
      <td>-0.115362</td>
      <td>-0.998139</td>
      <td>-0.980817</td>
      <td>-0.990482</td>
      <td>-0.998321</td>
      <td>-0.979672</td>
      <td>-0.990441</td>
      <td>-0.942469</td>
      <td>...</td>
      <td>0.087753</td>
      <td>-0.351471</td>
      <td>-0.699205</td>
      <td>0.123320</td>
      <td>0.122542</td>
      <td>0.693578</td>
      <td>-0.615971</td>
      <td>-0.847865</td>
      <td>0.185151</td>
      <td>-0.043892</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7347</th>
      <td>0.299665</td>
      <td>-0.057193</td>
      <td>-0.181233</td>
      <td>-0.195387</td>
      <td>0.039905</td>
      <td>0.077078</td>
      <td>-0.282301</td>
      <td>0.043616</td>
      <td>0.060410</td>
      <td>0.210795</td>
      <td>...</td>
      <td>-0.070157</td>
      <td>-0.588433</td>
      <td>-0.880324</td>
      <td>-0.190437</td>
      <td>0.829718</td>
      <td>0.206972</td>
      <td>-0.425619</td>
      <td>-0.791883</td>
      <td>0.238604</td>
      <td>0.049819</td>
    </tr>
    <tr>
      <th>7348</th>
      <td>0.273853</td>
      <td>-0.007749</td>
      <td>-0.147468</td>
      <td>-0.235309</td>
      <td>0.004816</td>
      <td>0.059280</td>
      <td>-0.322552</td>
      <td>-0.029456</td>
      <td>0.080585</td>
      <td>0.117440</td>
      <td>...</td>
      <td>0.165259</td>
      <td>-0.390738</td>
      <td>-0.680744</td>
      <td>0.064907</td>
      <td>0.875679</td>
      <td>-0.879033</td>
      <td>0.400219</td>
      <td>-0.771840</td>
      <td>0.252676</td>
      <td>0.050053</td>
    </tr>
    <tr>
      <th>7349</th>
      <td>0.273387</td>
      <td>-0.017011</td>
      <td>-0.045022</td>
      <td>-0.218218</td>
      <td>-0.103822</td>
      <td>0.274533</td>
      <td>-0.304515</td>
      <td>-0.098913</td>
      <td>0.332584</td>
      <td>0.043999</td>
      <td>...</td>
      <td>0.195034</td>
      <td>0.025145</td>
      <td>-0.304029</td>
      <td>0.052806</td>
      <td>-0.266724</td>
      <td>0.864404</td>
      <td>0.701169</td>
      <td>-0.779133</td>
      <td>0.249145</td>
      <td>0.040811</td>
    </tr>
    <tr>
      <th>7350</th>
      <td>0.289654</td>
      <td>-0.018843</td>
      <td>-0.158281</td>
      <td>-0.219139</td>
      <td>-0.111412</td>
      <td>0.268893</td>
      <td>-0.310487</td>
      <td>-0.068200</td>
      <td>0.319473</td>
      <td>0.101702</td>
      <td>...</td>
      <td>0.013865</td>
      <td>0.063907</td>
      <td>-0.344314</td>
      <td>-0.101360</td>
      <td>0.700740</td>
      <td>0.936674</td>
      <td>-0.589479</td>
      <td>-0.785181</td>
      <td>0.246432</td>
      <td>0.025339</td>
    </tr>
    <tr>
      <th>7351</th>
      <td>0.351503</td>
      <td>-0.012423</td>
      <td>-0.203867</td>
      <td>-0.269270</td>
      <td>-0.087212</td>
      <td>0.177404</td>
      <td>-0.377404</td>
      <td>-0.038678</td>
      <td>0.229430</td>
      <td>0.269013</td>
      <td>...</td>
      <td>-0.058402</td>
      <td>-0.387052</td>
      <td>-0.740738</td>
      <td>-0.280088</td>
      <td>-0.007739</td>
      <td>-0.056088</td>
      <td>-0.616956</td>
      <td>-0.783267</td>
      <td>0.246809</td>
      <td>0.036695</td>
    </tr>
  </tbody>
</table>
<p>7352 rows × 561 columns</p>
</div>



### Data Processing and Visualization

As you can see, this dataset has many features (561 total). There is also likely to be high multicollinearity between these features due to the feature engineering process, which repeatedly used some of the same raw sensor data (e.g. "tBodyAcc-mean()") to generate different columns. The code below will step through a technique called Principal Component Analysis (PCA), which will greatly reduce the dimensionality of our problem.

Your task is to look for some underlying patterns in this data using k-means clustering. You do not need to perform a train-test split.

First you'll need to scale the data, which is an important first step in PCA.

## 1. Prepare Data for Principal Component Analysis (PCA)

Instantiate a `StandardScaler` and use it to create a scaled variable called `data_scaled`.


```python
# CodeGrade step1
# Replace None with appropriate code

# Import relevant class
None

# Create scaled variable
data_scaled = None
data_scaled

# Convert data_scaled to a DataFrame for readability
data_scaled = pd.DataFrame(data_scaled, columns=df.columns)
data_scaled
```


```python
# data_scaled should have the same shape as df
assert data_scaled.shape == df.shape

# data_scaled should not be the same as df
assert data_scaled.loc[0,"angle(Z,gravityMean)"] != df.loc[0,"angle(Z,gravityMean)"]
```

## Perform PCA on the Dataset (Run Cells without Changes)


```python
# Run this cell without changes

from sklearn.decomposition import PCA
pca_transformer = PCA(n_components=0.95, random_state=42)
pca_data = pca_transformer.fit_transform(data_scaled)
pca_data
```




    array([[-16.13854371,   2.15202401,   3.14478025, ...,  -1.68153546,
             -1.20932492,  -1.17572672],
           [-15.2961943 ,   1.38714378,  -0.68222107, ...,  -1.34739246,
              0.14947399,  -0.73061489],
           [-15.13701861,   2.47335094,  -1.75664057, ...,   0.13803147,
              0.66226306,  -0.22741812],
           ...,
           [ 14.33343587, -12.26071193,   4.0259462 , ...,  -1.32974137,
             -0.08800409,  -0.09253083],
           [ 12.87601895, -14.07125595,   2.91606098, ...,  -0.79743323,
             -0.71168419,   0.56868751],
           [ 13.01610365, -12.24426121,   1.33604965, ...,  -0.043166  ,
             -0.34681818,  -0.67183662]])




```python
# Run this cell without changes
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(x=pca_data[:, 0], y=pca_data[:, 1])
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")
ax.set_title("Visualizing the First Two Principal Components of Sensor Data");
```


    
![png](index_files/index_8_0.png)
    


## 2. Interpret a Silhouette Plot to Perform Clustering Analysis

You handed the PCA-transformed dataset to a coworker, who produced this silhouette plot. Interpret the plot to choose the optimal number of clusters, then use the scikit-learn `KMeans` class ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)) to instantiate a KMeans model.

<!-- 
k_values = range(2,21)
silhouette_scores = [0.41540858143541637,
 0.3438069022316109,
 0.1775373667655337,
 0.1425018071505172,
 0.12365576879282861,
 0.11917044787937144,
 0.09524343091496036,
 0.09155853204663812,
 0.09410174137993227,
 0.09335504033772586,
 0.09386447654377134,
 0.09518329181757086,
 0.0864125952331548,
 0.08622712573000349,
 0.08350384589108541,
 0.07589314716366936,
 0.06926851258928322,
 0.06980908513512903,
 0.0710121422731501]

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(k_values, silhouette_scores, color="green", marker="s", )
ax.set_xticks(k_values)
ax.set_xlabel("k in KMeans")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Scores for KMeans Models of Varying K Values");
-->

![plot of silhouette scores, with the highest y value at an x value of 2](silhouette_score_plot.png)

Call the model `kmeans`, and use a `random_state` of 42.


```python
# CodeGrade step2
# Replace None with appropriate code

# Import relevant model
None

# Instantiate KMeans model
kmeans = None

# Fit the model on pca_data, using the best n_clusters value as indicated by the plot above
None

kmeans
```


```python
# kmeans should be a fitted KMeans model with as many dimensions in cluster
# centers as principal components
assert kmeans.cluster_centers_.shape[1] == pca_data.shape[1]
```

## 3. Update the Plot to Reflect the Clusters

Use the `kmeans` object to identify the labels for each data point in `pca_data`. Then modify the plotting code below so that the color of each dot on the scatterplot indicates which cluster the data point belongs to.

***Hint:*** The `c` keyword argument in the `scatter` method allows you to control the color of the dots.


```python
# CodeGrade step3
# Replace None with appropriate code
cluster_labels = None

fig, ax = plt.subplots()

# Modify the below line of code to specify the color:
scatter = ax.scatter(x=pca_data[:, 0], y=pca_data[:, 1])

# The rest of the code can be run as-is:
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")
ax.set_title("Visualizing the First Two Principal Components of Sensor Data")
# Un-comment the below line to add a legend once you have specified the color
# ax.legend(*scatter.legend_elements(fmt="Cluster {x:.0f}"));
```


```python
# cluster_labels should be a NumPy array
assert type(cluster_labels) == np.ndarray
```


```python
# Plot should have the same number of colors as kmeans has clusters
assert len(scatter.legend_elements()[0]) == kmeans.n_clusters
```


```python

```
