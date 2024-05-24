# Homogeneous Ensemble Methods

This repository contains a Jupyter Notebook that explores homogeneous ensemble methods in machine learning. By using the same type of base model in different ensemble techniques, we can improve predictive performance and model stability. The notebook demonstrates how to implement and evaluate bagging and boosting ensembles using decision tree regressors, showcasing their effectiveness with practical examples.

## Overview

### Homogeneous Ensembles
Homogeneous ensembles use multiple instances of the same model type to create a more robust predictive model. This notebook covers:
- **Bagging (Bootstrap Aggregating)**
- **Boosting**

## Notebook Content

### 1. Homogeneous Ensembling in Python

#### Import Libraries and Data
First, we import the necessary libraries and load the dataset. We visualize the data to understand its distribution.

```python
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("https://github.com/Explore-AI/Public-Data/blob/master/house_price_by_area.csv?raw=true")
X = df["LotArea"] # Independent variable
y = df["SalePrice"] # Dependent variable

plt.scatter(X, y)
plt.title("House Price vs Area")
plt.xlabel("Lot Area in m$^2$")
plt.ylabel("Sale Price in Rands")
plt.show()
```

#### Pre-processing
We normalize the data to ensure that the models perform optimally.

```python
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(np.array(X)[:, np.newaxis])
y_scaled = y_scaler.fit_transform(np.array(y)[:, np.newaxis])

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=6)
```

#### a) Bagging (Bootstrap Aggregating)
We train a bagging ensemble using decision tree regressors and evaluate its performance.

```python
from sklearn.ensemble import BaggingRegressor

# Instantiate decision tree regression model to use as the base model
d_tree = DecisionTreeRegressor(max_depth=4)

# Instantiate BaggingRegressor model with a decision tree as the base model
bag_reg = BaggingRegressor(estimator=d_tree)

# Train the bagging ensemble
bag_reg.fit(x_train, y_train[:, 0])

y_pred = bag_reg.predict(x_test)
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plotting
x_domain = np.linspace(min(x_train), max(x_train), 100)
y_pred_rescaled = y_scaler.inverse_transform(bag_reg.predict(x_domain).reshape(-1, 1))
x_rescaled = x_scaler.inverse_transform(x_domain)

plt.figure()
plt.scatter(X, y)
plt.plot(x_rescaled, y_pred_rescaled, color="red", label='predictions')
plt.xlabel("LotArea in m$^2$")
plt.ylabel("SalePrice in Rands")
plt.title("Decision Tree Bagging Regression")
plt.legend()
plt.show()
```
**Note:** The RMSE error metric can change with different runs due to the random sampling process in bagging. Setting a `random_state` ensures reproducibility.

#### b) Boosting
We train a boosting ensemble using decision tree regressors and evaluate its performance.

```python
from sklearn.ensemble import AdaBoostRegressor

# Instantiate decision tree regression model to use as the base model
d_tree = DecisionTreeRegressor(max_depth=3)

# Instantiate AdaBoostRegressor model with a decision tree as the base model
bst_reg = AdaBoostRegressor(estimator=d_tree)

# Train the boosting ensemble
bst_reg.fit(x_train, y_train[:, 0])

y_pred = bst_reg.predict(x_test)
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plotting
x_domain = np.linspace(min(x_train), max(x_train), 100)
y_pred_rescaled = y_scaler.inverse_transform(bst_reg.predict(x_domain).reshape(-1, 1))
x_rescaled = x_scaler.inverse_transform(x_domain)

plt.figure()
plt.scatter(X, y)
plt.plot(x_rescaled, y_pred_rescaled, color="red", label='predictions')
plt.xlabel("LotArea in m$^2$")
plt.ylabel("SalePrice in Rands")
plt.title("Decision Tree Boosting Regression")
plt.legend()
plt.show()
```

### Results
The notebook demonstrates how homogeneous ensemble methods can improve predictive performance by combining multiple instances of the same model type. Bagging and boosting both show significant improvements over a single decision tree regressor.

## Usage
To run this notebook, clone this repository and open the notebook in Jupyter:

```bash
git clone https://github.com/yourusername/Homogeneous-Ensemble-Methods.git
cd Homogeneous-Ensemble-Methods
jupyter notebook
```

## Conclusion
This notebook provides a comprehensive introduction to homogeneous ensemble methods, showing how to build, train, and evaluate various ensemble models. It serves as a valuable resource for data scientists and machine learning practitioners interested in enhancing their models' predictive performance.

Contributions and feedback are welcome! Feel free to open issues or submit pull requests to improve this repository.
