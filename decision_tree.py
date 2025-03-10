import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Create synthetic dataset
np.random.seed(42)
X = np.random.rand(100, 1) * 100  # House sizes
y = 5 * X + np.random.randn(100, 1) * 10  # Price with noise

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions
y_tree_pred = tree_model.predict(X_test)

# Plot results
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
plt.scatter(X_test, y_tree_pred, color="green", label="Decision Tree Predictions", alpha=0.7)
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price ($)")
plt.legend()
plt.title("Decision Tree - House Price Prediction")
plt.show()




