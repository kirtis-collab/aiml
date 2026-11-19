import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Original number of features:", X.shape[1])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification before PCA
model1 = LogisticRegression(max_iter=5000)

model1.fit(X_train_scaled, y_train)

y_pred1 = model1.predict(X_test_scaled)

accuracy_before = accuracy_score(y_test, y_pred1)

print("Accuracy before PCA:",
      accuracy_before)

# Apply PCA
pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Number of features after PCA:",
      X_train_pca.shape[1])

print("Explained variance:",
      pca.explained_variance_ratio_)

print("Total explained variance:",
      sum(pca.explained_variance_ratio_))

# Classification after PCA
model2 = LogisticRegression(max_iter=5000)

model2.fit(X_train_pca, y_train)

y_pred2 = model2.predict(X_test_pca)

accuracy_after = accuracy_score(y_test, y_pred2)

print("Accuracy after PCA:",
      accuracy_after)

# Visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - 2D Visualization")

plt.show()