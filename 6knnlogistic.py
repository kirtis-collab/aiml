
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# 1. Load Dataset
df = pd.read_csv("heart.csv")

print("First 5 records:")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Separate Features and Target
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures:")
print(X.head())
print("\nTarget:")
print(y.head())

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

# 4. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Create Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}

# 6. Train and Evaluate Models
for name, model in models.items():
    print("\n==============================")
    print(name.upper())
    print("==============================")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    results[name] = [acc, pre, rec, f1]

    print("Accuracy :", acc)
    print("Precision:", pre)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))

    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title(name + " - Confusion Matrix")
    plt.show()

# 7. Comparison of Both Models
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Logistic Regression": results["Logistic Regression"],
    "KNN": results["KNN"]
})

print("\n======================================")
print("COMPARISON OF BOTH MODELS")
print("======================================")
print(comparison)
