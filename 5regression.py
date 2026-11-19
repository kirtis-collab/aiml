
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Dataset
print("Loading California Housing Dataset...")
df = fetch_california_housing(as_frame=True).frame

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 2. Split Features and Target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

# 3. Train Models
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
}

results = []
predictions = {}

for name, model in models.items():
    print("\nTraining", name, "...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred

    results.append([
        name,
        mean_squared_error(y_test, pred),
        mean_absolute_error(y_test, pred),
        r2_score(y_test, pred)
    ])

# 4. Display Results
results = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "R2 Score"])

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

for _, row in results.iterrows():
    print("\n" + row["Model"] + ":")
    print("-" * 30)
    print("MSE      :", round(row["MSE"], 4))
    print("MAE      :", round(row["MAE"], 4))
    print("R2 Score :", round(row["R2 Score"], 4))

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(results.to_string(index=False))

# 5. Save Results
results.to_csv("regression_results.csv", index=False)
print("\nResults saved to: regression_results.csv")

# 6. Plot Actual vs Predicted
plt.figure(figsize=(12, 5))

for i, (name, pred) in enumerate(predictions.items(), 1):
    plt.subplot(1, 2, i)
    plt.scatter(y_test, pred, alpha=0.3,
                color="blue" if i == 1 else "red")
    plt.xlabel("Actual House Value")
    plt.ylabel("Predicted House Value")
    plt.title(name)
    plt.grid(True)

plt.tight_layout()
plt.savefig("regression_comparison.png")
plt.show()

# 7. Conclusion
print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

linear = results.iloc[0]
poly = results.iloc[1]

print("Polynomial Regression has a higher R2 Score."
      if poly["R2 Score"] > linear["R2 Score"]
      else "Linear Regression has a higher R2 Score.")

print("Polynomial Regression has a lower MSE."
      if poly["MSE"] < linear["MSE"]
      else "Linear Regression has a lower MSE.")

print("Polynomial Regression has a lower MAE."
      if poly["MAE"] < linear["MAE"]
      else "Linear Regression has a lower MAE.")
