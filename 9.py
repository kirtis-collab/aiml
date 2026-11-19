
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define Models and Parameters
models = {
    "SVM": (
        Pipeline([("scaler", StandardScaler()), ("model", SVC())]),
        {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100], "max_depth": [None, 5, 10]}
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]}
    )
}

results = []

# 4. Train, Evaluate and Optimize
for name, (model, params) in models.items():

    model.fit(X_train, y_train)
    before = model.predict(X_test)

    grid = GridSearchCV(model, params, cv=5,
                        scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    after = best.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy Before (%)": round(accuracy_score(y_test, before)*100, 2),
        "F1 Before": round(f1_score(y_test, before), 4),
        "Accuracy After (%)": round(accuracy_score(y_test, after)*100, 2),
        "Precision After": round(precision_score(y_test, after), 4),
        "Recall After": round(recall_score(y_test, after), 4),
        "F1 After": round(f1_score(y_test, after), 4),
        "Best Parameters": str(grid.best_params_)
    })

# 5. Display Results
df = pd.DataFrame(results)
print(df.to_string(index=False))
