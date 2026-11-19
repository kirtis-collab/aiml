
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as s
from sklearn.model_selection import train_test_split

# 1. Load Dataset
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

df = pd.read_csv("D:/aiml/heart.csv")

print(df.head())
df.info()

print("\nColumn Names:\n", df.columns)
print("\nData Types:\n", df.dtypes)
print("\nData Type Count:\n", df.dtypes.value_counts())
print("\nDataset Description:\n", df.describe())

# 2. Dataset Information
n = df.select_dtypes(exclude='object').columns

print("\nNumeric Data:\n", df[n])
print("\nDataset Size:", df.size)
print("\nDataset Shape:", df.shape)

# 3. Missing Values and Duplicates
print("\nMissing Values:\n", df.isnull().sum())

df['age'] = df['age'].fillna(df['age'].median())
df = df.drop_duplicates()

print("\nMissing Values After Handling:\n", df.isnull().sum())
print("\nShape After Removing Duplicates:\n", df.shape)

# 4. Statistical Analysis
print("\nStatistical Summary:\n", df.describe())
print("\nMean:\n", df.mean(numeric_only=True))
print("\nMedian:\n", df.median(numeric_only=True))
print("\nStandard Deviation:\n", df.std(numeric_only=True))
print("\nCorrelation Matrix:\n", df.corr(numeric_only=True))

print("\nAverage Age:", s.mean(df['age']))

# 5. Data Visualization
for col, bins in [('age', 20), ('chol', 15)]:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col], bins=bins, edgecolor='black')
    plt.title('Histogram of ' + col)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col])
    plt.title('Boxplot of ' + col)
    plt.show()

# 6. Display Particular Columns
for col in ['age', 'sex', 'trestbps', 'chol']:
    print("\n", col, ":\n", df[col])

# 7. Train-Test Split
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=56
)

print("\nTraining Data:\n", X_train)
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)
