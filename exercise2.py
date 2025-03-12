import pandas as pd
from numpy.ma.extras import average
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# (Optional) Convert to a Pandas DataFrame for easier viewing
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y
print(df.head())  # Uncomment to inspect

# 2. Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 3. Train a Naïve Bayes classifier (from Exercise 1)
gaussianModel = GaussianNB()
gaussianModel.fit(X_train, y_train)

# 4. Train a Logistic Regression model

logisticRegressionModel = LogisticRegression(class_weight='balanced', random_state=42)
logisticRegressionModel.fit(X_train, y_train)

gausY_pred = gaussianModel.predict(X_test)
logicY_pred = logisticRegressionModel.predict(X_test)

# 5. Compare metrics: accuracy, precision, and recall for each model
# Note: Because we have three classes in the Wine dataset, we set average='macro' (or 'weighted') for multi-class

print(accuracy_score(y_test, gausY_pred))
print(accuracy_score(y_test, logicY_pred))
# 6. Print results
print("---")
print(precision_score(y_test, gausY_pred, average='macro'))
print(precision_score(y_test, logicY_pred, average='macro'))
print("---")
print(recall_score(y_test, gausY_pred, average='macro'))
print(recall_score(y_test, logicY_pred, average='macro'))
# Optional: If you’d like to see a confusion matrix for each model
# from sklearn.metrics import confusion_matrix
