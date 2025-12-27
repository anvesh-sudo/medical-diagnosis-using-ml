# Diabetes Prediction using Machine Learning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("diabetes.csv")

# Split features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Sample prediction
sample = np.array([[5,116,74,0,0,25.6,0.201,30]])
result = model.predict(sample)

if result[0] == 1:
    print("Person is Diabetic")
else:
    print("Person is Not Diabetic")
