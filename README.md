# Machine-Learning
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model
_
selection import train
test
_
_
split
from sklearn.preprocessing import StandardScaler
from sklearn.linear
_
model import LogisticRegression
from sklearn.metrics import classification
_
report, confusion
_
matrix, accuracy_
score, roc
auc
score, roc
_
_
_
# Load Dataset
df = pd.read
_
csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv') # update path if local
# Data Preprocessing
df.drop(['id'
,
'Unnamed: 32'], axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1,
'B': 0})
# Splitting Features and Labels
X = df.drop('diagnosis'
, axis=1)
curve
y = df['diagnosis']
# Standardizing Data
scaler = StandardScaler()
X
scaled = scaler.fit
_
_
transform(X)
# Train-Test Split
X
train, X
_
_
test, y_
train, y_
test = train
test
_
_
split(X
_
scaled, y, test
size=0.2, random
_
_
# Logistic Regression Model
model = LogisticRegression()
model.fit(X
_
train, y_
train)
# Predictions
y_pred = model.predict(X
_
test)
# Evaluation
print("Accuracy:"
, accuracy_
score(y_
test, y_pred))
print("Classification Report:\n"
, classification
_
report(y_
test, y_pred))
print("Confusion Matrix:\n"
, confusion
_
matrix(y_
test, y_pred))
# ROC Curve
y_prob = model.predict
_proba(X
_
test)[:, 1]
fpr, tpr, _
= roc
_
curve(y_
test, y_prob)
auc = roc
auc
_
_
score(y_
test, y_prob)
# Plot ROC
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1],
'k--
')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
