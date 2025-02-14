import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from joblib import load
from sklearn.model_selection import train_test_split

path = "./"
data = pandas.read_csv(path+'creditcard_2023.csv')
data = data.dropna(subset=['Class'])

X = data.drop(columns='Class', axis=1)
Y = data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size=0.2, stratify=Y, random_state=1)


best_params = load('DT_random_search.joblib')


decision_tree = DecisionTreeClassifier(random_state=42, **best_params)

# Get learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    decision_tree, X_train, Y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

# Compute mean and standard deviation
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training Accuracy", color="blue", marker="o")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.1)
plt.plot(train_sizes, test_mean, label="Validation Accuracy", color="green", marker="s")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="green", alpha=0.1)

plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve for Decision Tree Classifier")
plt.legend()
plt.grid()
plt.show()
