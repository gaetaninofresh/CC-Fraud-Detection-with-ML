import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump
from sklearnex import patch_sklearn

patch_sklearn()


path = "./"
data = pandas.read_csv(path+'creditcard_2023.csv')
data = data.dropna(subset=['Class'])

X = data.drop(columns='Class', axis=1)
Y = data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size=0.2, stratify=Y, random_state=1)


decision_tree = DecisionTreeClassifier(random_state=42)

param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(decision_tree, param_grid,scoring='f1', n_jobs=-3, verbose=True)
random_search.fit(X_train, Y_train)


# Print the best hyperparameters and accuracy score
print("Best Hyperparameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)

# Evaluate the best model on the test data
best_model = random_search.best_estimator_
test_accuracy = best_model.score(X_test, Y_test)
print("Test Accuracy:", test_accuracy)

dump(random_search, 'DT_random_search.joblib')
