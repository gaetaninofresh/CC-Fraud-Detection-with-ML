import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import load, dump
from sklearnex import patch_sklearn

patch_sklearn()


path = "./"
data = pandas.read_csv(path+'creditcard_2023.csv')
data = data.dropna(subset=['Class'])

X = data.drop(columns='Class', axis=1)
Y = data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size=0.2, stratify=Y, random_state=1)


random_search = load('DT_random_search.joblib')

params = random_search.best_params_
decision_tree = DecisionTreeClassifier(**params, random_state=1, verbose=True)

decision_tree.fit(X_train, Y_train)

dump(decision_tree, 'DT_model.joblib')
