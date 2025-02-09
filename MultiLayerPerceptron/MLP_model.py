import pandas
from sklearn.model_selection import train_test_split
from joblib import load, dump
from sklearn.neural_network import MLPClassifier
from sklearnex import patch_sklearn

patch_sklearn()

path = "./"
data = pandas.read_csv(path+'creditcard_2023.csv')
data = data.dropna(subset=['Class'])

random_search = load('mlp_random_search.joblib')
best_params = random_search.best_params_

model = MLPClassifier(**best_params, verbose=True)



X = data.drop(columns='Class', axis=1)
Y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.2, stratify=Y, random_state=1)

model.fit(X, Y)

dump(model, 'TrainedMLP.joblib')
