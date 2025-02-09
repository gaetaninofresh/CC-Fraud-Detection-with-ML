from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas 

knn_params = load('./KNN/KNN_results.joblib')
mlp = load('./MultiLayerPerceptron/MLP_trained.joblib')
#dt = load('./DecisionTree/DT_trained.joblib')

path="."

dataset = pandas.read_csv(path+'\creditcard_2023.csv')
dataset = dataset.dropna(subset=['Class'])


X = dataset.drop(columns='Class', axis=1)
Y = dataset['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size=0.2, stratify=Y, random_state=1)

