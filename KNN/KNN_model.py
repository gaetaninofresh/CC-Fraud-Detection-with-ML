import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn.metrics import  precision_score, accuracy_score, recall_score, f1_score
from sklearnex import patch_sklearn

patch_sklearn()

params = load('KNN_results.joblib')
print(params)
