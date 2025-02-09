import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn.metrics import  precision_score, accuracy_score, recall_score, f1_score
from sklearnex import patch_sklearn
import os

local_path = os.path.dirname(__file__)

patch_sklearn()

res = load(f'{local_path}/KNN_results.joblib')

#min_idx_neigh = res.loc[res['Neighbors'].idxmin()]
max_idx_f1 = res.loc[res['F1-Score'].idxmax()]



best_par = pandas.DataFrame([max_idx_f1]).drop_duplicates()

knn = KNeighborsClassifier(
    n_neighbors= best_par['Neighbors'],
    weights=best_par['Weights'],
)

dump(knn, 'KNN_model.joblib')