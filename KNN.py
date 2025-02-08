import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from MLPRandomSearch import sampleDataset
from joblib import dump
from sklearn.metrics import  precision_score, accuracy_score, recall_score, f1_score
from sklearnex import patch_sklearn

patch_sklearn()


path="."
n_neighbors = 10

data_full = pandas.read_csv(path+'\creditcard_2023.csv')
data_full = data_full.dropna(subset=['Class'])



data = sampleDataset(data_full, .5)

def compareModels(features, target, n_neighbors, weights, verbose=False):
  results = []
  i = 0
  for nn in n_neighbors:
    for w in weights:
      model = KNeighborsClassifier(
          n_neighbors=nn,
          weights=w,
          algorithm='auto'
          )
      model.fit(features, target)
      target_pred = model.predict(features)

      #Cut ROC-AUC metrics because it was too intensive and it didn't give any significant results (it was 1.0 for pretty much every model)
      #prob_pred = model.predict_proba(features)[:, 1]
      results.append(
          {
              "Neighbors" : nn,
              "Weights" : w,
              "Accuracy" : accuracy_score(target, target_pred),
              "Precision" : precision_score(target, target_pred),
              "Recall" : recall_score(target, target_pred),
              "F1-Score" : f1_score(target, target_pred),
              #"ROC-AUC" : roc_auc_score(target_pred, prob_pred)
          }
      )
      if verbose:
        print(f"{i+1}/{len(n_neighbors)*len(weights)}\t", results[i])
        i+=1

  return pandas.DataFrame(results)


n_neighbors = [k for k in range(2, 16, 2)]
weights = ['uniform', 'distance']

data = sampleDataset(data_full, .50)
features = data.drop('Class', axis=1)
target = data['Class']

results = compareModels(features, target, n_neighbors, weights, verbose=True)

dump(results, 'KNN_results.joblib')