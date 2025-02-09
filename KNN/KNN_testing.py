import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
from sklearn.metrics import  precision_score, accuracy_score, recall_score, f1_score
from sklearnex import patch_sklearn

patch_sklearn()


#Quick and dirty way to resize the dataset keeping the classes balanced
def sampleDataset(dataframe, frac):
  return pandas.concat(
      [
          dataframe.loc[dataframe['Class'] == 0].sample(frac=frac, random_state=1),
          dataframe.loc[dataframe['Class'] == 1].sample(frac=frac, random_state=1)
      ]
)

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


path="."

data_full = pandas.read_csv(path+'\creditcard_2023.csv')
data_full = data_full.dropna(subset=['Class'])

data = sampleDataset(data_full, .5)


n_neighbors = [k for k in range(2, 16, 2)]
weights = ['uniform', 'distance']

features = data.drop('Class', axis=1)
target = data['Class']

results = compareModels(features, target, n_neighbors, weights, verbose=True)


dump(results, 'KNN_results.joblib')
dump()