import pandas



#Quick and dirty way to resize the dataset keeping the classes balanced
def sampleDataset(dataframe, frac):
  return pandas.concat(
      [
          dataframe.loc[dataframe['Class'] == 0].sample(frac=frac, random_state=1),
          dataframe.loc[dataframe['Class'] == 1].sample(frac=frac, random_state=1)
      ]
)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.neural_network import MLPClassifier
    from joblib import dump
    from sklearnex import patch_sklearn

    patch_sklearn()
    
    path = "./"
    data_full = pandas.read_csv(path+'creditcard_2023.csv')
    data_full = data_full.dropna(subset=['Class'])
    
    
    # Si è rivelato necessario prendere un sample più piccolo per effettuare 
    # la RandomSearch in tempi accettabili 
    balanced_subset = sampleDataset(data_full, .25)
    
    X = balanced_subset.drop(columns='Class', axis=1)
    Y = balanced_subset['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.2, stratify=Y, random_state=1)
    
    
    model= MLPClassifier()
    
    param_grid= {
        'hidden_layer_sizes': [(100,), (50,50), (100,50), (100,100), (200,100), (200,200), (100,50,25)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01], 
        'learning_rate': ['constant', 'adaptive'],
        'batch_size' : [32, 64, 128, 256],  
        'max_iter': [100],
        'early_stopping': [True, False],
        'validation_fraction': [0.1],
        'random_state': [1]
    }
    
    random_search = RandomizedSearchCV(model, param_grid, scoring='f1', n_jobs=-1, verbose=3 )
    random_search.fit(X_train, y_train)
    
    
    print(f"\nBest Params:\n{random_search.best_params_}\nScore:\n{random_search.best_score_}")
    
    dump(random_search, filename='MLP_random_search.joblib')



