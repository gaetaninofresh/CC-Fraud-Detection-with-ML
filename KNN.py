import pandas
from sklearn.model_selection import train_test_split


#PREPARAZIONE DEL DATASET

dataset_path = './creditcard_2023.csv'
data = pandas.read_csv(dataset_path)

# rimuoviamo tutte le righe non classificate

data = data.dropna(subset=['Class'])

features = data.drop('Class', axis=1)
target = data['Class']

features_train, features_test, taget_train, target_test = train_test_split(features, target, test_size=0.2, random_state=99)