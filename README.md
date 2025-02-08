# Progetto Introduzione All'Intelligenza Artificiale A.A. 2024/2025

## Introduzione

Il progetto mira a identificare le transazioni fraudolente utilizzando come metodologie di calssificazione K-Nearest Neighbor, un con Multi Layer Perceptron e Binary Decision Tree.

## Dataset

Abbiamo utilizzato il dataset [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/discussion?sort=undefined), i dati sono già stati normalizzati prima della pubblicazione per motivi di privacy e sicurezza e le classi erano già state bilanciate; non è stato quindi richiesto un preprocessing dei dati.

## Strumenti e librerie utilizzate

Abbiamo utilizzato le seguenti librerie Python:

* [pandas](https://pandas.pydata.org/) per gestire il dataset
* [sklearn](https://scikit-learn.org/dev/index.html) e la relativa patch [sklearnex](https://github.com/uxlfoundation/scikit-learn-intelex) per implementari i classificatori
* [numpy](https://numpy.org/) e [mathplotlib](https://matplotlib.org/) per la realizzazione dei grafici

## K-Nearest-Neighbor

Nella realizzazione di una classificatore KNN ci siamo trovati davanti alla necessità di valutare i miglior possibili iperparametri
