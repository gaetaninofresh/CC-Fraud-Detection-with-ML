# Credit card fraud detection using machine learning

L'obiettivo di questo progetto è sviluppare e confrontare diversi modelli di machine learning per la classificazione di transazioni fraudolente, utilizzando un dataset di transazioni con carte di credito. In particolare, sono stati implementati e analizzati tre classificatori:

1. **k-Nearest Neighbors (kNN)**
2. **Multi-Layer Perceptron (MLP)**
3. **Decision Tree (DT)**

Questa analisi mira a valutare l'efficacia di ciascun modello nel rilevare attività sospette, confrontandone le prestazioni in termini di accuratezza e altre metriche di valutazione.

## Dataset

Per questo progetto è stato utilizzato il dataset: [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/discussion?sort=undefined). I dati sono stati preventivamente normalizzati per motivi di privacy e sicurezza e le classi risultano già bilanciate.

## Strumenti e librerie utilizzate

Per l'implementazione dei modelli e l'analisi dei dati sono state utilizzate le seguenti librerie e strumenti:

- **[Scikit-Learn (sklearn)](https://scikit-learn.org/stable/)**: per l'implementazione dei classificatori (kNN, MLP, DT) e per la valutazione delle prestazioni.
- **[Sklearnex](https://intel.github.io/scikit-learn-intelex/)**: una patch per ottimizzare le prestazioni di scikit-learn su hardware moderno.
- **[NumPy](https://numpy.org/)**: per la manipolazione dei dati e le operazioni numeriche.
- **[Matplotlib](https://matplotlib.org/)**: per la visualizzazione dei grafici e delle curve di apprendimento.

## Analisi del dataset

### Caricamento ed esplorazione del dataset

Abbiamo utilizzato la libreria **pandas** per caricare e analizzare il dataset. Il dataset è stato letto tramite la funzione built-in di pandas per la lettura di file CSV. Per visualizzare un'anteprima delle prime rigeh del dataset abbiamo utilizzato la funzione *head()*. Il dataset è composto da  ***31 colonne,*** tra cui:

* **id** : identificativo univoco della transazione
* **V1-V28** : vettori delle feature (dati anonimizzati)
* **Amount** : importo della transazione
* **Class** : etichetta binaria che indica se la transazione è fraudolenta *1* o legittima *0*

### Verifica valori mancanti

Per verificare la presenza di valori mancanti abbiamo utilizzato la funzione *isnull().sum()*, in questo modo abbiamo potuto confermare che il datatset non contenesse valori nulli.

### Separazione delle feature e delle label

Abbiamo separato le feature *x* dalle label *y* utilizzando le funzioni di pandas. Le feature sono tutte le colonne eccettp la colonna *Class,* che rappresenta la label. Inoltre per facilitare una successiva elaborazione abbiamo convertito i DataFrame di pandas in array tramite NumPy.

### Suddivisione in training set e test set

Abbiamo suddiviso il dataset,seguendo la regola dell'**holdout set** in un training set con l'*80%* dei dati e il restante *20%* è stato utilizzato per il test set. Al fine di mantenere il bilanciamento delle classi, abbiamo utilizzato il parametro *stratify.*

### Normalizzazione dei dati

Infine, abbiamo normalizzato i dati utilizzando la classe *StandardScaler* di *scikit-learn*. Sebbene i dati fossero già normalizzati abbiamo optato per una seconda normalizzazione per garantire che tutte le feature avessero la stessa scala. Con questi passaggi abbiamo preparato il dataset per l'addestramento dei vari modelli di machine learning.

### Metriche di valutazione:

Poiché il dataset è bilanciato, abbiamo scelto di utilizzare metriche di valutazione semplici ma efficaci, come l' **accuratezza, precision** e **recall**. In particolare:

* **Precision** : utile per valutare la capacità del modello di ridurre i falsi positivi.
* **Recall** : fondamentale per comprendere quanto bene il modello minimizzi i falsi negativi.

Queste metriche ci permettono di analizzare il bilanciamento tra falsi positivi e falsi negativi, ottimizzando così le prestazioni del classificatore.

## MLP (Multi-layer-perceptron)

Per l'addestramento del classificatore basato su MLP abbiamo utilizzato la classe ***MLPClassifier*** della libreria *s***cikit-learn**. Per individuare i migliori iperparametri, ci siamo avvalsi della classe ***RandomizedSearchCV***, che effettua una ricerca casuale nello spazio degli iperparametri testando diverse combinazioni e selezionando valori casuali all'interno di un intervallo specificato dall'utente.

Il risultato migliore è stato ottenuto con i seguenti valori per ogni iperparametro:

![1739138136074](image/README/1739138136074.png)

Con la migliore configurazione il modello ha raggiunto un'accuratezza del **99,84%** nella classificazione. Inoltre, abbiamo visualizzato l'andamento della funzione di perdita durante l'addestramento, mostrando come le prestazioni del modello siano migliorate con l'aumento delle epoche.

![1739121890539](image/README/1739121890539.png "Learning curve for Multi-Layer-Perceptron")

* Training Accuracy (blu): l'accuratezza del modello sul set di addestramento.
* Validation Accuracy (verde): l'accuratezza del modello sul set di validazione

Possiamo osservare che, all'aumentare del numero di campioni di addestramento, l'accuratezza cresce rapidamente nelle prime fasi e poi si stabilizza intorno al 99,84%, confermando la buona capacità di generalizzazione del modello. La zona d'ombra intorno alle curve rappresenta la varianza tra le diverse iterazioni dell'addestramento.

## kNN (key - Nearest - Neighbor)

Per l'addestramento del classificatore basato sull'algoritmo k-Nearest-Neighbor (kNN), abbiamo utilizzato la classe *KneighborClassifier* della libreria **scikit-learn**. Il modello è stato testato utilizzando due approcci distinti per determinare quale metodo fosse più efficace nella classificazione dei dati: 

* **Approccio Uniforme**: in questo approccio tutti i vicini hanno lo stesso peso nel determinare la classe di un punto, indipendentemente dalla loro distanza. Abbiamo testato il modello con valori del parametro *k* (numero di vicini) compresi tra 2 e 14.
* **Approccio Basato sulla Distanza**: in questo caso invece i vicini vengono pesati in base alla loro distanza dal punto da classificare. I punti più vicini hanno un peso maggiore nel determinare la classe.

Per confrontare i due approcci, abbiamo definito una funzione chiamata *compareModels*, che prende in input i seguenti parametri:

* *features*: il vettore delle feature utilizzate per l'addestramento.
* *target*: la classe target da predire.
* *n_neighbors*: una lista di valori per il parametro k (numero di vicini).
* *weights*: la modalità di peso da utilizzare (uniform o distance).

![1739124853900](image/README/1739124853900.png)

Il grafico mostra il confronto tra le performance del modello *kNN* utilizzando due diverse modalità di peso: uniforme e basato sulla distanza. L'asse orizzontale rappresenta il numero di vicini *k*, mentre l'asse verticale rappresenta il valore di *F1-Score*.

* **Linea Blu (Distance)**: rappresenta il valore di *F1-Score* ottenuto dal modello kNN quando i vicini vengono pesati in base alla loro distanza. I punti più vicini al punto da classificare hanno un peso maggiore.
* **Linea Arancione (Uniform)**: rappresenta il valore di *F1-Score* ottenuto dal modello kNN quando tutti i vicini hanno lo stesso peso, indipendentemente dalla loro distanza.

## DT (Decision Tree)

Per l'addestamento del classificatore basato sull'albero di decisione binario abbiamo utilizzato la classe *DecisionTreeClassifier della libreria **scikit-Learn***. Per ottimizzare le prestazioni del modello abbiamo utilizzato ***RandomizedSearchCV***, una tecnica di ricerca casuale degli iperparametri, valutando diverse combinazioni per migliorare il punteggio F1. La ricerca è stata effettuata con la seguente griglia di iperparametri:

1. *criterion* [gini, entropy, log_loss]: funzione di impurità utilizzata per la suddivisione dei nodi.
2. *splitter* [best, random]: strategia di scelta delle feature per la suddivisione.
3. *max_depth* [10, 20, 30]: profondità massima dell'albero.
4. *min_samples_split* [2, 5, 10]: numero minimo di campioni richiesti per dividere un nodo.
5. *min_samples_leaf* [1, 2, 4]: numero minimo di campioni in un nodo foglia

La ricerca degli iperparametri è stata eseguita utlizzando una 5 fold cross validation, utilizzando *F1-Scrore* come metrica di valutazione e parallelizzando il calcolo con *n_jobs = -1.*

Il modello ha raggiunto un'accuratezza sul **test set** pari a *99,97%* con questi iperparametri:

* *Criterion: gini*
* *Splitter: best*
* *Max_depth: 10*
* *Min_samples_split: 2*
* *Min_samples_leaf: 2*

Dopo aver selezionato il miglior modello, ne abbiamo valutato l'andamento utilizzando una **learning curve** che mostra il progresso della performance su training e validation set in funzione della dimensione dei dati di addestramento.

![1739127377827](image/README/1739127377827.png)

## Conclusioni

Il progetto ha dimostrato che tutti e tre i modelli *MLP, kNN, DT* sono in grado di classificare con successo le transazioni fraudolente con un'accuratezza molto elevata, con kNN che ha raggiunto la migliore accuratezza sul set di test. Tuttavia, la scelta del modello migliore dipende dal contesto specifico e dalle esigenze del sistema di rilevazione delle frodi. Ad esempio, MLP potrebbe essere preferibile in scenari in cui è richiesta una maggiore generalizzazione, mentre kNN potrebbe essere utile per la sua semplicità e rapidità di implementazione.

| Modello | Accuracy | Precision | Recall   | F1-Score |
| :-----: | -------- | --------- | -------- | :------: |
|   kNN   | 1.000000 | 1.000000  | 1.000000 | 1.000000 |
|   MLP   | 0,998426 | 0.998427  | 0.998426 | 0,998426 |
|   DC   | 0.999710 | 0.999710  | 0.999710 | 0.999710 |



---
