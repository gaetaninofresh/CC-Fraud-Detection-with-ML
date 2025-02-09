from joblib import load 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

import os

local_path = os.path.dirname(__file__)


random_search = load(f'{local_path}/MLP_random_search.joblib')
results = random_search.cv_results_

hyperparameters = ['hidden_layer_sizes', 'activation', 'solver', 'early_stopping', 'learning_rate', 'batch_size']

test_scores = results['mean_test_score']
f1_scores = results['mean_test_score']

params_df = pd.DataFrame(results['params'])
params_df['f1-score'] = results['mean_test_score']

# Ordinamento per f1-score
params_df = params_df.sort_values(by='f1-score', ascending=False)

# Visualizzazione della tabella con Seaborn
plt.figure(figsize=(20, 10))
ax = plt.gca()
ax.xaxis.set_visible(False)
table = plt.table(cellText=params_df.values,
                  colLabels=params_df.columns,
                  cellLoc='center',
                  loc='center',
                  colColours=['lightgray'] * params_df.shape[1])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.auto_set_column_width([i for i in range(len(params_df.columns))])
plt.title("Hyperparameter Configurations Sorted by F1 Score")
plt.show()


