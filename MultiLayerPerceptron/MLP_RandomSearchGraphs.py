from joblib import load 
import pandas 
import numpy as np
import matplotlib.pyplot as plt
import seaborn


random_search = load('random_search.joblib')
results = random_search.cv_results_

df_results = pandas.DataFrame(results['params'])
df_score = pandas.DataFrame(results['mean_test_score'])
df_score.columns=['mean_test_score']


df_results['mean_test_score'] = results['mean_test_score']


def plot_f1_vs_hyperparameters(df, hyperparameters, score_col='mean_test_score'):
    num_plots = len(hyperparameters)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))  # 2x3 grid for 6 hyperparameters
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
    
    for i, param in enumerate(hyperparameters):
        ax = axes[i]
        if df[param].dtype == object or isinstance(df[param].iloc[0], tuple):
            # Handle categorical or tuple-based parameters
            param_values = df[param].astype(str).unique()  # Convert to string for uniformity
            param_means = df.groupby(param)[score_col].mean()
            ax.bar(param_values, param_means, color='skyblue')
            ax.set_xticks(range(len(param_values)))
            ax.set_xticklabels(param_values, rotation=45)
        else:
            # Handle numeric parameterscore
            ax.scatter(df[param], df_score, color='skyblue')
        ax.set_xlabel(param)
        ax.set_ylabel('F1 Score')
        ax.set_title(f'F1 Score vs {param}')
    
    plt.tight_layout()
    plt.show()

# List of hyperparameters to plot
hyperparameters = ['hidden_layer_sizes', 'activation', 'solver', 'early_stopping', 'learning_rate', 'batch_size']

# Plot F1-score vs hyperparameters
plot_f1_vs_hyperparameters(df_results, hyperparameters)