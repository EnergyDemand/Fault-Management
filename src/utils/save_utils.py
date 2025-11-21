import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np


def reformat_large_tick_values(tick_val, pos):
    if tick_val >= 1000000000:
        val = round(tick_val / 1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val / 1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val / 1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val
    new_tick_format = str(new_tick_format)
    index_of_decimal = new_tick_format.find(".")
    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal + 1]
        if value_after_decimal == "0":
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal + 2:]
    return new_tick_format


def class_distribution(load_data, col, label, data_dir):
    """Save samples distribution according to each class.

    Args:
        load_data: Contains input and output data features.
        col: String of the resulting element.
        data_dir: String of the data directory path.
        
    Returns:
        None
    """
    print(load_data[col].value_counts())
    # Plot the distribution using bar plot
    plt.figure(figsize=(12, 8))
    load_data[col].value_counts().sort_values().plot(kind="bar")
    plt.title("Samples Distribution", size=18)
    plt.xlabel('Classes', size=18)
    plt.ylabel('Samples', size=18)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.tight_layout()
    plt.savefig(data_dir + "/barchart_class_distribution_" + str(label) + ".png")
    plt.close()
    # Plot the distribution using seaborn countplot
    plt.figure(figsize=(12, 8))
    sns.countplot(x=load_data[col].sort_values(), palette="viridis")
    plt.title("Samples Distribution", size=18)
    plt.xlabel('Classes', size=18)
    plt.ylabel('Samples', size=18)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.tight_layout()
    plt.savefig(data_dir + "/countplot_barchart_class_distribution_" + str(label) + ".png")
    plt.close()


def data_analysis(df, label, data_dir):
    """Save pearson correlation analysis of input and output features.

    Args:
        df: Contains input and output data features.
        label: String of the resulting element.
        data_dir: String of the data directory path.
        
      
    Returns:
        None
    """
    pearsonr_corr = df.corr(method="pearson")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pearsonr_corr, annot=True, cmap='PuBuGn')
    plt.title("Pearson Correlation Analysis of Input Features")
    plt.tight_layout()
    plt.savefig(data_dir + "/correlation_" + str(label) + ".png")
    plt.close()


def pca_analysis(X_train, y_train, data_dir, n_components=2):
    """Save PCA analysis.

    Args:
        X_train: Contain input data features.
        y_train: Contain labels.
        data_dir: String of the data directory path.
        n_components: Number of dimensions to visualize non-linear data
        
    Returns:
        None
    """
   
    # Reduce to n dimensions with PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_train)

    # Plot the first two principal components
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_train, cmap='viridis')
    plt.colorbar()
    plt.title('PCA of Latent Features', size=20)
    plt.xlabel('PCA 1', size=20)
    plt.ylabel('PCA 2', size=20)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(data_dir + "/pca_analysis.png")
    plt.close()


def tsne_analysis(X_train, y_train, data_dir, n_components=2):
    """Save t-SNE analysis.

    Args:
        X_train: Contain input data features.
        y_train: Contain labels.
        data_dir: String of the data directory path.
        n_components: Number of dimensions to visualize non-linear data
        
    Returns:
        None
    """
    # Reduce to n dimensions with t-SNE
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_result = tsne.fit_transform(X_train)
    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_train, cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE of Latent Features', size=20)
    plt.xlabel('t-SNE 1', size=20)
    plt.ylabel('t-SNE 2', size=20)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(data_dir + "/tsne_analysis.png")
    plt.close()



def loss_analysis(model_history, model_name, data_dir):
    """Save training and validation loss.

    Args:
        model_history: Training history of the trained model.
        model_name: String of the resulting model.
        data_dir: String of the data directory path.
        
      
    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    plt.plot(model_history.history['loss'], color="b", linewidth=4, linestyle="solid")
    plt.title('Training and Validation Loss Analysis of ' + model_name, size=20)
    plt.ylabel('Loss', size=20)
    plt.xlabel('Epochs', size=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.close()
    plt.figure(figsize=(12, 8))
    plt.plot(model_history.history['loss'], color="b", linewidth=4, linestyle="solid")
    plt.plot(model_history.history['val_loss'], color="orange", linewidth=4, linestyle="dashed")
    plt.title('Training and Validation Loss Analysis of ' + model_name, size=20)
    plt.ylabel('Loss', size=20)
    plt.xlabel('Epochs', size=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.legend(['Training Loss', 'Validation Loss'], loc='best', shadow=True, fontsize=17)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(data_dir + "/loss_" + str(model_name) + ".png")
    plt.close()


def training_validation_analysis(model_history, model_name, metric, data_dir):
    """Save training and validation MAE.

    Args:
        model_history: Training history of the trained model.
        model_name: String of the resulting model.
        metric: String representing performance metric
        data_dir: String of the data directory path.
        
      
    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    plt.plot(model_history.history[metric], color="b",  linewidth=4, linestyle="solid")
    plt.plot(model_history.history['val_'+metric], color="orange", linewidth=4, linestyle="dashed")
    plt.title('Training and Validation '+metric+' Analysis of '+ model_name, size=20)
    plt.ylabel(metric, size=20)
    plt.xlabel('Epochs', size=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.legend(['Training '+metric, 'Validation ' +metric], loc='best', shadow=True, fontsize=17)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(data_dir + "/" +metric+ "_" + str(model_name) + ".png")
    plt.close()


def mae_analysis(model_history, model_name, data_dir):
    """Save training and validation MAE.

    Args:
        model_history: Training history of the trained model.
        model_name: String of the resulting model.
        data_dir: String of the data directory path.
        
      
    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    plt.plot(model_history.history['mae'], color="b",  linewidth=4, linestyle="solid")
    plt.plot(model_history.history['val_mae'], color="orange", linewidth=4, linestyle="dashed")
    plt.title('Training and Validation MAE Analysis of '+ model_name, size=20)
    plt.ylabel('MAE', size=20)
    plt.xlabel('Epochs', size=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.legend(['Training MAE', 'Validation MAE'], loc='best', shadow=True, fontsize=17)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(data_dir + "/mae_" + str(model_name) + ".png")
    plt.close()


def mape_analysis(model_history, model_name, data_dir):
    """Save training and validation MAPE.

    Args:
        model_history: Training history of the trained model.
        model_name: String of the resulting model.
        data_dir: String of the data directory path.
        
      
    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    plt.plot(model_history.history['mape'], color="b",  linewidth=4, linestyle="solid")
    plt.plot(model_history.history['val_mape'], color="orange", linewidth=4, linestyle="dashed")
    plt.title('Training and Validation MAPE Analysis of '+ model_name, size=20)
    plt.ylabel('MAPE', size=20)
    plt.xlabel('Epochs', size=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.legend(['Training MAPE', 'Validation MAPE'], loc='best', shadow=True, fontsize=17)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(data_dir + "/mape_" + str(model_name) + ".png")
    plt.close()


def confusion_matrix_vis(confusion_matrix, m_label, data_dir):
    """Save training and validation MAPE.

    Args:
        confusion_matrix: Training history of the trained model.
        m_label: String of the resulting model.
        data_dir: String of the data directory path.
        
    Returns:
        None
    """
    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 15})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.close()
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 15})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.title('Confusion Matrix using ' +(m_label), size=20)
    plt.savefig(data_dir+'/seaborn_confusion_matrix_'+m_label+'.png')
    plt.close()
    # Plot using scikit-learn
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    # Increase font size
    for labels in ax.texts:  # ax.texts contains the text elements in the plot
        labels.set_fontsize(15)
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.title('Confusion Matrix using ' +(m_label), size=20)
    plt.savefig(data_dir+'/sklearn_confusion_matrix_'+m_label+'.png')
    plt.close()


def plot_roc_curves(classifiers, data_dir):
    """Plots ROC curves for multiple classifiers with different line styles and colors.
    
    Args:
        classifiers: list of dictionaries
            Each dictionary should have the following keys:
            - 'model_name': Name of the model (str)
            - 'fpr': False Positive Rate (array-like)
            - 'tpr': True Positive Rate (array-like)
            - 'auc': Area Under the Curve (float)
    """
    plt.figure(figsize=(10, 8))
    # Define different line styles and colors
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    # Flatten the list of classifiers
    print(classifiers)
    flattened_classifiers = [dict(classifier) for classifier in classifiers]
    print(flattened_classifiers)
    # Plot ROC curve for each classifier
    for i, classifier in enumerate(flattened_classifiers):
        model_name = classifier['model_name']
        fpr = classifier['fpr']
        tpr = classifier['tpr']
        auc_score = classifier['auc']
        # Use different line styles and colors for each classifier
        plt.plot(fpr, tpr, linestyle=linestyles[i % len(linestyles)], color=colors[i % len(colors)],
                 lw=2, label=f'{model_name} (AUC = {auc_score:.2f})')
    
    # Plot the diagonal line (no skill)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic (ROC) Curves', size=20)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(False)
    plt.savefig(data_dir+'/roc_curve.png')
    plt.close()


def plot_bar_chart(evaluation_dic, data_dir):
    """Plots a bar chart for performance metrics of classifiers.
    
    Args:
        evaluation_dic: list of lists of tuples
            Each list contains tuples representing key-value pairs:
            - 'model_name': Name of the model (str)
            - 'accuracy': Accuracy of the model (float)
            - 'precision': Precision of the model (float)
            - 'recall': Recall of the model (float)
            - 'f1': F1 score of the model (float)
            - 'mcc': Matthews Correlation Coefficient of the model (float)
    """
    # Convert list of tuples to a dictionary
    flattened_classifiers = [dict(classifier) for classifier in evaluation_dic]
    
    # Extract metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
    model_names = [classifier['model_name'] for classifier in flattened_classifiers]
    
    # Initialize plot
    bar_width = 0.15
    index = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot bars for each classifier
    for i, classifier in enumerate(flattened_classifiers):
        values = [classifier[metric] for metric in metrics]
        ax.bar(index + i * bar_width, values, bar_width, label=classifier['model_name'])
    
    # Set plot properties
    ax.set_xlabel('Metrics', size=16)
    ax.set_ylabel('Scores', size=16)
    ax.set_title('Classifier Performance Comparison', size=20)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    ax.set_xticks(index + bar_width * (len(flattened_classifiers) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right', fontsize=15)
    plt.ylim([0, 1.05])
    plt.grid(axis='y', linestyle='--')
    plt.savefig(data_dir+ '/bar_chart_analysis.png')
    plt.close()