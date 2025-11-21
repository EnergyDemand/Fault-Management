"""Experiment for building ensemble and individual classifiers."""

from src.utils.arg_parser import *
from src.utils.directory_management import *
from src.data_processing.data_processing import *
from src.data_processing.data_resampling import *
from src.utils.save_utils import class_distribution, data_analysis
from src.models.ensemble_model import *
from src.inference.inference import binary_classification_inference, multi_classification_inference
from src.interpretation.interpret_model import *
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main(args=None):
    """Main function to set up the experiment."""
    arg_parser = get_parser(args)
    n_estimators = arg_parser.n_estimators
    resampling_flag = arg_parser.resampling_flag
    resample_method = arg_parser.resampling_method.name
    train_test_ratio = arg_parser.test_ratio
    voting_method = arg_parser.voting_method
    class_replace_flag = arg_parser.class_replace_flag
    print(train_test_ratio, voting_method, resample_method, n_estimators)
    # Setting file structure for accessing data and storing results
    res_dir = ''
    data_file_dir = ''
    data_path = get_data_dir()
    print(data_path)
    # exit()
    # Load data using pandas dataframe
    target_label = 'f_nv'
    df_data = pd.read_csv(data_path + '/combined_data_part1.csv')
    print(df_data[target_label].unique())
    # Initial pre-processing
    # Replace all of classes to one class belonging to fault types
    if class_replace_flag:
        unique_classes = df_data[target_label].unique()
        classes_to_replace = unique_classes[1:]
        df_data.loc[df_data[target_label].isin(classes_to_replace), target_label] = 1
    df_data[target_label] = df_data[target_label].astype('int64')
    print(df_data)

    # Data Spliting into training and testing samples
    X_train, X_test, y_train, y_test, feature_names = get_data(df_data.copy(), target_label, train_test_ratio)

    # Normalization using min-max
    X_train_normalized = normalize_data(X_train)
    X_test_normalized = normalize_data(X_test)
    # Shape
    print(type(X_train_normalized), X_train_normalized.shape)
    print(type(y_train), y_train.shape)
    print(type(X_test_normalized), X_test_normalized.shape)
    print(type(y_test), y_test.shape)

    # def plot_tsne(X, y, sample_type, title):
    #     tsne = TSNE(n_components=2, random_state=42)
    #     X_tsne = tsne.fit_transform(X)
    #
    #     plt.figure(figsize=(10, 7))
    #     for sample, color in zip(['original', 'synthetic'], ['blue', 'red']):
    #         indices = (sample_type == sample)
    #         plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=f'{sample} samples', alpha=0.5, color=color)
    #     plt.title(title)
    #     plt.legend()
    #     plt.show()

    # Define t-SNE plotting function for a single class
    def plot_tsne_class(X, y, sample_type, class_label, title):
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)

        plt.figure(figsize=(10, 7))
        for sample, color in zip(['original', 'synthetic'], ['blue', 'red']):
            indices = (sample_type == sample) & (y == class_label)
            plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=f'{sample} samples', alpha=0.5, color=color)
        plt.title(f"{title} - Class {class_label}")
        plt.legend()
        plt.show()

    if resampling_flag:
        # Get resulting directory path
        res_dir = get_result_dir('Ensemble Experiment', resample_method)
        data_file_dir = get_file_dir(res_dir, voting_method)

        # Attach target label to the training set
        X_train_normalized[target_label] = y_train

        # Class distribution before SMOTE/ADASYN re-sampling
        class_distribution(X_train_normalized, target_label, 'without-resampled', data_file_dir)

        # Apply resampling method
        if resample_method == 'SMOTE':
            X_train_normalized = smote_resampling(X_train_normalized, target_label)
        elif resample_method == 'ADASYN':
            X_train_normalized = adasyn_resampling(X_train_normalized, target_label)

        # Update y_resampled and sample_type after resampling
        y_resampled = X_train_normalized[target_label]
        sample_type = X_train_normalized['sample_type']

        # Drop the target and sample_type columns from features for t-SNE
        X_resampled = X_train_normalized.drop([target_label, 'sample_type'], axis=1)

        # # t-SNE plot for visual evaluation of synthetic sample quality
        # plot_tsne(X_resampled, y_resampled, sample_type, f't-SNE Plot for {resample_method} Resampling')

        # t-SNE plot for each class
        unique_classes = sorted(y_resampled.unique())
        for class_label in unique_classes:
            plot_tsne_class(X_resampled, y_resampled, sample_type, class_label, f't-SNE Plot for {resample_method}')

    else:
        res_dir = get_result_dir('Ensemble Experiment', 'Imbalanced')
        data_file_dir = get_file_dir(res_dir, voting_method)
        df_train = X_train_normalized.copy()
        df_train[target_label] = y_train
        # Class distribution after smote based re-sampling
        class_distribution(df_train, target_label, 'imbalanced', data_file_dir)

    # Correlation analysis of input features with respect to the output feature
    data_analysis(df_data, target_label, res_dir)
    # Train ensemble and inidiviudal learners
    print('Training...')
    ensemble_clf = ensemble_classifier(X_train_normalized, y_train, n_estimators, voting_method)
    individual_clf = individual_classifiers(X_train_normalized, y_train, n_estimators)
    # Evaluate ensmeble and individual learners
    print('Inference...')
    if class_replace_flag:
        binary_classification_inference(X_test_normalized, y_test, ensemble_clf, individual_clf, data_file_dir)
    else:
        multi_classification_inference(X_test_normalized, y_test, ensemble_clf, individual_clf, data_file_dir)

    print("X_train_normalized columns:", X_train_normalized.columns)
    print("X_test_normalized columns:", X_test_normalized.columns)

    # Interpret predictions made by ensemble and individual learners
    print('Interpretation...')
    ensemble_classifier_interpretation(ensemble_clf, X_train_normalized, X_test_normalized, feature_names,
                                       voting_method, data_file_dir)


if __name__ == '__main__':
    # main()
    # main(['-vm', 'hard'])
    main(['-rflag', '-rm', 'SMOTE'])