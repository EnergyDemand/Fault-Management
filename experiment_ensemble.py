"""Experiment for building ensemble and individual classifiers."""

from src.utils.arg_parser import *
from src.utils.directory_management import *
from src.data_processing.data_processing import *
from src.data_processing.data_resampling import *
from src.utils.save_utils import class_distribution, data_analysis
from src.models.ensemble_model_new import *
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
    print(train_test_ratio,voting_method, resample_method, n_estimators)

    # Setting file structure for accessing data and storing results
    res_dir = ''
    data_file_dir = ''
    data_path = get_data_dir()
    print(data_path)

    # Load data using pandas dataframe 
    target_label = 'f_nv'
    df_data = pd.read_csv(data_path+'/combined_data_part1.csv')
    print(df_data[target_label].unique())

    # Initial pre-processing
    # Replace all of classes to one class belonging to fault types
    if class_replace_flag:
        unique_classes = df_data[target_label].unique()
        classes_to_replace = unique_classes[1:]
        df_data.loc[df_data[target_label].isin(classes_to_replace), target_label] = 1
    df_data[target_label] = df_data[target_label].astype('int64')
    print(df_data)

    # X_train, X_test, y_train, y_test = new_get_data(df_data.copy(), target_label, data_split_ratio=0.4)

    X_train, X_test, y_train, y_test = new_get_data(df_data.copy(), target_label, train_test_ratio)

    X_test_copy = X_test.copy()
    X_test_copy[target_label] = y_test
    class_distribution(X_test_copy, target_label, 'testdata', data_file_dir)

    # Normalization using min-max 
    X_train_normalized = normalize_data(X_train)
    X_test_normalized = normalize_data(X_test)

    # Shape 
    print(type(X_train_normalized), X_train_normalized.shape)
    print(type(y_train), y_train.shape)
    print(type(X_test_normalized), X_test_normalized.shape)
    print(type(y_test), y_test.shape)

    if resampling_flag:
        # Get resulting directory path
        res_dir = get_result_dir('Ensemble Experiment', resample_method)
        # data_file_dir = get_file_dir(res_dir, voting_method)
        # model_short_names = "-".join(included_models)
        data_file_dir = get_file_dir(res_dir, voting_method, n_estimators)
        X_train_normalized[target_label] = y_train
        # Class distribution before smote based re-sampling
        class_distribution(X_train_normalized, target_label, 'without-resampled', data_file_dir)
        if resample_method=='SMOTE':
            X_train_normalized = smote_resampling(X_train_normalized, target_label)
        elif resample_method=='ADASYN':
            X_train_normalized = adasyn_resampling(X_train_normalized, target_label)
        # Class distribution after smote based re-sampling
        class_distribution(X_train_normalized, target_label, resample_method, data_file_dir)
        y_train = X_train_normalized[target_label]
        X_train_normalized.drop(target_label, axis=1, inplace=True)
        # Shape
        print(type(X_train_normalized), X_train_normalized.shape)
        print(type(y_train), y_train.shape)

    else:
        res_dir = get_result_dir('Ensemble Experiment', 'Imbalanced')
        # data_file_dir = get_file_dir(res_dir, voting_method)
        # model_short_names = "-".join(included_models)
        data_file_dir = get_file_dir(res_dir, voting_method, n_estimators)
        df_train = X_train_normalized.copy()
        df_train[target_label] = y_train
        # Class distribution after smote based re-sampling 
        class_distribution(df_train, target_label, 'imbalanced', data_file_dir)
    
    # Correlation analysis of input features with respect to the output feature
    data_analysis(df_data, target_label, res_dir)

    # Train ensemble and inidiviudal learners
    print('Training...')
    individual_clf = individual_classifiers(X_train_normalized, y_train, n_estimators)
    ensemble_clf = ensemble_classifier(X_train_normalized, y_train, n_estimators, voting_method)

    # Evaluate ensmeble and individual learners
    print('Inference...')
    if class_replace_flag:
        binary_classification_inference(X_test_normalized, y_test, ensemble_clf, individual_clf, data_file_dir)
    else:
        multi_classification_inference(X_test_normalized, y_test, ensemble_clf, individual_clf, data_file_dir)

    print("X_train_normalized columns:", X_train_normalized.columns)
    print("X_test_normalized columns:", X_test_normalized.columns)

    # Interpret predictions made by ensmeble and individual learners
    print('Interpretation...')
    ensemble_classifier_interpretation(ensemble_clf, X_train_normalized, X_test_normalized, feature_names, voting_method, data_file_dir)

if __name__ == '__main__':
    # main()
    main(['-vm', 'hard'])
    # main(['-rflag', '-rm', 'SMOTE'])
    # main(['-rflag', '-rm', 'SMOTE', '-vm', 'hard'])
    # main(['-rflag', '-rm', 'ADASYN'])
    # main(['-rflag', '-rm', 'ADASYN', '-vm', 'hard'])

    # main(['-ne', '10'])
    # main(['-vm', 'hard', '-ne', '25'])
    # main(['-rflag', '-rm', 'SMOTE', '-ne', '10'])
    # main(['-rflag', '-rm', 'SMOTE', '-vm', 'hard', '-ne', '10'])
    # main(['-rflag', '-rm', 'ADASYN', '-ne', '25'])
    # main(['-rflag', '-rm', 'ADASYN', '-vm', 'hard', '-ne', '25'])