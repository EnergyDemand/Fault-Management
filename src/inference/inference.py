from src.evaluation.binary_class_evaluation import binary_class_evaluation
from src.evaluation.multi_class_evaluation import multi_class_evaluation
from src.interpretation.interpret_model import *
from sklearn.metrics import confusion_matrix
from src.utils.save_utils import confusion_matrix_vis, plot_roc_curves, plot_bar_chart
from src.models.model_utils import calculate_reconstruction_loss
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def binary_classification_inference(X_test_normalized, y_test, ensemble_clf, individual_clf, data_dir):
    """Evaluate the ensemble classifier and individual classifiers.

    Args:
        X_test_normalized (ndarray): Normalized input features used for testing.
        y_test (ndarray): True labels for the test data.
        ensemble_clf (classifier): Trained ensemble classifier used for prediction.
        individual_clf (list of classifiers): List of individual classifiers used in the ensemble.
        data_dir (str): Directory path to save any analysis results.
        
    Returns:
        None
    """
    # Evaluate ensemble learner
    y_ensemble_pred = ensemble_clf.predict(X_test_normalized)
    c_matrix = confusion_matrix(y_test, y_ensemble_pred)
    print(c_matrix)
    confusion_matrix_vis(c_matrix, 'Proposed Model', data_dir)
    evaluation_list, auc_dic = binary_class_evaluation(y_test, y_ensemble_pred, 'Proposed Model', data_dir)
    # Define dictionary to store evaluation results
    evalution_res_dic = []
    evalution_res_dic.append(evaluation_list)
    # Define dictionary to store resulting data for ROC curve analysis
    roc_res_dic = []
    roc_res_dic.append(auc_dic)
    # Traverse the trained_models dictionary
    for name, model in individual_clf.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_normalized)
        c_matrix = confusion_matrix(y_test, y_pred)
        print(c_matrix)
        confusion_matrix_vis(c_matrix, name, data_dir)
        eval_list, roc_auc = binary_class_evaluation(y_test, y_pred, name, data_dir)
        evalution_res_dic.append(eval_list)
        roc_res_dic.append(roc_auc)
    # ROC curve analysis
    plot_roc_curves(roc_res_dic, data_dir)
    # Barchart analysis 
    plot_bar_chart(evalution_res_dic, data_dir)


def multi_classification_inference(X_test_normalized, y_test, ensemble_clf, individual_clf, data_dir):
    """Evaluate the ensemble classifier and individual classifiers.

    Args:
        X_test_normalized (ndarray): Normalized input features used for testing.
        y_test (ndarray): True labels for the test data.
        ensemble_clf (classifier): Trained ensemble classifier used for prediction.
        individual_clf (list of classifiers): List of individual classifiers used in the ensemble.
        data_dir (str): Directory path to save any analysis results.
        
    Returns:
        None
    """
    # Evaluate ensemble learner
    y_ensemble_pred = ensemble_clf.predict(X_test_normalized)
    # Ensure the predictions are 1D arrays
    # print(str(y_ensemble_pred.ndim))
    # if y_ensemble_pred.ndim > 1:
    #     y_ensemble_pred = y_ensemble_pred.ravel()
    c_matrix = confusion_matrix(y_test, y_ensemble_pred)
    print(c_matrix)
    confusion_matrix_vis(c_matrix, 'Proposed Model', data_dir)
    evaluation_list = multi_class_evaluation(y_test, y_ensemble_pred, 'Proposed Model', data_dir)
    # Define dictionary to store evaluation results
    evalution_res_dic = []
    evalution_res_dic.append(evaluation_list)
    # Define dictionary to store resulting data for ROC curve analysis
    # roc_res_dic = []
    # roc_res_dic.append(auc_dic)
    # Traverse the trained_models dictionary
    if individual_clf:
        for name, model in individual_clf.items():
            print(f"Evaluating {name}...")
            y_pred = model.predict(X_test_normalized)
            c_matrix = confusion_matrix(y_test, y_pred)
            print(f"{name} Confusion Matrix:\n", c_matrix)
            confusion_matrix_vis(c_matrix, name, data_dir)
            eval_list = multi_class_evaluation(y_test, y_pred, name, data_dir)
            evalution_res_dic.append(eval_list)
        #roc_res_dic.append(roc_auc)
    # ROC curve analysis
    #plot_roc_curves(roc_res_dic, data_dir)
    # Barchart analysis 
    plot_bar_chart(evalution_res_dic, data_dir)


def unsupervised_autoencoder_inference(autoencoder, X_test, y_test, lower_threshold, upper_threshold, lower_cs, upper_cs, data_dir):
    # Reconstruct test samples using trained autoencoder
    reconstrcuted_samples = autoencoder.predict(X_test)
    # Calculate reconstructed loss
    r_loss = calculate_reconstruction_loss(X_test, reconstrcuted_samples)
    r_cosine_similarity = []
    # Reshape samples to 2D arrays (each is a single row)
    X_test_copy = X_test.copy()
    X_test_copy = X_test_copy.to_numpy()
    reconstructed_sample_copy = reconstrcuted_samples.copy()
    # Calculate cosine similarity for each sample in the training data
    for i in range(len(X_test)):
        original_sample = X_test_copy[i].reshape(1, -1)
        r_sample = reconstructed_sample_copy[i].reshape(1, -1)
        try:
            c_similarlity = cosine_similarity(original_sample, r_sample)
            r_cosine_similarity.append(c_similarlity[0,0]) 
        except ZeroDivisionError:
            r_cosine_similarity.append(0.0)  # Handle division by zero by setting NaN
        # Determine the threshold based on the SSIM values from the training data
   
    r_loss_copy = r_loss.tolist()
    df_loss = pd.DataFrame()
    df_loss['tested_r_loss'] = r_loss_copy
    df_loss['tested_cosine_similarity'] = r_cosine_similarity
    df_loss.to_csv(data_dir+'/tested_reconstruction_metrics.csv', index = False)
    # Compare the test reconstruction loss with the threshold
    fault_samples = (r_loss < lower_threshold) | (r_loss > upper_threshold)
    num_faults = np.sum(fault_samples)
    print(f"Number of faults detected in X_test: {num_faults} out of {len(X_test)} samples")
    # Append the test reconstruction loss to the X_test dataframe
    X_test['reconstruction_loss'] = r_loss
    X_test['tested_cosine_similarity'] = r_cosine_similarity
    # Initialize a numpy array to store the anomaly detection results
    # Use a vectorized operation to create the anomaly flags
    #fault_flags = (X_test['reconstruction_loss'].values > threshold).astype(int)
    fault_rl_flags = np.zeros(len(X_test), dtype=int)
    fault_cs_flags = np.zeros(len(X_test), dtype=int)
    fault_hybrid_flags = np.zeros(len(X_test), dtype=int)
    # Traverse the DataFrame and compare each reconstruction loss with the threshold
    for i in range(len(X_test)):
        r_loss = X_test['reconstruction_loss'].iloc[i]
        if r_loss < lower_threshold or r_loss > upper_threshold:
            fault_rl_flags[i] = 1  # Mark as fault
        else:
            fault_rl_flags[i] = 0  # Mark as normal
    # Traverse the DataFrame and compare each cosine similarity with the lower and upper thresholds
    for i in range(len(X_test)):
        r_cosine_sim = X_test['tested_cosine_similarity'].iloc[i]
        if r_cosine_sim < lower_cs or r_cosine_sim > upper_cs:
            fault_cs_flags[i] = 1  # Mark as fault
        else:
            fault_cs_flags[i] = 0  # Mark as normal
    # Traverse the DataFrame and compare reconstruction loss and cosine similarity
    for i in range(len(X_test)):
        r_loss = X_test['reconstruction_loss'].iloc[i]
        r_cosine_sim = X_test['tested_cosine_similarity'].iloc[i]
        if (r_loss < lower_threshold or r_loss > upper_threshold ) and (r_cosine_sim < lower_cs or r_cosine_sim > upper_cs):
            fault_hybrid_flags[i] = 1  # Mark as fault
        else:
            fault_hybrid_flags[i] = 0  # Mark as normal
    # Evaluation
    # Define dictionary to store resulting data for ROC curve analysis
    roc_res_dic = []
    # Define dictionary to store evaluation results
    evalution_res_dic = []
    for i in range(3):
        label = ''
        fault_flags = None
        if i == 0:
            label = 'IQR-Reconstruction Loss'
            fault_flags = fault_rl_flags
        elif i == 1:
            label = 'IQR-Cosine Similarity'
            fault_flags = fault_cs_flags
        elif i == 2:
            label = 'IQR-Hybrid'
            fault_flags = fault_hybrid_flags
        c_matrix = confusion_matrix(y_test, fault_flags)
        print(c_matrix)
        confusion_matrix_vis(c_matrix, label, data_dir)
        evaluation_list, auc_dic = binary_class_evaluation(y_test, fault_flags, label, data_dir)
        evalution_res_dic.append(evaluation_list)
        roc_res_dic.append(auc_dic)
    # ROC curve analysis
    plot_roc_curves(roc_res_dic, data_dir)
    # Barchart analysis 
    plot_bar_chart(evalution_res_dic, data_dir)
    


