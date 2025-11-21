from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve,auc
import csv


def binary_class_evaluation(y_test, y_pred, model_name, csv_file_path):
    """Classification analysis in terms of accuracy, precision, etc.

    Args:
        y_test: array-like, shape (n_samples,)True labels of the test set.
        y_pred: array-like, shape (n_samples,)Predicted labels from the model.
        model_name: str Name or label of the model being evaluated.
        csv_file_path: str Path to the CSV file where metrics will be appended.
        
    Returns:
        fpr: False positive rate.
        tpr: True positive rate.
        roc_auc: Area under curve.
    """
    # Calculate various classification metrics
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred), 3)
    recall = round(recall_score(y_test, y_pred), 3)
    f_score = round(f1_score(y_test, y_pred), 3)
    mcc = round(matthews_corrcoef(y_test, y_pred), 3)
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = round(auc(fpr, tpr), 3)
    # Append results to CSV file
    with open(csv_file_path+'/evaluation_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file is empty
        if file.tell() == 0:
            writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC'])
        # Write the metrics
        writer.writerow([model_name, accuracy, precision, recall, f_score, mcc])
    # Prepare evaluation list to return
    evaluation_list = [
        ('model_name', model_name),
        ('Accuracy', accuracy),
        ('Precision', precision),
        ('Recall', recall),
        ('F1', f_score),
        ('MCC', mcc),
        ('AUC', roc_auc)
    ]
    # Prepare results into dictionary list for AUC analysis
    roc_dic = [
        ('model_name', model_name),
        ('fpr', fpr),
        ('tpr', tpr),
        ('auc', roc_auc)
    ]
    # Return the resulting values
    return evaluation_list, roc_dic


def set_auc_dic(fpr, tpr, roc_auc):
    """Store results into dictionary list for AUC analysis.

    Args:
        fpr: False positive rate.
        tpr: True positive rate.
        roc_auc: Area under curve.
        
    Returns:
        roc_dic: Return resulting dictionary list.
    """
    roc_dic = [
        ('FPR', fpr),
        ('TPR', tpr),
        ('AUC', roc_auc)
    ]
    
    return roc_dic