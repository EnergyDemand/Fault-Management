from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve,auc
import csv


def multi_class_evaluation(y_test, y_pred, model_name, csv_file_path):
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
    precision = round(precision_score(y_test, y_pred, average='weighted'), 3)
    recall = round(recall_score(y_test, y_pred, average='weighted'), 3)
    f_score = round(f1_score(y_test, y_pred, average='weighted'), 3)
    mcc = round(matthews_corrcoef(y_test, y_pred), 3)
    # Calculate ROC curve and AUC
    # fpr, tpr, _ = roc_curve(y_test, y_pred)
    # roc_auc = round(auc(fpr, tpr), 3)
    # Print the results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f_score)
    print("Matthews Correlation Coefficient (MCC):", mcc)
    #print("AUC:", mcc)
    # Save to CSV file
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
        ('MCC', mcc)
    ]
    # Return the resulting values
    return evaluation_list



    