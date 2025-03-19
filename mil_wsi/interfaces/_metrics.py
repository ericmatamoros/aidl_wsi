import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(y_pred, y_true, num_classes):
    """
    Computes classification metrics including precision, recall, F1 score, 
    and the confusion matrix for binary and multi-class classification.

    Args:
        y_pred (array-like): Predicted class labels.
        y_true (array-like): Ground-truth class labels.
        num_classes (int): Number of classes in the classification task.

    Returns:
        dict: A dictionary containing the following metrics:
            - 'precision' (float): Precision score.
            - 'recall' (float): Recall score.
            - 'f1' (float): F1 score.
            - 'confusion_matrix' (pd.DataFrame): Confusion matrix as a DataFrame.
    """
    if num_classes == 2:
        # Calculate precision, recall, and F1 score
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        breakpoint()
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Convert confusion matrix to DataFrame
        cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    else:
        # Calculate precision, recall, and F1 score for each class
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Convert confusion matrix to DataFrame
        class_labels = sorted(set(y_true))  # Ensure correct ordering of classes
        cm_df = pd.DataFrame(cm, index=[f'Actual {cls}' for cls in class_labels], 
                                columns=[f'Predicted {cls}' for cls in class_labels])
    
    # Return results in a dictionary
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm_df
    }