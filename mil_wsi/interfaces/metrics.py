import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(y_pred, y_true):
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert confusion matrix to DataFrame
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    
    # Return results in a dictionary
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm_df
    }