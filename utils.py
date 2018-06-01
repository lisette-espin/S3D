import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def obtain_metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    auc_macro = roc_auc_score(y_true, y_pred)
    auc_micro = roc_auc_score(y_true, y_pred, 'micro')
    f1_binary = f1_score(y_true, y_pred, average='binary')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    d = {'accuracy': acc, 'auc_macro': auc_macro, 'auc_micro': auc_micro,
         'f1_binary':f1_binary, 'f1_macro': f1_macro, 'f1_micro': f1_micro}
    return pd.Series(d)
