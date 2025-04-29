from sklearn.metrics import accuracy_score, f1_score, classification_report

def calculate_metrics(y_true, y_pred, target_names=None, labels=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(
        y_true, y_pred, 
        target_names=target_names, 
        labels=labels if labels is not None else list(range(len(target_names))),
        zero_division=0
    )
    return {
        "accuracy": acc,
        "f1": f1,
        "classification_report": report
    }