from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, classification_report, roc_curve, auc, recall_score, precision_score, roc_auc_score, mean_squared_error


def get_confusion_matrix(op_thresholds, model, xtest, ytest, xval, yval):
    
    preds = model.predict_proba(xtest)[:, 1]
    val_preds = model.predict_proba(xval)[:, 1]
    
    for opt in op_thresholds:
        print(f"Calculating for {opt} threshold sensitivity on val set")
        fpr, tpr, thresholds = roc_curve(yval, val_preds)
        th = thresholds[(np.abs(tpr - opt)).argmin()]
                
        # get confusion matrix on test dataset sample
        preds_th = np.where(preds > th, 1, 0)
        tn, fp, fn, tp = confusion_matrix(ytest, preds_th).ravel()
        
        # get 95% confidence intervals 
        print(f"True positives ---- {tp}")
        print(f"False positives ---- {fp}")
        print(f"False negatives ---- {fn}")
        print(f"True negatives ---- {tn}")

def get_test_characteristics(op_thresholds, model, xtest, ytest, xval, yval):
    
    preds = model.predict_proba(xtest)[:, 1]
    val_preds = model.predict_proba(xval)[:, 1]
    
    for opt in op_thresholds:
        print(f"Calculating for {opt} threshold sensitivity on val set")
        fpr, tpr, thresholds = roc_curve(yval, val_preds)
        th = thresholds[(np.abs(tpr - opt)).argmin()]
        
        sensitivities, specificities, npvs, ppvs = [], [], [], []
        
        # bootstrap over test dataset
        for i in range(10000):
            bootstrap_indices = np.random.choice(range(len(ytest)), size=len(ytest), replace=True)
        
            bs_xtest = [xtest[i] for i in bootstrap_indices]
            bs_preds = model.predict_proba(bs_xtest)[:, 1]
            bs_preds_th = np.where(bs_preds > th, 1, 0)
            bs_labels = [ytest[i] for i in bootstrap_indices]
            
            tn, fp, fn, tp = confusion_matrix(bs_labels, bs_preds_th).ravel()
            num_samples = len(bs_labels)
            
            sensitivities.append(tp / (tp + fn))
            specificities.append(tn / (tn + fp))
            npvs.append(tn / (tn + fn))
            ppvs.append(tp / (tp + fp))
            
        # get score on test dataset sample
        preds_th = np.where(preds > th, 1, 0)
        tn, fp, fn, tp = confusion_matrix(ytest, preds_th).ravel()
        sens_sample = round(tp / (tp + fn), 3)
        spec_sample = round(tn / (tn + fp), 3)
        npv_sample = round(tn / (tn + fn), 3)
        ppv_sample = round(tp / (tp + fp), 3)
        
        # get 95% confidence intervals 
        print(f"Sensitivity ---- {sens_sample} ({np.round(np.percentile(sensitivities, [2.5]), 3)}, {np.round(np.percentile(sensitivities, [97.5]), 3)})")
        print(f"Specificity ---- {spec_sample} ({np.round(np.percentile(specificities, [2.5]), 3)}, {np.round(np.percentile(specificities, [97.5]), 3)})")
        print(f"NPV ---- {npv_sample} ({np.round(np.percentile(npvs, [2.5]), 3)}, {np.round(np.percentile(npvs, [97.5]), 3)})")
        print(f"PPV ---- {ppv_sample} ({np.round(np.percentile(ppvs, [2.5]), 3)}, {np.round(np.percentile(ppvs, [97.5]), 3)})")

