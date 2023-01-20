import numpy as np
from sklearn.metrics import roc_curve

def mews_subgroup(mews_labels, xval, yval, xtest, lgbm):

    preds = lgbm.predict_proba(xtest)[:, 1]
    pred_val = lgbm.predict_proba(xval)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(yval, pred_val)
    th = thresholds[(np.abs(tpr - 0.85)).argmin()]

    binned_pred = np.where(preds > th, 1, 0) # binarized predictions bucketed at thresholds
    
    # get the indices of the patients with mews score >= cutoff and mews score < cutoff
    # excludes values where the mews score is NaN
    mews_pos_indices = np.where(mews_labels == 1)
    mews_neg_indices = np.where(mews_labels == 0)

    # get the number of patients with mews score >= cutoff 
    mews_pos_pred_pos = np.sum(binned_pred[mews_pos_indices] == 1)
    mews_pos_pred_neg = np.sum(binned_pred[mews_pos_indices] == 0)

    # get the number of patients with mews score < cutoff
    mews_neg_pred_pos = np.sum(binned_pred[mews_neg_indices] == 1)
    mews_neg_pred_neg = np.sum(binned_pred[mews_neg_indices] == 0)

    print("MEWS Score >= cutoff: \n Predicted Positive: {} \t Predicted Negative: {} \t Total: {}".format(mews_pos_pred_pos, mews_pos_pred_neg, mews_pos_pred_pos + mews_pos_pred_neg))
    print("MEWS Score < cutoff: \n Predicted Positive: {} \t Predicted Negative: {} \t Total: {}".format(mews_neg_pred_pos, mews_neg_pred_neg, mews_neg_pred_pos + mews_neg_pred_neg))

