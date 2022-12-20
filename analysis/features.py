import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, classification_report, roc_curve, auc,recall_score, precision_score, roc_auc_score, mean_squared_error
from tqdm import tqdm
from scipy.stats import ttest_ind

prefix_path = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed-monitor-decompensation-clean/"

def plot_roc_curve(models, xtests, ytest):
    plt.figure()
    t = []
    colors = ['crimson', 'deepskyblue', 'violet']
    for i in range(len(models)):
        model = models[i]
        xtest = xtests[i]
        preds = model.predict_proba(xtest)[:, 1]
        fpr, tpr, thresholds = roc_curve(ytest, preds)
        index = (np.abs(tpr - 0.85)).argmin()
        t.append(thresholds[index])
        plt.plot(fpr, tpr, color=colors[i])
                
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.show()
    return t

# model 1 is the best model, model 2 is triage only model
def get_feature_distributions(features, lgb_model1, lgb_model2, all_xtest1, xtest1, xtest2, ytest, xval1, xval2, yval, time, task, names):
    preds1 = lgb_model1.predict_proba(xtest1)[:, 1]
    preds2 = lgb_model2.predict_proba(xtest2)[:, 1]  
    
    val1 = lgb_model1.predict_proba(xval1)[:, 1]
    val2 = lgb_model2.predict_proba(xval2)[:, 1] 
    
    fpr1, tpr1, thresholds1 = roc_curve(yval, val1)
    th1 = thresholds1[(np.abs(tpr1 - 0.85)).argmin()]
    
    fpr2, tpr2, thresholds2 = roc_curve(yval, val2)
    th2 = thresholds2[(np.abs(tpr2 - 0.85)).argmin()]

    bin1 = np.where(preds1 > th1, 1, 0) # binarized predictions bucketed at thresholds
    bin2 = np.where(preds2 > th2, 1, 0)

    scores1 = np.where(bin1 == ytest, 1, 0) # samples where full model is correct
    scores2 = np.where(bin2 == ytest, 1, 0) # samples where partial model is correct
    
    a = np.where((scores1 == scores2) & (scores1 == 1), 1, 0) # both correct
    b = np.where((scores1 != scores2) & (scores1 == 1), 1, 0) # full model correct, partial incorrect
    c = np.where((scores1 != scores2) & (scores2 == 1), 1, 0) # full model incorrect, partial correct
    d = np.where((scores1 == scores2) & (scores1 == 0), 1, 0) # both incorrect 
    
    b_fp_tn = np.where((bin1 != bin2) & (bin1 == 0), b, 0) # subset of b where full model predicts negative
    b_fn_tp = np.where((bin1 != bin2) & (bin1 == 1), b, 0) #
    
    print(np.sum(c))
    print(np.sum(b))
    print(np.sum(b_fp_tn))
    print(np.sum(b_fn_tp))
    
    # realigns acuity scores such that lower acuity is more severe triage score
    acuity = np.empty_like(all_xtest1[:, 2])
    acuity = np.where(all_xtest1[:, 2] == 0, 3, acuity)
    acuity = np.where(all_xtest1[:, 2] == 1, 2, acuity)
    acuity = np.where(all_xtest1[:, 2] == 2, 5, acuity)
    acuity = np.where(all_xtest1[:, 2] == 3, 1, acuity)
    acuity = np.where(all_xtest1[:, 2] == 4, 4, acuity)
    acuity = np.where(np.isnan(all_xtest1[:, 2]), np.nan, acuity)
    
    all_xtest1[:, 2] = acuity

    # total number of each cross comparison condition
    print(np.sum(a), np.sum(b), np.sum(c), np.sum(d))
    
    df = pd.DataFrame(columns = ['Feature name', 'Cohort excluding',
                                 'Cohort reclassified GT Neg', 'p_value GT Neg', 
                                 'Cohort reclassified GT Pos', 'p_value GT Pos'])

    for i in tqdm(range(all_xtest1.shape[1])):
        name = features[i]
        feature_a = all_xtest1[:, i][a == 1]
        
        feature_b = all_xtest1[:, i][b == 1]
        feature_b_fp_tn = all_xtest1[:, i][b_fp_tn == 1]
        feature_b_fn_tp = all_xtest1[:, i][b_fn_tp == 1]
        
        feature_c = all_xtest1[:, i][c == 1]
        feature_d = all_xtest1[:, i][d == 1]
        feature_acd = all_xtest1[:, i][b != 1]
        feature_all = all_xtest1[:, i]
        
        p_value = ttest_ind(feature_b, feature_acd, equal_var=False, nan_policy='omit').pvalue
        p_value_fp_tn = ttest_ind(feature_b_fp_tn, feature_acd, equal_var=False, nan_policy='omit').pvalue
        p_value_fn_tp = ttest_ind(feature_b_fn_tp, feature_acd, equal_var=False, nan_policy='omit').pvalue
                        
        row = [name, str(round(np.nanmean(feature_acd), 3)) + ' (' + str(round(np.nanstd(feature_acd), 3)) + ')', 
               str(round(np.nanmean(feature_b_fp_tn), 3)) + ' (' + str(round(np.nanstd(feature_b_fp_tn), 3)) + ')', p_value_fp_tn,
               str(round(np.nanmean(feature_b_fn_tp), 3)) + ' (' + str(round(np.nanstd(feature_b_fn_tp), 3)) + ')', p_value_fn_tp]
        df.loc[len(df.index)] = row
    
    df['Cohort reclassified GT Neg'] = df['Cohort reclassified GT Neg'].astype(str)
    df['Cohort reclassified GT Pos'] = df['Cohort reclassified GT Pos'].astype(str)
    df['Cohort excluding'] = df['Cohort excluding'].astype(str)
    
    df = df.sort_values("p_value GT Neg", ascending=True)
    print(df)
    save_filename = prefix_path + time + "_" + task + "_GTNeg_ttested_feature_stats_results" + names[0] + "_" + names[1] + ".csv"
    df.to_csv(save_filename)
    
    df = df.sort_values("p_value GT Pos", ascending=True)
    print(df)
    save_filename = prefix_path + time + "_" + task + "_GTPos_ttested_feature_stats_results" + names[0] + "_" + names[1] + ".csv"
    df.to_csv(save_filename)
        
    