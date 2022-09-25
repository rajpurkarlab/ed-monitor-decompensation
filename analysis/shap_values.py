import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

import shap
# load JS visualization code to notebook
shap.initjs()

def get_shap_analysis(final_lgbm_class, xtest, config, task, time, num_values=5):
    explainer = shap.TreeExplainer(final_lgbm_class)
    shap_values = explainer.shap_values(xtest)
    shap_absval_array = np.abs(np.array(shap_values)[1])
    print(shap_absval_array.shape) # 1 x num_samples x num_features

    num_features = len(config['features'])
    
    # fix the pd.factorize such that low acuity is more severe
    acuity = np.empty_like(xtest[:, 2])
    acuity = np.where(xtest[:, 2] == 0, 3, acuity)
    acuity = np.where(xtest[:, 2] == 1, 2, acuity)
    acuity = np.where(xtest[:, 2] == 2, 5, acuity)
    acuity = np.where(xtest[:, 2] == 3, 1, acuity)
    acuity = np.where(xtest[:, 2] == 4, 4, acuity)
    acuity = np.where(np.isnan(xtest[:, 2]), np.nan, acuity)
    
    xtest[:, 2] = acuity

    correlation = np.empty((num_features))

    for f in range(num_features):
        shap_forf = pd.Series(shap_values[1][:, f])
        feat_value = pd.Series(xtest[:, f])
        correlation[f] = shap_forf.corr(feat_value)

    # take mean over all samples
    df_shap = pd.DataFrame(shap_absval_array, columns=config['features'])
    mean_vals = np.abs(df_shap.values).mean(0)
    total = np.sum(mean_vals)

    pct_vals = mean_vals / total

    shap_importance = pd.DataFrame(list(zip(config['features'], pct_vals, correlation)),
                      columns=['col_name','feature_importance_vals', 'correlation'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                   ascending=False, inplace=True)
    shap_importance.round(decimals=4)
    print(shap_importance.head(num_values))

    shap.initjs()
    shap.summary_plot(shap_values[1][:, :], xtest, config['features'], plot_type="bar", max_display=8, show=False)
    plt.title(task[0].capitalize() + " @ " + time + ": " + config['name'])
    plt.show()

    shap.summary_plot(shap_values[1][:, :], xtest, config['features'], plot_type="violin", max_display=8, show=False)
    plt.title(task[0].capitalize() + " @ " + time + ": " + config['name'])
    plt.show()