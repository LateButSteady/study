#! usr/bin/env python
#-*- encoding: utf-8 -*-

import sys, os
import numpy as np
import pickle 
np.random.seed(2022)
sys.path.append('../')
import weightedSHAP
from matplotlib import pyplot as plt

# Load dataset
dir_path='./'

# problem='classification'
# dataset='fraud'
# ML_model='boosting'
# fname_dump='fraud_example.pickle'


problem='regression' 
dataset='diabetes'
ML_model='boosting'
fname_dump='diabetes.pickle'

os.system("pause")

(X_train, y_train), (X_val, y_val), (X_est, y_est), (X_test, y_test)=weightedSHAP.load_data(problem, dataset, dir_path)    

# train a baseline model
model_to_explain=weightedSHAP.create_model_to_explain(X_train, y_train, X_val, y_val, problem, ML_model)


if not os.path.exists(f'{dir_path}/{fname_dump}'):
    # Generate a conditional coalition function
    conditional_extension=weightedSHAP.generate_coalition_function(model_to_explain, X_train, X_est, problem, ML_model)
    
    # With the conditional coalition function, we compute attributions
    exp_dict=weightedSHAP.compute_attributions(problem, ML_model,
                                                 model_to_explain, conditional_extension,
                                                 X_train, y_train,
                                                 X_val, y_val, 
                                                 X_test, y_test)

    with open(f'{dir_path}/{fname_dump}', 'wb') as handle:
        pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{dir_path}/{fname_dump}', 'rb') as handle:
        exp_dict = pickle.load(handle)

exp_dict.keys()


y_test=np.array(exp_dict['true_list'])
pred_list=np.array(exp_dict['pred_list'])
print(f'MSE of the original model: {np.mean((y_test-pred_list)**2):.3f}') 


exp_dict['value_list'][0] # WeightedSHAP value of the first test sample


def find_optimal_list(cond_pred, pred_list, is_lower_better=True):
    diff_mat=np.mean(np.abs(cond_pred - pred_list.reshape(-1,1,1)), axis=-1)
    return np.argmin(diff_mat, axis=1)

cond_pred_keep_absolute=np.array(exp_dict['cond_pred_keep_absolute'])

# Find a optimal weight and construct WeightedSHAP
list_of_optimal_weights=find_optimal_list(cond_pred_keep_absolute, pred_list) 
N_test_sample=len(list_of_optimal_weights)

SHAP_condi_coal=cond_pred_keep_absolute[:,6,:] # SHAP
WeightedSHAP_condi_coal=cond_pred_keep_absolute[np.arange(N_test_sample), list_of_optimal_weights,:]
cond_pred_keep_absolute_short=np.array((SHAP_condi_coal, WeightedSHAP_condi_coal)).transpose((1,0,2))
recovery_curve_keep_absolute=np.mean(np.abs(cond_pred_keep_absolute_short - pred_list.reshape(-1,1,1)), axis=0)



plt.figure(figsize=(5,4))
n_features=len(recovery_curve_keep_absolute[0])
n_display_features=int(n_features*0.6)

plt.plot(recovery_curve_keep_absolute[0][max(1,int(n_features*0.075)):n_display_features],
         label='Shapley', color='green', linewidth=2, alpha=0.6)
plt.plot(recovery_curve_keep_absolute[1][max(1,int(n_features*0.075)):n_display_features],
         label='WeightedSHAP', color='red', linewidth=2, alpha=0.6)
plt.legend(fontsize=12)
xlabel_text='Number of features added' 
plt.title(f'Prediction recovery error curve\n Dataset: {dataset} \n the lower, the better', fontsize=15)
plt.xticks(np.arange(n_features)[max(1,int(n_features*0.075)):n_display_features][::n_display_features//6],
               np.arange(n_features)[max(1,int(n_features*0.075)):n_display_features][::n_display_features//6])
plt.xlabel(xlabel_text, fontsize=15)
plt.ylabel(r'$|f(x)-\mathbb{E}[f(X) \mid X_S = x_S]|$', fontsize=15)
plt.show()
os.system("pause")  