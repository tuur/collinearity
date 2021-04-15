

import os, shutil, sys, time
import logging
import pickle
import numpy as np
import itertools, argparse, os
from numpy.core._multiarray_umath import positive
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import sklearn
from lib.preproc_data.datasets import find_X_pairs, describe_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from lib.utils import plot_difference_in_predicted_probs, assess_collinearity, get_vif_stats, Rval
from lib.PCALogit import PCALogit
from lib.constrainedlogit import ConstrainedLogisticRegression
from lib.evaluation import Evaluator, write_table_line
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, log_loss
import random
import dill
import pandas as pd
from copy import deepcopy
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import scipy.linalg as la


class Experiment:


    def __init__(self, data, out_folder, name="EXP"):
        self.data = data
        self.name = name
        self.exp_dir = out_folder + '/' + self.name + '/'
        self.log_file = self.exp_dir + 'log.txt'
        if os.path.exists(self.exp_dir):
            shutil.rmtree(self.exp_dir)
        os.makedirs(self.exp_dir)
        self.logger = self.set_logger()
        self.log('')
        self.log(name, 'logging to', self.log_file)

    def set_logger(self):
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s : %(message)s')
        fh = logging.FileHandler(self.log_file)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(ch)
        logger.addHandler(fh)
        return logger

    def log(self, *args):
        self.logger.info(' '.join([str(a) for a in args]))

    def conduct(self):
        self.log("test line")
        pass

    def save_object(self, to_be_saved_object, file_id):
        self.log('saving to file:',file_id)
        with open(self.exp_dir + '/' + file_id + '.p', 'wb') as f:
            pickle.dump(to_be_saved_object, f)

def encode_variables(df, yaml):
    x_names = []

    # add continuous variables as is
    for pred_name in yaml['preds_continuous']:
        x_name='X_' + pred_name
        df[x_name] = df[pred_name]
        x_names.append(x_name)

    # create one-hot / boolean / binary / dichotomised encodings for categorical variables
    for pred_name in yaml['preds_boolean']:
        for value in yaml['preds_boolean'][pred_name][1:]: #ignore the first value in the list : sets it as reference value
            x_name = 'X_'+pred_name+'_'+str(value)
            df[x_name] = df.apply(lambda row: (float(row[pred_name]==value) if not pd.isna(row[pred_name]) else row[pred_name]), axis = 1)
            x_names.append(x_name)

    y_name = 'y_'+yaml['outcome_var_name']
    df[y_name] = df.apply(lambda row: (float(row[yaml['outcome_var_name']] in yaml['outcome_positive_values']) if not pd.isna(row[yaml['outcome_var_name']]) else row[yaml['outcome_var_name']]), axis = 1)

    return df, x_names, y_name

def select_train_test_patients(df, yaml, y_name, x_names,bs=0):
    df_unlabeled_train_indices, df_train_indices, df_unlabeled_test_indices, df_test_indices = [], [], [], []

    # select n data rows for which the constraints are met for the test set
    for index, row in df.iterrows():
        add=True
        for constraint,values in yaml['data_val_included'].items():
            if not row[constraint] in values:
                add=False
        # If the constraints match, the max test size is not exceeded
        if add and len(df_test_indices) < yaml['data_val_max_size']:
            if pd.isna(row[y_name]): #or pd.isna(row[x_names]).any():#
                df_unlabeled_test_indices.append(index)
            else:
                df_test_indices.append(index)

    # select n data rows for which the constraints are met for the train set
    for index, row in df.iterrows():
        add=True
        for constraint,values in yaml['data_dev_included'].items():
            if not row[constraint] in values:
                add=False

        # If the constraints match, the max dev size is not exceeded and the item is not in the test set, add it to the training set
        if add and len(df_train_indices) < yaml['data_dev_max_size'] and not (index in df_test_indices or index in df_unlabeled_test_indices):
            if pd.isna(row[y_name]): # or pd.isna(row[x_names]).any():
                df_unlabeled_train_indices.append(index)
            else:
                df_train_indices.append(index)

    if bs > 0 and len(df_test_indices) > 0: # only use bootstrapping if bs > 0 and IF test data is provided (otherwise repeated cross validation is used)
        random.seed(bs)
        df_train_indices = random.choices(df_train_indices, k=len(df_train_indices))
        df_test_indices = random.choices(df_test_indices, k=len(df_test_indices))

    return  df_unlabeled_train_indices, df_train_indices, df_unlabeled_test_indices, df_test_indices

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def check_singularity(a, tol=1e-8):
    return abs(np.linalg.det(a)) < tol

def isPSD(A, tol=1e-8):
  E = np.linalg.eigvalsh(A)
  return np.all(E > -tol)

def noisy_bernoulli_model_sample(probs, noise_level=0.0):
    random_outcomes = np.random.binomial(1, [0.5 for _ in range(len(probs))]).flatten()
    true_bernoulli_sampled_outcomes = np.random.binomial(1, probs).flatten()
    true_separated_outcomes = [1.0 if p > 0.5 else 0.0 for p in probs]
    take_random_outcome = np.random.binomial(1, [abs(noise_level) for _ in range(len(probs))]).flatten()
    if noise_level >=0:
        return [random_outcomes[i] if take_random_outcome[i] else true_bernoulli_sampled_outcomes[i] for i in range(len(probs))]
    elif noise_level <0: # remove the stochasticity of the bernoulli (resulting in a separation as the 'noise_level' becomes more negative: full separation if noise_level = -1)
        return [true_separated_outcomes[i] if take_random_outcome[i] else true_bernoulli_sampled_outcomes[i] for i in range(len(probs))]


def simulate_data(df, x_names, y_name, yaml_dict, base_model, df_train_indices, printer=print, save_dir=None):

    printer('Preparing simulated data and ground truth model...')
    t0 = time.time()
    # 1. getting the mean and covariance of the real data
    sim_mean = np.nanmean(df.iloc[df_train_indices][x_names].values, axis=0)
    masked = np.ma.MaskedArray(df.iloc[df_train_indices][x_names].values, np.isnan(df.iloc[df_train_indices][x_names]))
    original_cov = np.ma.cov(masked, rowvar=0).data
    sim_cov = deepcopy(original_cov)
    printer(sim_cov)
    printer('pos def', isPSD(sim_cov), check_symmetric(sim_cov), check_singularity(sim_cov))
    median_vif = -1
    vif_cov_stepsize = 0.01
    for attempt in range(0, 100): #
        # 2. sample new train and test X (from a multivariate Gaussian)
        sim_df_train = pd.DataFrame(index=range(yaml_dict['sim_train_size']), columns=df.keys())
        sim_df_train.at[range(yaml_dict['sim_train_size']), x_names] = np.random.multivariate_normal(sim_mean, sim_cov, yaml_dict[
            'sim_train_size'],check_valid='warn')
        mean_vif, median_vif, vifs = get_vif_stats(sim_df_train[x_names])

        # 2.1 If needed, change the covariance such that the target VIF is obtained.
        printer('train vif', round(median_vif, 0), 'target',round(yaml_dict['sim_target_vif']))
        if yaml_dict['sim_target_vif'] == 0:
            printer('No changing of the covariance matrix: keeping the same covariance as the real data.')
            break

        L = la.cholesky(sim_cov)
        Ldiag = np.diag(L)
        if round(median_vif,0) == round(yaml_dict['sim_target_vif']):
            printer('good vif')
            break
        elif round(median_vif,0) < round(yaml_dict['sim_target_vif']): # too low vif; increase cov (outside diagonal)
            Ln = L* (1.0 + vif_cov_stepsize)
            np.fill_diagonal(Ln, Ldiag)
            sim_cov = np.dot(Ln.T, Ln)

        elif round(median_vif,0) > round(yaml_dict['sim_target_vif']):# too high vif; decrease cov (outside diagonal)
            Ln = L* (1.0 - vif_cov_stepsize)
            np.fill_diagonal(Ln, Ldiag)
            sim_cov = np.dot(Ln.T, Ln)

    sim_df_test = pd.DataFrame(index=range(yaml_dict['sim_test_size']), columns=df.keys())
    sim_df_test.at[range(yaml_dict['sim_test_size']), x_names] = np.random.multivariate_normal(sim_mean, sim_cov, yaml_dict[
            'sim_test_size'],check_valid='warn')
    mean_test_vif, median_test_vif, test_vifs = get_vif_stats(sim_df_test[x_names])
    mean_train_vif, median_train_vif, train_vifs = get_vif_stats(sim_df_train[x_names])

    printer('train med vif', round(median_train_vif, 0), 'mean vif', round(mean_train_vif, 0))
    printer('test med vif', round(median_test_vif, 0), 'mean vif', round(mean_test_vif, 0))

    # 3. fit a GT model
    df_tmp = df_tmp = deepcopy(df)
    if yaml_dict['preproc_zero_mean'] or yaml_dict['preproc_unit_variance']:
        scaler = StandardScaler(with_mean=yaml_dict['preproc_zero_mean'], with_std=yaml_dict['preproc_unit_variance'])
        df_tmp.at[df_train_indices, x_names] = scaler.fit_transform(df_tmp.iloc[df_train_indices][x_names].values)
        sim_X_train = scaler.transform(sim_df_train[x_names].values)
        sim_X_test = scaler.transform(sim_df_test[x_names].values)
    else:
        sim_X_train = sim_df_train[x_names].values
        sim_X_test = sim_df_test[x_names].values

    imputer = IterativeImputer(max_iter=yaml_dict['mice_imputations'], random_state=0, missing_values=np.nan)
    df_tmp.at[df_train_indices, x_names] = imputer.fit_transform(df_tmp.iloc[df_train_indices][x_names])
    base_model.fit(df_tmp.iloc[df_train_indices][x_names].values,
                   df_tmp.iloc[df_train_indices][y_name].values.flatten())
    real_prevalence = sum(df_tmp.iloc[df_train_indices][y_name].values.flatten()) / len(df_train_indices)
    printer('real prevalence', real_prevalence)


    # 4. Tune GT AUC by potentiall adding noise to the predicted probability (control the overlap between outcomes)
    intercept_stepsize = 0.1
    scale_step = 0.1
    scale_factor=1.0
    max_patience = 10 # after 25 noise levels without improvement cancel
    argmin_noise, argmin_intercept = None,None
    min_diff = np.inf
    patience = max_patience
    initial_intercept = deepcopy(base_model.best_estimator_.get_intercept().data.numpy())
    initial_coeffs = deepcopy(base_model.best_estimator_.linear.weight.data)
    explored_scalings = set([])
    for i in range(1000):
        base_model.best_estimator_.linear.weight.data = initial_coeffs * scale_factor
        explored_intercepts = set([])
        for j in range(1000):
            test_probs = list(base_model.predict_proba(sim_X_test)[:, 1].flatten())
            train_probs = list(base_model.predict_proba(sim_X_train)[:, 1].flatten())
            prevs, aucs = [], []
            current_intercept = base_model.best_estimator_.get_intercept().data.numpy()[0]
            for rseed in range(10):
                test_labels = noisy_bernoulli_model_sample(test_probs, 0.0)
                train_labels = noisy_bernoulli_model_sample(train_probs, 0.0)

                aucs.append(sklearn.metrics.roc_auc_score(test_labels,test_probs))  # <<<<< AUC is measured for TEST
                prevs.append(sum(train_labels) / len(train_labels))                 # <<<<< prevalence is measured for TRAIN (to ensure correct EPV)

            auc_diff = yaml_dict['sim_target_gt_AUROC'] - np.mean(aucs)
            prev_diff = real_prevalence - np.mean(prevs)
            diff = abs(auc_diff) + abs(prev_diff) # AUC and prevalence are equally weighted
            if diff < min_diff:
                argmin_scale_factor, argmin_intercept = deepcopy(scale_factor), deepcopy(base_model.best_estimator_.get_intercept())
                min_diff = diff
                #print('> saving noise',round(argmin_noise,4),'intercept',current_intercept)
                printer('target auc',yaml_dict['sim_target_gt_AUROC'], 'target prev', round(real_prevalence,4), 'scale', round(scale_factor, 4), 'intercept', current_intercept, 'auc', round(np.mean(aucs), 4),
                      'prev', round(np.mean(prevs), 4), '<')
                patience = max_patience
            else:
                printer('target auc',yaml_dict['sim_target_gt_AUROC'], 'target prev', round(real_prevalence,4), 'scale', round(scale_factor, 4), 'intercept', current_intercept, 'auc', round(np.mean(aucs), 4), 'prev', round(np.mean(prevs), 4))


            if current_intercept in explored_intercepts:
                break
            elif prev_diff > 0.0: # prevalence is too low -> increase intercept
                explored_intercepts.add(current_intercept)
                base_model.best_estimator_.set_intercept(base_model.best_estimator_.get_intercept() + intercept_stepsize)
            elif prev_diff < 0.0:  # prevalence is too high -> decrease intercept
                explored_intercepts.add(current_intercept)
                base_model.best_estimator_.set_intercept(base_model.best_estimator_.get_intercept() - intercept_stepsize)
        #patience-= 1
        #if patience == 0 and not min_diff == np.inf:
        #    printer('no more patience')
        #    break
        #print('scale factor',scale_factor, 'auc',np.mean(aucs))
        if yaml_dict['sim_target_gt_AUROC'] == 0:
            printer('Not tuning AUC / scaling model, keeping original AUC.')
            break

        if scale_factor in explored_scalings:
            printer('Scale factor',scale_factor,'already explored', explored_scalings)
            break
        elif auc_diff > 0: # AUC is too low (make stronger separation)
            explored_scalings.add(scale_factor)
            scale_factor += scale_step
        elif auc_diff < 0: # AUC is too high (add noise)
            explored_scalings.add(scale_factor)
            scale_factor -= scale_step


    # *** use the best found values as actual scale factor and intercept
    base_model.best_estimator_.set_intercept(argmin_intercept)
    base_model.best_estimator_.linear.weight.data = initial_coeffs * argmin_scale_factor


    y_sim_test_gt_probs = base_model.predict_proba(sim_X_test)[:,1].flatten()
    y_sim_train_gt_probs = base_model.predict_proba(sim_X_train)[:,1].flatten()
    sim_y_binary_test = noisy_bernoulli_model_sample(y_sim_test_gt_probs, 0.0)
    sim_y_binary_train = noisy_bernoulli_model_sample(y_sim_train_gt_probs, 0.0)
    printer('REAL')
    printer('Original intercept', initial_intercept, 'used intercept', base_model.best_estimator_.linear.bias.data)
    printer('real prev', real_prevalence)
    printer('target AUROC',yaml_dict['sim_target_gt_AUROC'])
    printer('SIMULATED')
    printer('scale factor', argmin_scale_factor)
    printer('Adjusted intercept',base_model.best_estimator_.get_intercept())
    printer('test prevalence', round(sum(sim_y_binary_test) / len(sim_y_binary_test), 4))
    printer('train prevalence', round(sum(sim_y_binary_train) / len(sim_y_binary_train),4))
    printer('train AUROC', round(sklearn.metrics.roc_auc_score(sim_y_binary_train,y_sim_train_gt_probs),4))
    printer('test AUROC',  round(sklearn.metrics.roc_auc_score(sim_y_binary_test,y_sim_test_gt_probs),4))

    # 5. add simulated data to the dataframe
    # create train and test indices
    sim_train_indices, sim_test_indices = range(df.shape[0],df.shape[0]+yaml_dict['sim_train_size']), range(df.shape[0]+yaml_dict['sim_train_size'], df.shape[0]+yaml_dict['sim_train_size']+yaml_dict['sim_test_size'])

    printer(len(sim_train_indices), len(sim_test_indices))
    # set the column names of the predictors (X) and outcome (y)
    sim_column_names = x_names + [y_name]

    # create an empty train dataframe
    sim_train_df = pd.DataFrame(np.nan, index=sim_train_indices, columns=sim_column_names)
    # and fill it
    sim_train_df.at[sim_train_indices, x_names] = sim_df_train[x_names].values
    sim_train_df.at[sim_train_indices, y_name] = sim_y_binary_train

    # create an empty test dataframe
    sim_test_df = pd.DataFrame(np.nan, index=sim_test_indices, columns=sim_column_names)
    # and fill it
    sim_test_df.at[sim_test_indices, x_names] = sim_df_test[x_names].values
    sim_test_df.at[sim_test_indices, y_name] = sim_y_binary_test



    df = pd.concat([df, sim_train_df, sim_test_df])

    if save_dir:
        printer('saving simulation information at', save_dir)
        os.makedirs(save_dir)
        with open(save_dir + '/real_cov.p','wb') as f:
            pickle.dump(original_cov, f)
        with open(save_dir + '/means.p','wb') as f:
            pickle.dump(sim_mean, f)
        with open(save_dir + '/sim_cov.p','wb') as f:
            pickle.dump(sim_cov, f)
        with open(save_dir + '/sim_mean.p','wb') as f:
            pickle.dump(sim_mean, f)
        with open(save_dir +'/sim_gt_model.p','wb') as f:
            pickle.dump(base_model, f)
        with open(save_dir+'/train_data.p','wb') as f:
            pickle.dump(sim_train_df, f)
        with open(save_dir+'test_data.p','wb') as f:
            pickle.dump(sim_test_df, f)
        with open(save_dir+'test_probs.p','wb') as f:
            pickle.dump(y_sim_test_gt_probs, f)

    printer('Simulated data constructed in', round(time.time()-t0,0), 's')
    return df, sim_train_indices, sim_test_indices, base_model

class CompareMethods(Experiment):

    def __init__(self, data, out_folder, name="EXP", reverse_train_test=False, describe_data=False):
        self.reverse_train_test=reverse_train_test
        self.describe_data=describe_data
        super(CompareMethods, self).__init__(data, out_folder=out_folder, name="CompareMethods:" + name)

    def conduct(self, yaml_dict, metrics_file=False, train_metrics_file=False, models_file=False, bs=0):

        self.log('Comparison:', yaml_dict['models'])

        # encode all predictor variables into dichotomous variables if needed
        df, x_names, y_name = encode_variables(self.data, yaml_dict)

        # split train and test (if the test set set empty, cross validation is performed later on)
        df_unlabeled_train_indices, df_train_indices, df_unlabeled_test_indices, df_test_indices = select_train_test_patients(df, yaml_dict, y_name, x_names, bs=bs)

        # If there is any validation data specified, use a train test split evaluation, otherwise use a k-fold cross validation on the development data
        use_train_test_split = yaml_dict["data_val_max_size"] > 0

        self.log('Trainset: X shape:',len(x_names),'x',len(df_train_indices), '\ty shape: 1 x',len(df_train_indices))
        self.log('X:',', '.join(x_names),'.')


        if yaml_dict['log_include_data_description_bs0'] and bs == 0:
            # describe the training data (only if the bootstrap id = 0; the original dataset)
            describe_data(df.iloc[df_train_indices][[y_name]+yaml_dict['log_vars_in_data_description']], self.exp_dir + '/dev_data_description/', y_name, pairplot=yaml_dict['log_include_pairplot'])
            assess_collinearity(df.iloc[df_train_indices][x_names], self.exp_dir + '/collinearity_diagnostics/', self.log)

        if use_train_test_split:
            # describe the test data (only if the bootstrap id = 0; the original dataset)
            self.log('Testset: X shape:',len(x_names),'x',len(df_test_indices), '\ty shape: 1 x',len(df_test_indices))
            if yaml_dict['log_include_data_description_bs0'] and bs == 0:
                describe_data(df.iloc[df_test_indices][[y_name]+yaml_dict['log_vars_in_data_description']], self.exp_dir + '/val_data_description/', y_name, pairplot=yaml_dict['log_include_pairplot'])

        # optionally reverse the train and test splits (in the cross-validation) to mimic a smaller training set
        self.log('reverse_train_test:',self.reverse_train_test)

        # read non-negative dimensions and order constraints from yaml
        nn_dims = [i for i,x in enumerate(x_names) if x[2:] in yaml_dict['preds_nonneg']]
        order_vol_dims_labels, order_vol_dims_indices = find_X_pairs(yaml_dict['preds_coef_order'], x_names) # [('v05','v10'),('v10','v20'),('v20','v30'),('v30','v40'),('v40','v50'),('v50','v60'),('v60','v70')]
        self.log('Order', order_vol_dims_labels)

        # Hyperparameters from yaml
        c_grid = Real(min(yaml_dict['hypps_regularization_grid']),max(yaml_dict['hypps_regularization_grid']), prior='log-uniform')
        n_components_grid = Integer(min([v for v in yaml_dict['hypps_latent_component_grid'] if v < len(x_names)]),min(len(x_names),len(df_train_indices)))
        self.log('n_components_grid',n_components_grid)

        dropout_grid = Real(min(yaml_dict['hypps_dropout_grid']),max(yaml_dict['hypps_dropout_grid']), prior='uniform')
        verbosity=self.exp_dir if yaml_dict['log_verbose'] else 0
        n_search_folds=yaml_dict['hypps_n_search_folds']
        n_jobs=yaml_dict['hypps_n_jobs']
        n_BO_hypp_iters=yaml_dict['hypps_n_bo_iters']
        print('BO iters', n_BO_hypp_iters)

        self.log('nested hyperparameter n_folds:', n_search_folds)
        if not use_train_test_split:
            self.log('Evaluation n_folds:', yaml_dict['hypps_eval_cv_folds'])

        # negative log likelihood is used to select hyperparameters in the nested cross-validation hypp grid search
        hypp_score_function = make_scorer(log_loss, greater_is_better=False, needs_proba=True, needs_threshold=False)

        # Set up some model specifications
        models = {}


        models['SK-LR'] = GridSearchCV(LogisticRegression(penalty='none',solver='saga'), {}, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)
        models['LR'] = GridSearchCV(ConstrainedLogisticRegression(verbose=verbosity), {}, scoring=hypp_score_function, cv=2, n_jobs=n_jobs, iid=False)
        models['Lasso'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity), {'L1_C': c_grid},n_iter=n_BO_hypp_iters,scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)
        models['Ridge'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity), {'L2_C':c_grid}, n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)
        models['ElasticNet'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity), {'L1_C': c_grid, 'L2_C':c_grid}, n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)

        models['PCLR'] = BayesSearchCV(PCALogit(n_components=1,verbose=verbosity), {'n_components':n_components_grid}, n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)
        models['PCLRunlab'] = BayesSearchCV(PCALogit(n_components=1,verbose=verbosity, unlab_X=[True]), {'n_components':n_components_grid}, n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)
        models['LAELR'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity), {'LAE_h':n_components_grid,'LAE_C':c_grid},n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)
        models['LAELRunlab'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity, unlab_X=[True]), {'LAE_h':n_components_grid,'LAE_C':c_grid},n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)
        models['Dropout'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity), {'dropout_ratio':dropout_grid},n_iter=n_BO_hypp_iters , scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)
        models['LRnn'] = GridSearchCV(ConstrainedLogisticRegression(verbose=verbosity,positive_coef=nn_dims), {}, scoring=hypp_score_function, cv=2, n_jobs=n_jobs, iid=False)

        models['L2+NN'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity,positive_coef=nn_dims), {'L2_C':c_grid}, n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=2, n_jobs=n_jobs, iid=False)
        models['LAELR+Dropout'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity), {'dropout_ratio':dropout_grid,'LAE_h':n_components_grid,'LAE_C':c_grid},n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)
        models['LAELR+L2'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity), {'L2_C':c_grid,'LAE_h':n_components_grid,'LAE_C':c_grid},n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)
        models['LAELR+L2+NN'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity,positive_coef=nn_dims), {'nn_C':c_grid, 'L2_C':c_grid,'LAE_h':n_components_grid,'LAE_C':c_grid},n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)

        models['LAELRnn'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity,positive_coef=nn_dims), {'nn_C':c_grid, 'LAE_h':n_components_grid,'LAE_C':c_grid},n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)

        models['LAELRunlab+nn'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity,positive_coef=nn_dims, unlab_X=[True]), {'nn_C':c_grid, 'LAE_h':n_components_grid,'LAE_C':c_grid},n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)


        if len(order_vol_dims_labels) > 0:
            models['LRmon'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity,ordered_coefficients=[order_vol_dims_indices]), {'order_C':c_grid},n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)
            models['LRnn+mon'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity,ordered_coefficients=[order_vol_dims_indices],positive_coef=nn_dims), {'order_C':c_grid},n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)

            models['LAELRnn+mon'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity,ordered_coefficients=[order_vol_dims_indices],positive_coef=nn_dims), {'nn_C':c_grid, 'LAE_h':n_components_grid,'LAE_C':c_grid,'order_C':c_grid},n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)
            models['LAELRunlab+nn+mon'] = BayesSearchCV(ConstrainedLogisticRegression(verbose=verbosity,ordered_coefficients=[order_vol_dims_indices],positive_coef=nn_dims, unlab_X=[True]), {'nn_C':c_grid, 'LAE_h':n_components_grid,'LAE_C':c_grid,'order_C':c_grid},n_iter=n_BO_hypp_iters, scoring=hypp_score_function, cv=n_search_folds, n_jobs=n_jobs, iid=False)


        models = {model_name:model_setup for (model_name, model_setup) in models.items() if model_name in yaml_dict['models']}


        print('order_vol_dims_indices',order_vol_dims_indices)

        # Set up an evaluator
        evaluator = Evaluator(oar_dims=nn_dims, oar_order=[], vol_order=order_vol_dims_indices, print_func=self.log)



        # ============= HERE THE ACTUAL EXPERIMENTS START ===================


        print('train indices',df_train_indices)
        if yaml_dict['simulate']:
            np.random.seed(bs)
            random.seed(bs)
            self.log('Running a simulation!!! <======')
            base_model = deepcopy(models[yaml_dict['sim_base_model']])
            df, sim_train_indices, sim_test_indices, ground_truth_model = simulate_data(df, x_names, y_name, yaml_dict, base_model, df_train_indices, printer=self.log, save_dir=self.exp_dir + '/sim/')

            train_test_pairs = [(sim_train_indices, sim_test_indices)]

        elif use_train_test_split:
            train_test_pairs = [(df_train_indices, df_test_indices)]
        else:
            cv_splitter = StratifiedKFold(n_splits=yaml_dict['hypps_eval_cv_folds'], shuffle=bool(bs), random_state=bs)
            train_test_pairs_skl = list(cv_splitter.split(df.iloc[df_train_indices][x_names].values, df.iloc[df_train_indices][y_name].values))
            # map cv_splitter indices back to original dataframe indices
            train_test_pairs = [([df_train_indices[i] for i in cv_train_index], [df_train_indices[j] for j in cv_test_index]) for (cv_train_index, cv_test_index) in train_test_pairs_skl]


        print('train_test_pairs', train_test_pairs)
        metrics = {model_name:{m:[] for m in evaluator.METRICS} for model_name in models}
        train_metrics = {model_name:{m:[] for m in evaluator.METRICS} for model_name in models}

        trained_models = {model_name:[] for model_name in models}
        predicted_probs_per_model = {model_name:[] for model_name in models}
        labels = []
        loss_dir = self.exp_dir + '/loss_plots/'
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)

        for split_index, (train_indices, test_indices) in enumerate(train_test_pairs):
            self.log('===== Running data split', split_index + 1,'/',len(train_test_pairs), '======')
            self.log('labeled train:',len(train_indices),'test:', len(test_indices))
            train_indices = train_indices
            self.log('unlabeled',len(df_unlabeled_train_indices))
            num_events = sum(df.iloc[train_indices][y_name].values)
            self.log('No. events', num_events)
            self.log('EPV',round(num_events / len(x_names),1))

            df_tmp = deepcopy(df) # create a temporary copy of the data, to prevent any leaking of standardization / imputation info across folds

            if yaml_dict['preproc_zero_mean'] or yaml_dict['preproc_unit_variance']:
                # Standardize the data
                self.log('Standardizing data')
                scaler = StandardScaler(with_mean=yaml_dict['preproc_zero_mean'], with_std=yaml_dict['preproc_unit_variance'])
                df_tmp.at[train_indices, x_names] = scaler.fit_transform(df_tmp.iloc[train_indices][x_names].values)
                df_tmp.at[test_indices, x_names] = np.nan_to_num(scaler.transform(df_tmp.iloc[test_indices][x_names].values))

            # Impute missing input values
            self.log('Imputing missing values')
            imputer = IterativeImputer(max_iter=yaml_dict['mice_imputations'], random_state=0, missing_values=np.nan)
            df_tmp.at[train_indices, x_names] = imputer.fit_transform(df_tmp.iloc[train_indices][x_names])
            df_tmp.at[df_unlabeled_train_indices, x_names] = imputer.transform(df_tmp.iloc[df_unlabeled_train_indices][x_names]) # UNLABELED DATA PREPROCESSING (TAKES TIME IF NO UNLABELED DATA IS USED...) <<<<<<<<
            df_tmp.at[test_indices, x_names] = imputer.transform(df_tmp.iloc[test_indices][x_names])


            labels += list(df_tmp.iloc[test_indices][y_name].values)
            for model_name, model in models.items():
                t0 = time.time()
                self.log('---', model_name, '---')
                model_tmp = deepcopy(model)
                if len(model_tmp.estimator.unlab_X) > 0:
                    model_tmp.estimator.unlab_X=df_tmp.iloc[df_unlabeled_train_indices][x_names].values
                    self.log('using unlab data')

                trained_model = model_tmp.fit(df_tmp.iloc[train_indices][x_names].values, df_tmp.iloc[train_indices][y_name].values.flatten())


                plt.plot(trained_model.best_estimator_.streamed_loss)
                plt.savefig(loss_dir + 'split_'+str(split_index)+'_' +model_name+'.png')
                plt.close()

                t1 = time.time()
                self.log('model training took', int(t1-t0),'s')
                split_eval, pred_probs = evaluator.eval_model(trained_model, df_tmp.iloc[test_indices][x_names], df_tmp.iloc[test_indices][y_name])
                train_split_eval, train_pred_probs = evaluator.eval_model(trained_model, df_tmp.iloc[train_indices][x_names], df_tmp.iloc[train_indices][y_name])
                predicted_probs_per_model[model_name] += list(pred_probs[:,1])

                t2=time.time()
                self.log('model eval took',int(t2-t1),'s')
                self.log('best params',trained_model.best_params_)
                self.log('TRAIN EVAL',train_split_eval)
                self.log('HELD OUT EVAL',split_eval)
                for m in split_eval:
                    metrics[model_name][m].append(split_eval[m])
                for m in train_split_eval:
                    train_metrics[model_name][m].append(train_split_eval[m])
                trained_models[model_name].append(trained_model)


        write_metrics_to_file(metrics, self.exp_dir +'/metrics.csv', evaluator, prefix='bs' + str(bs) + '-')
        write_metrics_to_file(train_metrics, self.exp_dir +'/train_metrics.csv', evaluator, prefix='bs' + str(bs) + '-')
        write_models_to_file(trained_models, self.exp_dir+'/models.csv', x_names, prefix='bs'+str(bs)+'-')
        write_metrics_to_file(metrics, metrics_file, evaluator, prefix='bs'+str(bs)+'-')
        write_metrics_to_file(train_metrics, train_metrics_file, evaluator, prefix='bs'+str(bs)+'-')
        write_models_to_file(trained_models, models_file, x_names, prefix='bs'+str(bs)+'-')

        # write models and metrics to file as backup (pickle)

        with open(self.exp_dir + '/trained_models.p','wb') as f:
            pickle.dump(trained_models,f)
        with open(self.exp_dir + '/metrics.p','wb') as f:
            pickle.dump(metrics,f)
        with open(self.exp_dir + '/train_metrics.p','wb') as f:
            pickle.dump(train_metrics,f)

        plot_difference_in_predicted_probs(predicted_probs_per_model, labels, self.exp_dir +'/pred_probs/')
        Rval(predicted_probs_per_model, labels, self.exp_dir +'/rvals/')

        self.log(self.name, 'Done!')




def write_metrics_to_file(metrics, metrics_file, evaluator, prefix):
    if not os.path.exists(metrics_file):
        write_table_line([[m+'_mean', m+'_std'] for m in evaluator.METRICS], metrics_file, write_style='w', prefix='Model', format='csv')
    for model_name, results in metrics.items():
        write_table_line([[np.mean(values),np.std(values)] for metric, values in metrics[model_name].items()], metrics_file, write_style='a', prefix=prefix + model_name)

def write_models_to_file(models, models_file, predictor_names, prefix):
    if not os.path.exists(models_file):
        write_table_line([['Intercept_mean','Intercept_std']]+ [[p+'_mean', p+'_std'] for p in predictor_names], models_file, write_style='w', prefix='Model', format='csv')
    for model_name, results in models.items():
        intercept_mean, intercept_std = np.mean([m.best_estimator_.intercept_ for m in models[model_name]], axis=0).flatten()[0], np.std([m.best_estimator_.intercept_ for m in models[model_name]], axis=0).flatten()[0]
        coef_means, coef_stds = np.mean([m.best_estimator_.coef_ for m in models[model_name]], axis=0).flatten(), np.std([m.best_estimator_.coef_ for m in models[model_name]], axis=0).flatten()
        write_table_line([[intercept_mean,intercept_std]] + [[coef_means[i],coef_stds[i]] for i in range(len(coef_means))], models_file, write_style='a', prefix=prefix + model_name)


def test():
    exp = Experiment([], 'test_exp')
    exp.conduct()

#test()
