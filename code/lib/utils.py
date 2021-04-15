
import pandas, re, time
import numpy as np
import math, os, shutil
import scipy
import pylab
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import numpy as np
import pickle
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import statsmodels.api as sm

import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
calR = importr('CalibrationCurves')
grdevices = importr('grDevices')


def calculate_recommended_sample_sizes(outcome_proportion, num_patients, num_features, alpha=0.05, expected_uniform_shrinkage_factor=0.9, expected_nagR2=0.15, expected_optimism=0.05):
    # Implemented sample size calculations based on recent sample size calculation research: http://doi.org/10.1136/bmj.m441

    print('---')
    # STEP ONE
    n_1 = (1.96/ alpha)**2 * outcome_proportion * (1.0-outcome_proportion)
    print('n_1:', math.ceil(n_1))

    # STEP 2: empirically obtained formula, recommended for num_features <= 30
    n_2 = np.exp((-0.508 + 0.259 * np.log(outcome_proportion) + 0.504 * np.log(num_features) - np.log(alpha)) / 0.544)
    print('n_2:', math.ceil(n_2))

    # STEP 3: sample size needed to achieve an expected uniform shrinkage factor of S
    #  What sample size will produce a small required shrinkage of predictor effects?
    # first estimate the Max Cox-Snell R-squared (following code by E. Schuit)
    num_events = outcome_proportion * num_patients
    lnll = num_events * np.log(outcome_proportion) + (num_patients-num_events) * np.log(1.0-outcome_proportion)
    maxR2cs = 1 - np.exp(2*lnll / num_patients)
    R2cs = expected_nagR2 * maxR2cs

    n_3 = float(num_features) / ((expected_uniform_shrinkage_factor-1.0) * np.log(1.0 - (R2cs / expected_uniform_shrinkage_factor)))
    print('n_3', math.ceil(n_3))

    # STEP 4 the sample size needed to target a small optimism in model fit
    S = R2cs / (R2cs + expected_optimism * maxR2cs)
    n_4 = num_features / ((S-1) * np.log(1 - (R2cs / S)))
    print('n_4', math.ceil(n_4))
    print('>> RSS:', np.max([n_1, n_2, n_3, n_4]))

# # XER M6
# calculate_recommended_sample_sizes(outcome_proportion=0.27, num_features=7, num_patients=741, expected_nagR2=0.42)
# calculate_recommended_sample_sizes(outcome_proportion=0.27, num_features=19, num_patients=741, expected_nagR2=0.42)

# # DYS M6
# calculate_recommended_sample_sizes(outcome_proportion=0.14, num_features=13, num_patients=744, expected_nagR2=0.26)
# calculate_recommended_sample_sizes(outcome_proportion=0.14, num_features=43, num_patients=744, expected_nagR2=0.26)


def assess_collinearity(X, out_dir, printing_func=print):
    t0 = time.time()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    Xcopy = X.dropna()
    printing_func('assessing collinearity')
    printing_func('excluded rows with missings (only for collinearity assessment)',len(X)-len(Xcopy.values))
    vif = pandas.DataFrame()

    corr_mat = np.array(Xcopy.corr())
    inv_corr_mat = np.linalg.inv(corr_mat)
    vif['VIF Factor'] = pandas.Series(np.diag(inv_corr_mat))
    vif["Features"] = X.columns

    vif.to_csv(out_dir +'/VIFs.csv')
    printing_func('mean VIF', round(np.mean(vif['VIF Factor']),2))
    printing_func('median VIF', round(np.median(vif['VIF Factor']),2))
    printing_func('assessing collinearity took', round(time.time()-t0,2),'s')


def get_vif_stats(X):
    Xcopy = X.dropna()

    vif = pandas.DataFrame()
    Xcorr= Xcopy.astype(float).corr()
    corr_mat = np.array(Xcorr)
    inv_corr_mat = np.linalg.inv(corr_mat)
    vif['VIF Factor'] = pandas.Series(np.diag(inv_corr_mat))
    vif["Features"] = X.columns
    return np.mean(vif['VIF Factor']), np.median(vif['VIF Factor']), vif




def plot_difference_in_predicted_probs(predicted_probs_per_model, true_labels, out_dir):
    # taken from https://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python
    t0 = time.time()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(out_dir +'/model_preds.pickle', 'wb') as f:
        pickle.dump(predicted_probs_per_model, f)
    with open(out_dir +'/true_labels.pickle', 'wb') as f:
        pickle.dump(true_labels, f)

    make_dendogram(predicted_probs_per_model, out_dir + '/dendrogram.png')

    plot_calib_curves(predicted_probs_per_model, true_labels, out_dir + '/calibration.png')
    print('plotting took', round(time.time()-t0),'s')

def Rval(predicted_probs_per_model, true_labels, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    vals = {}
    rval = ro.r['val.prob.ci.2'] # Calculate metrics using R function val.prob.ci.2 and save them to a pickle file
    for model in predicted_probs_per_model:
        val = rval(ro.FloatVector(predicted_probs_per_model[model]),ro.FloatVector(true_labels), pl = False, smooth=False)
        vals[model] = val
    with open(out_dir + '/rvals_per_model.p', 'wb') as f:
        pickle.dump(vals, f)
        print(vals)


def plot_calib_curves(predicted_probs_per_model, true_labels, fig_path, n_bins=10, dpi=300, ece=False, bbox_inches='tight'):
    # Plot calibration plots
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 20}

    plt.rc('font', **font)
    print('bins',n_bins)

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=3)

    ax1.plot([0, 1], [0, 1], "k:", label="")
    for name, prob_pos in predicted_probs_per_model.items():

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(true_labels, prob_pos, n_bins=n_bins, strategy='quantile')

        expected_calibration_error = sum([abs(o-e) for o,e in zip(fraction_of_positives, mean_predicted_value)]) / n_bins

        lab = "%s (ECE: %1.3f)" % (name,expected_calibration_error) if ece else "%s" % (name,)

        lowess = sm.nonparametric.lowess(fraction_of_positives, mean_predicted_value, frac=0.25)

        ax1.plot(lowess[:, 0], lowess[:, 1], "s-",
                          label=lab, ms=0, marker='o')

    ax1.set_ylabel("Fraction of positive outcomes")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('')
    ax1.set_xlabel("Mean predicted probability")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()



def make_dendogram(predicted_probs_per_model, img_file_path):
    model_names = list(predicted_probs_per_model.keys())
    num_samples = len(predicted_probs_per_model[model_names[0]])
    print(num_samples, model_names)

    # Generate random features and distance matrix.
    D = np.zeros([len(model_names), len(model_names)])
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            pm1 = np.array(predicted_probs_per_model[model_names[i]])
            pm2 = np.array(predicted_probs_per_model[model_names[j]])
            D[i, j] = sum(abs(pm1 - pm2)) / num_samples


    condensedD = squareform(D)

    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=(8, 8))

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.2, 0.71, 0.6, 0.2], frame_on=True)
    Y = sch.linkage(condensedD, method='single')
    Z2 = sch.dendrogram(Y)

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.2, 0.1, 0.75, 0.6])
    idx2 = Z2['leaves']
    D = D[idx2, :]
    D = D[:, idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.Reds)
    model_names_idx2 = [model_names[i] for i in idx2]
    axmatrix.set_xticks(range(len(model_names)))
    axmatrix.set_xticklabels(model_names_idx2, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()
    pylab.xticks(rotation=-40) #, fontsize=8)
    axmatrix.set_yticks(range(len(model_names)))
    axmatrix.set_yticklabels(model_names_idx2, minor=False)
    axmatrix.yaxis.set_label_position('left')
    axmatrix.yaxis.tick_left()
    # Plot colorbar.
    pylab.colorbar(im)
    fig.show()
    fig.savefig(img_file_path)

