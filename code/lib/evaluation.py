import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import random, torch
import itertools
from lib.utils import plot_calib_curves
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression, RFE
from copy import copy
from lib.constrainedlogit import ConstrainedLogisticRegression
import random

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

rms = importr('rms')
calR = importr('CalibrationCurves')


def eval(X, y, classifier, num_splits=5, title='Evaluation', univariate_feature_selection=0, show_plot=False, save_plot=False):
    cv = StratifiedKFold(n_splits=num_splits)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    overall_probs, overall_true_probs = [],[]
    calibs_in_the_largs = []
    for train, test in cv.split(X, y):

        Xtrain, Xtest = X[train], X[test]

        if univariate_feature_selection:
            selector = SelectKBest(f_classif, k=univariate_feature_selection).fit(Xtrain, y[train])
            #selector = RFE(classifier, univariate_feature_selection, step=10).fit(Xtrain, y[train])
            Xtrain = selector.transform(Xtrain)
            Xtest = selector.transform(Xtest)


        m =classifier.fit(Xtrain, y[train])
        probas_ = m.predict_proba(Xtest)
        #print(probas_)
        #print(probas_)

        overall_probs += list(probas_[:, 1])
        overall_true_probs += list(y[test])
        calib_in_the_large = (sum(y[test]) / len(y[test])) - np.mean(probas_.T[1])
#        print(calib_in_the_large)
        #print(sum(y[test]) / len(y[test]),  probas_)
        calibs_in_the_largs.append(calib_in_the_large)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f / Citl = %0.2f)' % (i, roc_auc, calib_in_the_large))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: ' + title)
    plt.legend(loc="lower right")
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_plot+'_auc.pdf')

    plt.close()

    #plt.figure(2)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    #ax2 = plt.subplot2grid((3, 1), (2, 0))
    fraction_of_positives, mean_predicted_value = calibration_curve(overall_true_probs, overall_probs, normalize=False, n_bins=10, strategy='uniform')
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s" % (title,), color='b', marker='o', markersize=5)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    #print(list(fraction_of_positives), list(mean_predicted_value))

    #print(len(fraction_of_positives),len(mean_predicted_value))
    br = ((np.array(fraction_of_positives)-np.array(mean_predicted_value))**2).mean(axis=0)
    EPV = sum(y) / Xtrain.shape[1]
    ax1.set_title("Calibration (Brier: {:0.4f} / EPV: {:0.2f} / Citl: {:0.4f})".format(br,EPV, np.mean(calibs_in_the_largs)))

    if show_plot:
        plt.show()

    if save_plot:
        plt.savefig(save_plot+'_calib.pdf')

    return {'aucs':aucs, 'mean_auc':mean_auc, 'std_auc':std_auc, 'epv':EPV, 'brier':br}



class Evaluator:
    AUC = "auc"
    CITL = "calibration-in-the-large"
    BRIER = "brier"
    CSLOPE = "calibration-slope"
    NEGOAR = "sum-neg-oars"
    PERCOARNEG = "perc-neg-oars"
    PERCGTZ = "perc-greater-than-zero"
    PERCOARORDER = "perc-oar-order-correct"
    PERCVOLORDER = "perc-vol-order-correct"

    METRICS = [AUC, CITL, CSLOPE, BRIER]

    def __init__(self, oar_dims=[], oar_order=[], vol_order=[], print_func=print):
        self.oar_dims=oar_dims
        self.oar_order=oar_order
        self.vol_order=vol_order
        self.print_func=print_func
        if len(oar_dims) > 0:
            if not self.NEGOAR in self.METRICS:
                self.METRICS.append(self.NEGOAR)
            if not self.PERCOARNEG in self.METRICS:
                self.METRICS.append(self.PERCOARNEG)
            if not self.PERCGTZ in self.METRICS:
                self.METRICS.append(self.PERCGTZ)

        if len(oar_order) > 0:
            if not self.PERCOARORDER in self.METRICS:
                self.METRICS.append(self.PERCOARORDER)
        if len(vol_order) > 0:
            if not self.PERCVOLORDER in self.METRICS:
                self.METRICS.append(self.PERCVOLORDER)


    def train_test_eval(self, model_object, X_train, y_train, X_test, y_test, bootstrap=1, reverse_train_test=False):
        metrics = {m:[] for m in self.METRICS}
        models = []
        if reverse_train_test:
            tmp_test_X, tmp_test_y  = copy(X_test), copy(y_test)
            X_test, y_test = copy(X_train), copy(y_train)
            X_train, y_train = tmp_test_X, tmp_test_y

        for b_i in range(bootstrap):
            indices = random.choices(range(X_train.shape[0]), k=X_train.shape[0]) if b_i > 0 else list(range(len(X_train))) # first bootstrap is the original set
            x_b, y_b = X_train.values[indices], y_train.values[indices]
            trained_model = model_object.fit(x_b, y_b.flatten())
            b_eval = self.eval_model(trained_model, X_test, y_test, bootstrap=0)
            for m in metrics:
                metrics[m].append(b_eval[m])
            models.append(copy(trained_model))

        resulting_metrics = {m: (np.mean(metrics[m]), np.std(metrics[m])) for m in metrics}
        intercept_mean, intercept_std = np.mean([m.best_estimator_.intercept_ for m in models],axis=0).flatten(), np.std([m.best_estimator_.intercept_ for m in models],axis=0).flatten()
        coef_means, coef_stds = np.mean([m.best_estimator_.coef_ for m in models],axis=0).flatten(),np.std([m.best_estimator_.coef_ for m in models], axis=0).flatten()
        model_parameters = {'intercept':(intercept_mean, intercept_std), 'coef':(coef_means, coef_stds)}
        return resulting_metrics, models, model_parameters



    def cv_eval(self, model_object, X, y, k=5,reverse_train_test=False):
        metrics = {m:[] for m in self.METRICS}
        cv = StratifiedKFold(n_splits=k,shuffle=True)
        models = []
        for train ,test in cv.split(X.values, y.values):
            if reverse_train_test: # train and test are swapped: train on small section, and test on larger section
                tmp_test = copy(test)
                test = copy(train)
                train = tmp_test
            trained_model = model_object.fit(X.values[train], y.values[train].flatten())
            fold_eval = self.eval_model(trained_model, X.iloc[test], y.iloc[test], bootstrap=0)
            for m in metrics:
                metrics[m].append(fold_eval[m])
            models.append(copy(trained_model))
        resulting_metrics = {m: (np.mean(metrics[m]), np.std(metrics[m])) for m in metrics}
        intercept_mean, intercept_std = np.mean([m.best_estimator_.intercept_ for m in models],axis=0).flatten(), np.std([m.best_estimator_.intercept_ for m in models],axis=0).flatten()
        coef_means, coef_stds = np.mean([m.best_estimator_.coef_ for m in models],axis=0).flatten(),np.std([m.best_estimator_.coef_ for m in models], axis=0).flatten()
        model_parameters = {'intercept':(intercept_mean, intercept_std), 'coef':(coef_means, coef_stds)}
        return resulting_metrics, models, model_parameters



    def optimism_corrected_bootstrap_eval(self, model_object, X, y, bootstrap=1000):
        or_model = model_object.fit(X, y)
        or_metrics = self.eval_model(self, or_model, X, y, bootstrap=0)
        optimisms = {m:[] for m in self.METRICS}
        for bstrap in range(bootstrap):
            indices = random.choices(range(X.shape[0]), k=X.shape[0])
            x_b, y_b = X[indices], y[indices]
            bstrap_model = model_object.fit(x_b, y_b)
            bstrap_eval = self.eval_model(bstrap_model, X, y)
            optimism = {m:bstrap_eval[m]-or_metrics[m] for m in self.METRICS}
            for m in self.METRICS:
                optimisms[m].append(optimism[m])
        corrected_metrics = {m:or_metrics[m] - np.mean(optimisms[m]) for m in self.METRICS}
        return corrected_metrics




    def get_perc_vol_order(self, model):
        correct, mistake = 0,0
        for (lower,upper) in self.vol_order:
            lower_coef = model.best_estimator_.coef_[0][lower]
            upper_coef = model.best_estimator_.coef_[0][upper]
            if lower_coef <= upper_coef:
                correct += 1
            else:
                mistake += 1
        return correct / (correct + mistake + 0.0000001)

    def get_perc_oar_order(self, model):
        correct, mistake = 0,0
        for (lower,upper) in self.oar_order:
            lower_coef = model.best_estimator_.coef_[0][lower]
            upper_coef = model.best_estimator_.coef_[0][upper]
            if lower_coef <= upper_coef:
                correct += 1
            else:
                mistake += 1
        return correct / (correct + mistake + 0.0000001)

    def get_perc_oar_neg(self, model):
        pos, neg = 0,0
        for i, coefficient in enumerate(model.best_estimator_.coef_[0]):
            if i in self.oar_dims:
                if coefficient < 0:
                    neg += 1
                else:
                    pos += 1
        return (neg / (neg + pos ))

    def get_perc_oar_greater_than_zero(self, model):
        pos, neg = 0,0
        for i, coefficient in enumerate(model.best_estimator_.coef_[0]):
            if i in self.oar_dims:
                if coefficient <= 0:
                    neg += 1
                else:
                    pos += 1
        return pos / (neg + pos)

    def get_negoar_score(self, model):
        score = 0
        for i, coefficient in enumerate(model.best_estimator_.coef_[0]):
            if i in self.oar_dims and coefficient < 0:
                score += coefficient
        return score


    def eval_model(self, model, X, y, bootstrap=0):
        Xvs, Yvs = X.values, y.values
        results = {}
        if bootstrap:
            metrics = {}
            for bstrap in range(bootstrap):
                indices = random.choices(range(Xvs.shape[0]),k=Xvs.shape[0])
                x_b,y_b = Xvs[indices], Yvs[indices]
                predicted_probs = model.predict_proba(x_b)[:, 1]
                #if getattr(model, "predict_lp", None) and callable(getattr(model, "predict_lp", None)):
                #    predicted_lp = model.predict_lp(Xvs)
                for metric,value in self.eval_model_predictions(predicted_probs, y_b).items():
                    if not metric in metrics:
                        metrics[metric] = []
                    metrics[metric].append(value)
            results = {m:(np.mean(metrics[m]),np.std(metrics[m])) for m in metrics}
        else:
            predicted_probs = model.predict_proba(Xvs)
            #if getattr(model, "predict_lp", None) and callable(getattr(model, "predict_lp", None)):
            #    predicted_lp = model.predict_lp(Xvs)
            results = self.eval_model_predictions(predicted_probs, Yvs)
        if self.NEGOAR in self.METRICS:
            results[self.NEGOAR] = self.get_negoar_score(model)
        if self.PERCOARNEG in self.METRICS:
            results[self.PERCOARNEG] = self.get_perc_oar_neg(model)
        if self.PERCGTZ in self.METRICS:
            results[self.PERCGTZ] = self.get_perc_oar_greater_than_zero(model)

        if self.PERCOARORDER in self.METRICS:
            results[self.PERCOARORDER] = self.get_perc_oar_order(model)
        if self.PERCVOLORDER in self.METRICS:
            results[self.PERCVOLORDER] = self.get_perc_vol_order(model)

        return results, predicted_probs

    def eval_model_predictions(self, predicted_probs, true_ys):
        if len(predicted_probs.shape) > 1 and predicted_probs.shape[1]==2:
            predicted_probs = predicted_probs[:,1]
        #fraction_of_positives, mean_predicted_value = calibration_curve(true_ys, predicted_probs, normalize=False, n_bins=10, strategy='uniform')
        brier_score = np.mean((np.array(predicted_probs) - np.array(true_ys)) **2)
        calibration_slope, calib_itl = self.calibration_slope_intercept(predicted_probs, true_ys)
        
        fpr, tpr, thresholds = roc_curve(true_ys, predicted_probs)
        roc_auc = auc(fpr, tpr)

        return {Evaluator.BRIER:brier_score,Evaluator.CITL:calib_itl, Evaluator.AUC:roc_auc, Evaluator.CSLOPE: calibration_slope}

    def logit_function(self, p):
        p = np.clip(p, 0.000000001, 0.999999999) # to prevent division by 0
        #r = np.log(p) - np.log(np.ones(len(p)) - p)
        return np.log(p/(1-p))


    def calibration_slope_intercept(self, predicted_y, true_y):
        #logit_function = lambda p: np.log(p/(1-p))
        y_pred_linear_scores = self.logit_function(predicted_y)
        #print(y_pred_linear_scores)
        calib_predictor = LogisticRegression(penalty='none',solver='saga',max_iter=10000000, tol=0.00001, intercept_scaling=0)
        #calib_predictor = ConstrainedLogisticRegression()

        try:
            calib_model = calib_predictor.fit(X=np.array(y_pred_linear_scores).reshape(-1, 1), y=np.array(true_y)) #, num_epochs=1000, early_stopping=1000, lr=0.1)
        except:
            self.print_func('predicted_y', predicted_y)
            self.print_func('true_y',true_y)
            self.print_func('y_pred_linear_scores',y_pred_linear_scores)
        print('COEF',calib_model.coef_)
        return calib_model.coef_[0], calib_model.intercept_

def write_table_line(list_of_mean_std_pairs, file_path, write_style='a', prefix="",format="csv", mean_only=False):
    with open(file_path, write_style) as f:
        if format =='tex':
            if type(list_of_mean_std_pairs[0][0]) == str or mean_only:
                formatted_values = [mean for mean, std in list_of_mean_std_pairs]
            else:
                formatted_values = ["${0:.2f}".format(round(mean,2)) + "^{\pm " + "{0:.2f}".format(round(std,2)) +"}$" for (mean, std) in list_of_mean_std_pairs]
                #formatted_values = list(itertools.chain.from_iterable(mean_std_values_tex))
            string = prefix + "&" + "&".join(formatted_values) + "\\\\\n"
        else:
            if mean_only:
                string = prefix + "\t" + "\t".join([str(mean) for mean, std in list_of_mean_std_pairs]) + "\n"
            else:
                string = prefix + "\t" + "\t".join([str(i) for i in itertools.chain.from_iterable(list_of_mean_std_pairs)]) + "\n"
        f.write(string)



def test():
    ev = Evaluator()
    random.seed(2)

    ps = [.1,.2,.3,.4,.5,.6,.7,.8,.9, 0.99] * 100
    X =  np.array([[x * 1 + 3, .5*x-1] for x in ps])
    LR = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=15, tol=0.0001, intercept_scaling=0, C=100)
    y_true = [int(round(i,0)) for i in ps]
    LRM = LR.fit(X, y_true)
    preds= LRM.predict_proba(X)[:,1]# [round(i,1) for i in ps]
    y_pred = preds #np.clip(preds, 0.1, 0.9)
    #print([i for i in zip(y_pred,y_true)])

    cs, ci = ev.calibration_slope_intercept(y_pred, y_true)

    rval = ro.r['val.prob']
    #rval = ro.r['val.prob']


    #Pred = ro.r.matrix(ro.FloatVector(y_pred))
    Pred = ro.FloatVector(y_pred)
    #print('Pred',Pred)
    yval = ro.IntVector(y_true)
    #print('yval', yval)
    res = rval(Pred, yval)
    print('R',res)
    #r = ro.FloatVector(y_pred)
    #print('FR',ro.FloatVector(y_pred))
    print(np.mean(y_true)- np.mean(y_pred),np.mean(y_true), np.mean(y_pred))
    #plot_calib_curves({'testmodel':y_pred},y_true,'test.png')
    #val.prob(y_pred, y_true)
    print('Python >>> Intercept', ci[0], 'Slope',cs[0])



#test()

