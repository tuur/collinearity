import numpy as np
from sklearn.metrics import r2_score
from sklearn import datasets
from sklearn.decomposition import PCA, SparsePCA, FastICA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from lib.spca import SupPCA
from lib.constrainedlogit import ConstrainedLogisticRegression

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class PCALogit(BaseEstimator):

    def __init__(self, n_components, penalty='none', C=1.0, pca_l1=0,verbose=1, unlab_X=[], streamed_loss=[]):
        self.n_components = n_components
        self.penalty = penalty
        self.C = C
        self.verbose=verbose
        self.pca_l1 = pca_l1
        self.unlab_X=unlab_X
        self.streamed_loss=streamed_loss


    def fit(self, Xlogit, Ylogit):
        self.pca = PCA(self.n_components) if not self.pca_l1 else SparsePCA(n_components=self.n_components, ridge_alpha=self.pca_l1)
        self.logit = ConstrainedLogisticRegression(L1_C=self.C if self.penalty=='l1' else 0.0, L2_C=self.C if self.penalty=='l2' else 0.0,verbose=self.verbose)
        if len(self.unlab_X) > 0:
            Xpca = np.concatenate([Xlogit, self.unlab_X])
        else:
            Xpca = Xlogit

        #print('len(xpca)',len(Xpca))
        self.pca.fit(Xpca)
        Hlogit = self.pca_transform(Xlogit)
        self.logit.fit(Hlogit, Ylogit)
        self.fitted = True

        xylogit = self.get_xy_logit()
        self.coef_ = xylogit.coef_
        self.intercept_ = xylogit.intercept_
        self.streamed_loss = self.logit.streamed_loss
        #print('r2:',self.explained_variance(Xpca))
        return self

    def get_params(self, deep=True):
        return {'streamed_loss':self.streamed_loss, 'unlab_X':self.unlab_X, 'n_components':self.n_components, 'penalty':self.penalty, 'C':self.C, 'pca_l1':self.pca_l1, 'verbose':self.verbose}


    def predict_proba(self, X):
        if not self.fitted:
            print('WARNING: PCALogit was used for prediction before fitting!')
        Hlogit = self.pca_transform(X)
        probs = self.logit.predict_proba(Hlogit)
        return probs

    def predict(self, X):
        Hlogit = self.pca_transform(X)
        return self.logit.predict(Hlogit)

    def get_xy_logit(self):
        xy_logit = LogisticRegression()
        xy_logit.coef_ = self.coefficients_XY()
        xy_logit.intercept_ = self.logit.intercept_
        #xy_logit.classes_ = self.logit.classes_
        return xy_logit

    def coefficients_XY(self):
        new_coef =  np.array([np.sum(self.pca.components_.T * self.logit.coef_,axis=1)])
        return new_coef

    def coefficients_HY(self):
        return self.logit.coef_

    def pca_transform(self, X):
        return self.pca.transform(X)

    def pca_inverse_transform(self, H):
        return np.array([self.pca.components_.T.dot(h) + self.pca.mean_ for h in H])

    def reconstruct(self, X):
        return self.pca_inverse_transform(self.pca_transform(X))

    def explained_variance(self, X):
        Xr = self.reconstruct(X)
        return r2_score(X, Xr)

class ICALogit(PCALogit):

    def __init__(self, n_components, penalty='none', C=1.0):
        self.n_components = n_components
        self.penalty = penalty
        self.C = C

    def get_params(self, deep=True):
        return {'n_components':self.n_components, 'penalty':self.penalty, 'C':self.C}

    def fit(self, Xlogit, Ylogit, Xpca=False):
        self.pca = FastICA(self.n_components, max_iter=1000)
        self.logit = LogisticRegression(penalty=self.penalty, C=self.C, solver='saga',max_iter=10000)
        if not Xpca:
            Xpca = Xlogit
        self.pca.fit(Xpca)
        Hlogit = self.pca_transform(Xlogit)
        self.logit.fit(Hlogit, Ylogit)
        self.fitted = True

        xylogit = self.get_xy_logit()
        self.coef_ = xylogit.coef_
        self.intercept_ = xylogit.intercept_
        return self

def test():
    random_seed = 2
    np.random.seed(random_seed)
    mean = np.zeros(10)
    cov = datasets.make_spd_matrix(len(mean), random_state=random_seed)
    X = np.random.multivariate_normal(mean, cov, 4000)
    X = StandardScaler().fit_transform(X) #  standardize : MEAN=0, and STD=1 / VAR=1
    y = np.array([1.0 if x[1] > .6 else 0.0 for x in X])


    pca_logit = PCALogit(n_components="mle")
    pca_logit.fit(Xpca=X, Xlogit=X, Ylogit=y)
    pca_logit.explained_variance(X)
    print(pca_logit.pca.n_components_)
    ypred = pca_logit.predict_proba(X)
    xy_logit = pca_logit.get_xy_logit()
    ynpred = xy_logit.predict_proba(X)

    print(1 - sum(abs(ypred - ynpred)) / len(ypred) == 1)


#test()
