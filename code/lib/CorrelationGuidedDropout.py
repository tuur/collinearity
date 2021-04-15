import torch.nn as nn
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_spd_matrix

class CorrelationGuidedDropout(nn.Module): # started from https://stackoverflow.com/questions/54109617/implementing-dropout-from-scratch
    def __init__(self, p: float = 0.5, correlation_matrix=[]):
        """ Dimension with high correlation are more likely to get dropped out, this way the resulting set is more likely to correlate less"""
        super(CorrelationGuidedDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.correlation_matrix = torch.Tensor(correlation_matrix)
        if len(self.correlation_matrix) > 0:
            mean_corrs= torch.mean(abs(self.correlation_matrix), axis=1)
            self.ppd = mean_corrs * self.p #torch.nn.functional.softmax(sum_corr, dim=0) * self.p * len(correlation_matrix) # mean prob across dimensions is self.p (as with normal dropout)

            self.binomial = torch.distributions.binomial.Binomial(probs=1-self.ppd)
        else:
            self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        if self.training:
            if len(self.correlation_matrix) > 0 :
                masking_pattern = self.binomial.sample()
                correction = (1.0 / (1-self.ppd))
            else:
                masking_pattern = self.binomial.sample(X.size())
                correction = (1.0/ (1-self.p))
            return X * masking_pattern * correction
        return X


def test():
    random_seed = 1
    np.random.seed(random_seed)
    mean = np.zeros(3)
    cov = make_spd_matrix(len(mean), random_state=random_seed)
    print('cov',cov)
    X = np.random.multivariate_normal(mean, cov, 500)
    X = StandardScaler().fit_transform(X) #  standardize : MEAN=0, and STD=1 / VAR=1
    print(X.shape)
    corr = np.corrcoef(X.T)
    dropout_normal = CorrelationGuidedDropout(0.5)
    dropout_corr = CorrelationGuidedDropout(0.5, correlation_matrix=corr)
    x0 = torch.Tensor(X[0])
    print('x',x0)
    print('x_dn',dropout_normal(x0))
    print('x_dc',dropout_corr(x0))

