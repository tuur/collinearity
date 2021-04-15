#!/usr/bin/env python

"""Logistic regression implementation in pytorch, with non-negativity constraints on coefficients, and coefficient order constraints """

__author__      = "Tuur Leeuwenberg"
__email__ = "A.M.Leeuwenberg-15@umcutrecht.nl"

import numpy as np
import torch.nn as nn
import torch, time
import torch.optim as optim
from torchcontrib.optim import SWA
import random
from lib.CorrelationGuidedDropout import CorrelationGuidedDropout
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class ConstrainedLogisticRegression(nn.Module):

    def __init__(self, dropout_ratio=0.0, verbose=0, positive_coef=[],ordered_coefficients=[], order_C=[], L1_C=0, L2_C=0, corr_guided_dropout=False,fit_intercept=True, hinge_type='squared', LAE_C=0, LAE_h=0, nn_C=0, classes_=[[1]], unlab_X=[],streamed_loss=[]):
        """Constructs a logistic regression"""
        super(ConstrainedLogisticRegression, self).__init__()
        self.dropout_ratio=dropout_ratio # percentage of dropout during training
        self.corr_guided_dropout = corr_guided_dropout
        self.verbose=verbose # verbosity of intermediate printing
        self.positive_coef = positive_coef # list of integers : dimensions in the input for which the coefficients should constrained >= 0
        self.ordered_coefficients = ordered_coefficients
        self.order_C=order_C
        self.L1_C=L1_C # inverse L1 regularization hyperparameter (Lasso)
        self.L2_C=L2_C # inverse L2 regularization hyperparameter (Ridge)
        self.fit_intercept=fit_intercept
        self.hinge_type=hinge_type
        self.LAE_C=LAE_C
        self.LAE_h=LAE_h
        self.nn_C=nn_C
        self.classes_=classes_
        self.unlab_X=unlab_X
        self.streamed_loss=streamed_loss

    def get_params(self, deep=True): # required function of SKLearn interface; params needed to create a similar object
        return {'streamed_loss':self.streamed_loss,'unlab_X':self.unlab_X, 'fit_intercept':self.fit_intercept, 'dropout_ratio': self.dropout_ratio, 'corr_guided_dropout':self.corr_guided_dropout, 'verbose': self.verbose, 'positive_coef': self.positive_coef, 'L1_C': self.L1_C, 'L2_C': self.L2_C, 'ordered_coefficients':self.ordered_coefficients, 'order_C':self.order_C, 'hinge_type':self.hinge_type, 'LAE_C':self.LAE_C, 'LAE_h':self.LAE_h, 'nn_C':self.nn_C, 'classes_':self.classes_}

    def set_params(self, **params): # required function of SKLearn interface
        for param, value in params.items():
            exec('self.'+str(param)+'='+str(value))
        return self

    def transitive_closure(self, a): # takes the transitive closure over a list of pairs (here: a)
        closure = set(a)
        while True:
            new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)
            closure_until_now = closure | new_relations
            if closure_until_now == closure:
                break
            closure = closure_until_now
        return closure

    def build(self, X): # build the model
        self.positive_coef = set(self.positive_coef)
        self.positive_weights = set(self.positive_coef)
        if type(self.order_C) == float:
            self.ordered_coefficients = {self.order_C:self.ordered_coefficients[i] for i in range(len(self.ordered_coefficients))}
        else:
            self.ordered_coefficients = {self.order_C[i]:self.ordered_coefficients[i] for i in range(len(self.order_C))}

        self.ordered_coefficients = {C: self.transitive_closure(order) for C,order in self.ordered_coefficients.items()}
        self.soft_ordered_lowers, self.soft_ordered_uppers = {C:[i for i,_ in order] for C,order in self.ordered_coefficients.items()}, {C:[j for _,j in order] for C,order in self.ordered_coefficients.items()}
        self.x_dim = X.shape[1]
        self.unconstrained_coef = set(range(self.x_dim)).difference(self.positive_coef)
        self.dropout = nn.Dropout(self.dropout_ratio) if not self.corr_guided_dropout else CorrelationGuidedDropout(p=self.dropout_ratio, correlation_matrix=np.corrcoef(X.T))
        self.sigmoid = nn.Sigmoid()
        if self.LAE_C > 0 and self.LAE_h > 0:
            self.in_to_h = nn.Linear(self.x_dim, self.LAE_h, bias=False)
            self.h_to_in = nn.Linear(self.LAE_h,self.x_dim)
            self.h_to_y = nn.Linear(self.LAE_h, 1, bias=self.fit_intercept)
            self.linear = nn.Linear(self.x_dim, 1, bias=self.fit_intercept) # create x_dim parameters, and a bias term
        else:
            self.linear = nn.Linear(self.x_dim, 1, bias=self.fit_intercept) # create x_dim parameters, and a bias term
            self.enforce_positive_weight_constraints() # corrects all violated weights

    def enforce_positive_weight_constraints(self):# set all weights that should be non-negative but are smaller than 0, to 0
        self.linear.weight.data[0][list(self.positive_weights)] = self.linear.weight.data[0][list(self.positive_weights)].clamp_(min=0)

    def get_coef(self): # NOTE! the model parameters / weights are NOT always the same as the coefficients of the logisitic regression; the result of this function are the actual regression coefficients.
        if self.LAE_h > 0 and self.LAE_C > 0:
            return torch.sum(self.in_to_h.weight.T * self.h_to_y.weight, axis=1).view(1,self.x_dim) # product of the transposed linear transformation into H, and the regression from H to y
        else:
            return self.linear.weight

    def get_intercept(self):
        if self.LAE_h > 0 and self.LAE_C > 0:
            return self.h_to_y.bias
        else:
            return self.linear.bias

    def set_intercept(self, new_intercept):
        self.linear.bias = torch.nn.Parameter(torch.Tensor(new_intercept))
        self.intercept_ = new_intercept
        return self

    def forward(self, X, prob=True):
        X_dropped_out = self.dropout(X)
        coefficients = self.get_coef()
        linear_predictor = X_dropped_out.mm(coefficients.T) + (self.get_intercept() if self.fit_intercept else 0.0)
        return self.sigmoid(linear_predictor) if prob else linear_predictor

    def reconstruct_X(self, X):
        return self.h_to_in(self.in_to_h(self.dropout(X)))

    def get_loss(self, y_true, y_pred, hinge_type='squared'):
        binary_cross_entropy = nn.BCEWithLogitsLoss()
        CE_loss = binary_cross_entropy(y_pred.flatten(), y_true.flatten()).mean()
        loss = CE_loss
        if self.L1_C or self.L2_C or len(self.ordered_coefficients) > 0 or (self.nn_C > 0 and len(self.positive_coef)) > 0:
            l1_crit, l2_crit = nn.L1Loss(), nn.MSELoss()
            coefficients = self.get_coef()
            if self.L1_C:
                L1_loss = l1_crit(coefficients, torch.zeros(coefficients.shape)) / self.L1_C if self.L1_C else 0.0  # L1 regularization
                loss = loss + L1_loss
            if self.L2_C:
                L2_loss = l2_crit(coefficients,torch.zeros(coefficients.shape)) / self.L2_C if self.L2_C else 0.0 # L2 regularization
                loss = loss + L2_loss
            if len(self.ordered_coefficients) > 0:
                for order_C in self.ordered_coefficients:
                    loss = loss + (1.0/order_C) * self.get_mean_hinge_loss(coefficients, self.soft_ordered_lowers[order_C], self.soft_ordered_uppers[order_C], hinge_type=hinge_type)
            if self.nn_C > 0 and len(self.positive_coef) :
                coefficients_that_should_be_non_negative = self.get_coef()[0][[list(self.positive_weights)]]
                nn_loss = torch.pow(coefficients_that_should_be_non_negative.clamp(max=0), 2).mean()
                loss = loss + (1.0 / self.nn_C) * nn_loss
        return loss

    def get_lae_loss(self, x_batch, x_rec):
        mse = nn.MSELoss()
        return mse(x_batch, x_rec) / self.LAE_C

    def get_mean_hinge_loss(self, coeffs, lowers, uppers, hinge_type='squared'):
        lowers = coeffs[0][lowers]
        uppers = coeffs[0][uppers]
        hinge = (lowers-uppers).clamp(min=0)
        if hinge_type == 'squared':
            l = torch.pow(hinge, 2).mean()
        if hinge_type == 'exp':
            l = torch.exp(hinge).mean()
        else:
            l = hinge.mean()
        return l


    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        preds = self.forward(X, prob=True)
        proba_out = np.array([[1.0-p, p] for p in preds.detach().numpy()],dtype='float64')
        return proba_out

    def predict_lp(self, X):
        return self.forward(X, prob=False).detach().numpy()


    def get_labeled_and_unlabeled_indices(self, y):
        unlabeled_indices = list(np.argwhere(np.isnan(y))[0])
        labeled_indices = [i for i in range(len(y)) if not i in unlabeled_indices]
        return labeled_indices, unlabeled_indices


    def fit(self, X, y, num_epochs=1000, batch_size=1000000, wd=0, verbose=0, lr=0.05, gradient_norm_clip=2, early_stopping=250, fix_intercept_after_n_epochs=False):
        t0 = time.time()
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.Tensor(y)
        self.build(X)
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd, amsgrad=True)

        labeled_indices, unlabeled_indices = self.get_labeled_and_unlabeled_indices(y)
        if len(self.unlab_X)> 0:
            self.unlab_X = torch.Tensor(self.unlab_X)

        size_X = len(X)
        mean_str_losses = []
        X_indices = labeled_indices

        lowest_loss, patience = np.inf, early_stopping
        for e in range(num_epochs):

            streaming_loss = 0
            batch_indices = list(range(0,size_X,batch_size))
            random.shuffle(X_indices)
            for i in batch_indices:
                optimizer.zero_grad()
                x_batch = X[X_indices[i:i+batch_size]]
                y_batch = y[X_indices[i:i+batch_size]]
                y_pred = self.forward(x_batch, prob=False)
                ml_loss = self.get_loss(y_batch, y_pred, hinge_type=self.hinge_type)

                if self.LAE_C > 0 and self.LAE_h > 0:
                    x_rec = self.reconstruct_X(x_batch)
                    if len(self.unlab_X) > 0:
                        unlab_rec = self.reconstruct_X(self.unlab_X)
                        lae_unlab_loss = self.get_lae_loss(self.unlab_X, unlab_rec)
                        lae_lab_loss = self.get_lae_loss(x_batch, x_rec)
                        lae_loss = lae_unlab_loss + lae_lab_loss
                    else:
                        lae_loss = self.get_lae_loss(x_batch, x_rec)
                    loss = ml_loss + lae_loss
                else:
                    loss = ml_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), gradient_norm_clip)

                if fix_intercept_after_n_epochs and e > fix_intercept_after_n_epochs:
                    self.linear.bias.grad.data = torch.Tensor([0.0])
                optimizer.step()
                if len(self.positive_weights) > 0:
                    if self.LAE_C ==0 and self.nn_C ==0:
                        self.enforce_positive_weight_constraints()
                    if self.LAE_C > 0 and self.LAE_h > 0 and self.nn_C == 0:
                        print('WARNING: when using LAE with positive coefficients, nn_C (the inverse non-negativity constraint importance) should be set!')
                        exit()

                streaming_loss += loss.sum().detach().item()
            mean_str_loss = streaming_loss / size_X
            if early_stopping and mean_str_loss < lowest_loss:
                lowest_loss, patience = mean_str_loss, early_stopping
            patience -= 1
            if early_stopping and patience < 0:
                print('break',e)
                break

            if verbose > 1:
                print('e', mean_str_loss)
            mean_str_losses.append(mean_str_loss)
        if self.verbose:
            plt.plot(mean_str_losses)
            print('plotting loss')
            plt.savefig(self.verbose + '/loss.png')
            plt.close()
            plt.cla()
            plt.clf()
            print('t',round(time.time()-t0,1),'s')


        print('loss:', mean_str_losses[-1], mean_str_losses[-2])
        self.streamed_loss = mean_str_losses
        self.eval()
        self.coef_ = self.get_coef().detach().numpy()
        self.intercept_ = self.get_intercept().detach().numpy() if self.fit_intercept else 0.0
        return self




def test(lae=1):
    n = 10
    x_dim = 6
    positive_coef = [0,2]
    coefficient_order =  {1: [(0,1)], 100:[(2,3)]}
    x = torch.ones((n, x_dim))
    y = torch.zeros(n)

    model = ConstrainedLogisticRegression(positive_coef=positive_coef, ordered_coefficients=coefficient_order,verbose=1, LAE_C=lae, LAE_h=2, nn_C=0.01)
    model = model.fit(x, y, lr=0.01, num_epochs=1000)
    print('coefficients', model.coef_)
    print('intercept', model.intercept_)
    print('pos_c',model.positive_coef)
    print('pos_w',model.positive_weights)
    #print('W',model.linear.weight)


# test(2)

