from itertools import accumulate
import numpy as np
import scipy as sp
import sys
sys.path.append('..')
from utils import *


#rng = np.random.default_rng()

class DataGenerator:

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    @staticmethod
    def check_chol(U, A):
        assert np.isclose(U.T.dot(U), A).all(), "Something wrong as Cholesky decomposition did not produce UtU = A"


    def rcorrmatrix(self, n, k, diag):
        A = self.rng.normal(size=[n, k])
        res = A.dot(A.T) + np.eye(n)*diag
        assert is_pos_def(res), 'Resulting matrix is not positive definite'
        return res


    def ac_errors(self, n, coef, var):
        var = var*(1-coef**2)
        res = self.rng.normal(size=n, scale=np.sqrt(var))
        res = np.array(list(accumulate(res, lambda x, y: coef*x + y)))
        return res


    def generate_true_X(self, shape, variance=1):

        if isinstance(variance, int):
            variance = np.sqrt(variance)
            X = self.rng.normal(loc=0, scale=variance, size=shape)
        elif isinstance(variance, dict):
            X = self.rng.normal(loc=0, scale=1, size=shape)
            variance['n'] = shape[1]
            variance = self.rcorrmatrix(**variance)
            U = sp.linalg.cholesky(variance)
            self.check_chol(U, variance)
            X = X.dot(U)
        elif len(variance.shape) == 1:
            variance = np.sqrt(variance)
            X = self.rng.normal(loc=0, scale=variance, size=shape)

        self.true_X = X

    def generate_coefs(self, magnitude, spread, prop_neg=0.5):
        size = self.true_X.shape[1]
        coefs = self.rng.normal(loc=magnitude, scale=magnitude*spread, size=size)
        coefs = -(self.rng.binomial(1, prop_neg, size=size)*2 - 1) * coefs
        self.coefs = coefs

    def generate_Y(self, variance=1, autocorr=0):
        #print(variance)
        if autocorr == 0:
            errors = self.rng.normal(loc=0, scale=np.sqrt(variance), size=self.true_X.shape[0])
        else:
            errors = self.ac_errors(self.true_X.shape[0], autocorr, variance)
        Y = self.true_X.dot(self.coefs)
        self.Y = Y + errors
        self.errors = errors


    def generate_observed_X(self, coef=0.6):
        stdX = np.std(self.true_X, axis=0)
        std_noise = np.sqrt(1/coef**2 - 1)*stdX
        noise = self.rng.normal(size=self.true_X.shape, scale=std_noise)
        self.obs_X = self.true_X + noise


    def generate_bad_features(self, n=1, prop=0.005):
        if isinstance(prop, list):
            prop = np.array(prop)
        self.choice = self.rng.choice(self.true_X.shape[1], n)
        parent = self.true_X[:, self.choice]
        parentstd = np.std(parent, axis=0)
        parentmean = np.mean(parent, axis=0)
        features = parent * prop + (1-prop)*self.rng.normal(size=parent.shape, loc=parentmean, scale=parentstd)
        self.obs_X = np.c_[self.obs_X, features]


    def from_dict(self, ddict):
        shape = ddict['shape']
        self.generate_true_X(shape, **ddict['gen_true_X'])
        self.generate_coefs(**ddict['gen_coefs'])
        self.generate_Y(**ddict['gen_Y'])
        self.generate_observed_X(**ddict['gen_obs_X'])
        if ddict.get('gen_bad_features'):
            self.generate_bad_features(**ddict['gen_bad_features'])

