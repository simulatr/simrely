# -*- coding: utf-8 -*-
"""
    simrel
    ~~~~~~~~~~~
    This simrel library can be used to simulate a linear model data with 1, 2, or many response variables.
    With few parameters, simrel can simulate data with wide range of properties. The parameters will be discussed
    more in discription of relevant classes
    :copyright: (c) 2018 by Raju Rimal
    :license: MIT, see LICENSE for more details.

"""

from .utilities import *


# noinspection PyUnresolvedReferences
class Simrel(object):
    """Base class only containing common parameters and methods to compute common
    properties

    """

    def __init__(self, **kwargs):
        """Take parameters from the initiated object and set some of those as an
        attributes of the object itself. These attributes will also be used by
        some of the methods for computing properties of simrel object
        
        Parameters:
        -------------
        **kwargs :
            All the possible parameters values accepted by their name. The individual child class will discuss on the arguments in details.

        """

        args = {
            'nobs': None,  # Number of observations
            'npred': None,  # Number of predictors
            'nrelpred': None,  # Number of relevant predictors
            'relpos': None,  # Position of relevant predictor components
            'gamma': None,  # Decay factor of eigenvalue of predictor
            'rsq': None,  # Coefficient of determination
            'sim_type': None,  # Type of simulation: univariate, bivariate, multivariate
        }
        for key, value in args.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_eigen(self, predictor=True):
        """Compute eigenvalues based on the ``gamma`` or ``eta`` parameters using following expression
            
            For predictor:

            math:: \lambda_i = e^{-\gamma(i - 1)}, \gamma >0 \text{ and } i = 1, 2, \ldots, p

            For response:

            math:: \kappa_j = e^{-\eta(j - 1)}, \eta >0 \text{ and } j = 1, 2, \ldots, m
        
        predictor : 
            Computes eigenvalues of predictor is true using ``gamma`` paramters else computes eigenvalues corresponding to response using ``eta`` parameters, defaults to True ('predictors')
        
        Returns
        -------
        An array
            Eigenvalues equals to the length of number of predictor/response

        """

        if predictor is True:
            vec_ = range(1, self.npred + 1)
            fctr_ = self.gamma
        else:
            vec_ = range(1, self.nresp + 1)
            fctr_ = self.eta
        eigen = np.exp([-fctr_ * float(p_) for p_ in vec_]) / np.exp(-fctr_)
        return eigen

    def get_sigmaz(self):
        """Compute variance-covariance matrix of latent components of predictors
        
        Returns
        -------
        An array
            A diagonal matrix (as array data-type) with eigenvalues of predictors in its diagonal

        """

        try:
            out = np.diag(self.eigen_x)
        except AttributeError:
            self.get_eigen()
            out = np.diag(self.eigen_x)
        return out

    def get_sigmaw(self):
        """Compute variance-covariance matrix of latent components of response
        
        Returns
        -------
        An array
            A diagonal matrix (as array data-type) with eigenvalues of responses in its diagonal

        """

        try:
            out = np.diag(self.eigen_y)
        except AttributeError:
            self.get_eigen(predictor=False)
            out = np.diag(self.eigen_y)
        return out

    def get_sigmazinv(self):
        """Compute the inverse of ``sigmaz`` (eigenvalue matrix of predictors) just by taking inverse of eigenvalues
        
        Returns
        -------
        A numpy array
            Diagonal matrix in an array data type is returned as an inverse of ``sigma_z``

        """

        try:
            out = np.diag(1 / self.eigen_x)
        except AttributeError:
            self.get_eigen(predictor=True)
            out = np.diag(1 / self.eigen_x)
        return out

    def get_sigma(self):
        """Concatinate the variance and variance-covariance matrices
        
        Returns
        -------
        An array
            An array as a variance-covariance matrix of joint distribution of responses and predictors

        """

        if not hasattr(self, 'sigma_zw'):
            self.get_sigma_zw()
        if not hasattr(self, 'sigmaw'):
            self.get_sigmaw()
        if not hasattr(self, 'sigmaz'):
            self.get_sigmaz()
        out = np.hstack((
            np.vstack((self.sigma_w, self.sigma_zw.reshape((-1, 1)))),
            np.vstack((self.sigma_zw, self.sigma_z))
        ))
        return out

    def get_relpred(self):
        """Computes (identifies) the position of relevant and irrelevant positions of predictor variables. Since the number of relevant predictors are given in argument ``q``, it will use position specified in ``relpos`` and sample remaining position from total availiable position to fill ``q``.
        
        Returns
        -------
        A dictionary
            With two elements: ``irrel`` contains a list of irrelevant positon and ``rel`` contains list of list of relevant positions corresponding to each response variables (components)

        """

        predpos_ = predpos(self.npred, [self.nrelpred], [self.relpos])
        irrelpred = list(predpos_['irrel'])
        relpred = [list(x) for x in predpos_['rel']]
        return dict(irrel=irrelpred, rel=relpred)

    def get_data(self, data_type):
        """Uses ``simulate`` utility function to simulate data
        
        Parameters:
        -------------

        data_type : {character}
            Can take values: ``train``, ``test`` or ``both``. If only ``train`` is specified will only simulate training samples even if number of test observation is specified in the argument

        Returns
        -------

        A dictionary with pandas DataFrame
            The dictionary contains two elements at most: ``train`` and ``test`` each as pandas DataFrame. The DataFrame contains ``nobs`` (for training) and ``ntest`` (for test) samples observation with columns with names ``Y1``, ``Y2`` so on for response and ``X1``, ``X2`` and so on for predictors

        """

        sigma = self.sigma
        # Check for positive definite of sigma
        nobs = self.nobs
        npred = self.npred
        nresp = self.nresp
        rotation_x = self.rotation_x
        rotation_y = self.rotation_y
        mu_x = self.mu_x
        mu_y = self.mu_y
        ntest = self.ntest
        out = dict()
        if data_type in ['train', 'both']:
            out['train'] = simulate(nobs, npred, sigma, rotation_x, nresp, rotation_y, mu_x, mu_y)
        if data_type in ['test', 'both'] and ntest is not None:
            out['test'] = simulate(ntest, npred, sigma, rotation_x, nresp, rotation_y, mu_x, mu_y)
        return out


class Unisimrel(Simrel):
    """This class is responsible for uniresponse simulation, i.e. with one response variable. This is heaviely dependent of the properties and methods of its parent class. The parameters required for this class are as follows:

    PARAMETERS
    ------------

    nobs: integer
        100 (default) Number of training observations to simulate

    npred: integer
        20 (default) Number of predictor variables

    nrelpred: a list
        7 (default) Number of relevant predictor the the response variable

    relpos: a list
        [1, 3, 4, 6] (default) Position of relevant components for the response

    gamma: float
        0.8 (default) Exponential decay factor of eigenvalues corresponding to predictors

    rsq: list
        0.8 (default) Coefficient of determination

    sim_type: string
        'univariate' (default) Can only take this value for this class

    ntest: integer
        None (default) Number of test observations

    mu_x: list
        None (default) A list of average values for each predictor variable

    mu_y: list
        None (default) A list with average value for the response. In this case it will be a list  of one element

    RETURNS
    --------

    class
        A simrel class

    """

    def __init__(self, nobs=100, npred=20, nrelpred=7, relpos=(1, 3, 4, 6),
                 gamma=0.8, rsq=0.9, sim_type='univariate', **kwargs):
        """This function initilize all the arguments and also compute properties when the object is initiated. However this will not simulate the data. To simulate the data from using the computed properties, one need to use ``get_data`` method.

        """

        # Assigning Attributes which might get replaced when calling parent constructor
        self.mu_x = None
        self.mu_y = None
        self.ntest = None
        self.nresp = 1

        # Calling for parent constructor
        Simrel.__init__(self, nobs=nobs, npred=npred, nrelpred=nrelpred, rsq=rsq,
                        relpos=relpos, gamma=gamma, nresp=self.nresp, sim_type=sim_type, **kwargs)

        # Assigning more attributes to unisimrel class
        self.eta = 1

        # Lets start computing different properties
        self.rotation_y = None
        self.eigen_x = self.get_eigen(predictor=True)
        self.eigen_y = self.get_eigen(predictor=False)
        self.sigma_z = self.get_sigmaz()
        self.sigma_zinv = self.get_sigmazinv()
        self.sigma_w = self.get_sigmaw()
        self.sigma_zw = self.get_sigma_zw()
        self.sigma = self.get_sigma()
        relpred_ = self.get_relpred()
        self.relpred = relpred_['rel']
        self.irrelpred = relpred_['irrel']
        self.rotation_x = self.rotate_pred()
        self.beta_z = self.get_beta_z()
        self.beta = self.get_beta()
        self.beta0 = self.get_beta0()
        self.rsq_y = self.get_rsq_y()
        self.minerror = self.get_minerror()

    def get_sigma_zw(self):
        """Compute the covariance between principal components of predictors and response variable satisfying all the input value of ``relpos``, ``rsq``, ``npred`` and ``gamma``
        
        Returns
        -------

        A numpy array
            The array with dimension of ``nresp`` (1) by ``npred`` containing non-zero elements at the position of relevant components and zero at other places.

        """

        if not hasattr(self, 'eigen_x'):
            self.get_eigen(predictor=True)
        sigma_zw = get_cov(pos=self.relpos, rsq=self.rsq, eta=self.eta,
                           p=self.npred, lmd=self.eigen_x)
        return sigma_zw

    def rotate_pred(self):
        """A rotation matrix for predictor is generated. This rotation matrix is used as a eigenvector matrix which will rotates the principal components to obtain predictor variables.
        
        Returns
        -------

        A numpy array
            The arrary will of of size ``npred`` times ``npred``. 

        """

        relpred = self.relpred
        npred = self.npred
        irrelpred = self.irrelpred
        out = np.identity(npred)
        rot_rel = [get_rotate(x) for x in relpred]
        rot_irrel = get_rotate(irrelpred)
        out[[[x] for x in irrelpred], [irrelpred]] = rot_irrel
        for k in range(len(relpred)):
            out[[[x] for x in relpred[k]], [relpred[k]]] = rot_rel[k]
        return out

    def get_beta_z(self):
        """Regression coefficients corresponding to principal components.
        
        Returns
        -------

        A numpy array
            The array contains non-zero regression coefficints where the components are relevant and zero otherwise

        """

        return np.matmul(self.sigma_zinv, self.sigma_zw)

    def get_beta(self):
        """Regression coefficients corresponding to the simulated data
        
        Returns
        -------

        A numpy array
            An array of true regression coefficient of size equals to ``npred``.

        """

        return np.matmul(self.rotation_x, self.beta_z)

    def get_beta0(self):
        """Intercept term
        
        Returns
        -------

        An array
            The array contains the intercept terms of the regression model. The default is zero if both ``mu_y`` and ``mu_x`` are None. In this particular class, the array only contains one element

        """

        beta0 = np.zeros((self.nresp,))
        beta = self.beta
        beta0 = beta0 + self.mu_y if self.mu_y is not None else beta0
        beta0 = beta0 - np.matmul(beta.T, self.mu_x) if self.mu_x is not None else beta0
        return beta0

    def get_rsq_y(self):
        """Compute the coefficient of determination from ``beta_z`` and ``sigma_zw``. This is the true coefficient of simulated data which will resembles input ``rsq`` parameter
        
        Returns
        -------

        An array
            The array contains single element containing coefficient of determination for the response variable

        """

        return np.matmul(self.beta_z.T, self.sigma_zw)

    def get_minerror(self):
        """Minimum error of the model, for this uniresponse simulation it is the difference between total variation in response ``sigma_w`` and coefficient of determination ``rsq_y``

        Returns
        -------

        An array
            For this uniresponse simulation it only contains one element

        """

        return self.sigma_w - self.rsq_y
