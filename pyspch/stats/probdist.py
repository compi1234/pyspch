# -*- coding: utf-8 -*-
"""

Defining classes of pdf's and probability distributions commonly used in GENERATIVE models for statistical machine learning and classification such as HMM's, Bayesian Classification, Decision Trees, ...

The API is modeled after sklearn (i.c. _BaseNB, the Naive Bayesian classifiers)
The models are built from scratch or are wrappers around existing sklearn classifiers and/or distribution models.
 
Following Models are currently supported:

- Gaussian :    wrapper around sklearn GaussianNB
- Discrete :    discrete densities with probability estimates in the model
                  (counterpart of sklearn CategoricalNB that uses log_probs instead)
- GMM :         Gaussian Mixture Models
- Prob :        A pass through for probabilistic features (both likelihoods and posteriors)
                   for purposes of classification or HMM recognition 

Data and Classes
----------------
    X: ndarray (n_samples, n_features)   IS ALWAYS 2D
    y: ndarray (n_samples, )             IS ALWAYS 1D
        y is either integers (unnamed classes) or strings (named classes) 
        classes_ holds the class names, it is either inferred from training data or can be set

Attributes
----------
    n_classes     number of classes
    n_features    number of features   (equivalent to n_features_in_ )
    classes_      ndarray of class names   
    class_count_  ndarray of class counts from training data   
    class_prior_  ndarray (n_classes,) of class priors from training
    priors        enforced priors, overriding class_prior_ if not None
    
Methods existing in sklearn
---------------------------
    .fit(X,y)               train from examples   
    .predict_proba(X)       computes class probabilities of feature vectors   
    .predict_log_proba(X)   computes class log probabilities of feature vectors   
    .predict(X)             computes predicted classes from X   

    .predict_ftr_prob(X)    computes feature likelihoods for X
    .predict_ftr_log_prob() computes feature log-likelihoods for X
    [.predict_prob(X)       computes feature likelihoods for X  -- will be deprecated]
    [.predict_log_prob()    computes feature log-likelihoods for X -- will be deprecated]
    
Additional Methods
------------------
    .print_model()          prints the key attributes of the model for each feature or per class
    .plot_model()           plots the distribution for each feature or per class
    

History:
=========
11/01/2022: added 'style' option for printing of logprobs in class=Discrete, module=print_model()
20/03/2023: added 'llscore', 'bic' and ''ll_and_bic' methods to class GMM
21/03/2023: bug_fix in GMM.print_model() avoid sqrt() of negative correlation coefficients
10/04/2024: big_fix in print_model() for discrete observation models, now correctly printing stored labels
18/04/2024: bug_fix in .fit() method for discrete models
"""
##################################################################################
### The code below is to avoid memory leaks in sklearn KMEANS on windows machines 
### "UserWarning: KMeans is known to have a memory leak on Windows with MKL, 
###   when there are less chunks than available threads. 
###   You can avoid it by setting the environment variable OMP_NUM_THREADS=2.
import os
os.environ["OMP_NUM_THREADS"] = '1'
##################################################################################

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB, CategoricalNB
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import norm
from .. import core as Spch

# reusable initialization routine for classes and priors
def _init_classes_(n_classes=None,classes=None,priors=None):
    ''' 
    infer n_classes, classes and priors
    optimally from given parameters
    '''
    # first determine the number of classes
    if n_classes is None:
        if (classes is None) and (priors is None): 
            return n_classes, classes, priors
        
        if classes is None:  n_classes = len(priors)
        else:                n_classes = len(classes)
            
    # now, give the classes a name
    if classes is None :  classes = np.arange(n_classes)
    else:                 classes = np.array(classes)
        
    # enforcing priors to be not None ?
    # if priors is None: priors = np.ones(n_classes,dtype='float64')/(n_classes) 
    return n_classes, classes, priors

def _posteriors_to_probs_(post,priors=None):
    '''
    converts posteriors to normalized likelihoods !!
    i.e. likelihoods sum to one (like posteriors do)
    this is done from two perspectives:  
        (i) it is impossible to get exact likelihoods from posteriors
        (ii) some arbitrary scaling is required, thus normalized values are a convenient and logical choice
    '''
    if priors is None: return(post)
    probs = post / priors
    probs = probs / np.sum(probs,axis=1,keepdims=True)
    return(probs)    

def _probs_to_posteriors_(probs,priors=None):
    '''
    converts likelihoods to posteriors
    '''
    if priors is None:  post = probs
    else:               post = probs * priors
    post = post / np.sum(post,axis=1,keepdims=True)
    return(post)

class Prob():
    '''
    An interface for direct feeds of probability densities or posteriors

    Attributes:
        style      str    style of input: "Probs"(Default), "logProbs", "Posteriors", "logPosteriors"
        priors     array with priors or None (default)
                    trained priors should be passed explicitly
    '''
    def __init__(self,style="Probs",priors=None):
        self.prob_style = style
        self.class_prior_ = priors
        
    def predict_ftr_prob(self,X):
        if self.prob_style == "Probs": return(X)
        elif self.prob_style == "logProb": return np.exp(X)
        elif self.prob_style == "Posteriors": return _posteriors_to_probs_(X,priors=self.class_prior_)
        elif self.prob_style == "logPosteriors": return _posteriors_to_probs_(np.exp(X),priors=self.class_prior_)
        
    def predict_ftr_log_prob(self,X):
        if self.prob_style == "logProbs": return(X)
        elif self.prob_style == "Probs": return Spch.logf(X)  
        elif self.prob_style == "Posteriors": return Spch.logf( _posteriors_to_probs_(X,priors=self.class_prior_) )    
        elif self.prob_style == "logPosteriors": return Spch.logf( _posteriors_to_probs_(np.exp(X),priors=self.class_prior_ )  )

    def predict_prob(self,X):
        return self.predict_ftr_prob(X)
    def predict_log_prob(self,X):
        return self.predict_ftr_log_prob(X)
        
    def predict_proba(self,X):
        if self.prob_style == "Posteriors": return(X)
        elif self.prob_style == "logPosteriors": return np.exp(X)
        elif self.prob_style == "Probs": return _probs_to_posteriors_(X,priors=self.class_prior_)
        elif self.prob_style == "logProbs": return _probs_to_posteriors_(np.exp(X),priors=self.class_prior_) 
        
    def predict_log_proba(self,X):
        if self.prob_style == "logPosteriors": return(X)
        elif self.prob_style == "Posteriors": return Spch.logf(X)      
        elif self.prob_style == "Probs": return Spch.logf( _probs_to_posteriors_(X,priors=self.class_prior_) )
        elif self.prob_style == "logProbs": return Spch.logf( _probs_to_posteriors_(np.exp(X),priors=self.class_prior_ ) )        
        


        
class Discrete():
    '''
    Probability Model for discrete densities for multiple classes, supporting multiple features
    This is very similar to sklearn categoricalNB
    
    The main difference is that this class works with feature_probs instead of feature_log_prob_'s
    It allows for hard zeros in the ML estimation

    Attributes:

        n_classes       int       number of classes
        n_features      int       number of features
        n_categories    array of shape (n_features,) categories for each feature
        classes_        array of shape (n_classes,)  known labels for categories
        feature_prob_   list of arrays of shape (n_classes,n_categories of feature)
        
    '''
    
    def __init__(self,alpha=1.0,feature_probs=None,n_categories=None,labels=None,classes=None,priors=None):
        ''' 
        initialization looking in order at feature_probs, labels, n_categories
        to determine model dimensions
        
        if labels is None, we will assume that all labels are numerical values in range of n_categories
        '''
        
        self.feature_prob_ = feature_probs
        self.alpha = alpha
        self.class_prior_ = None
        self.labels = labels
        self.n_classes = None

        # initialization from feature_probs
        if feature_probs is not None: 
            self.n_features = len(self.feature_prob_)
            self.n_categories = np.zeros((self.n_features,),'int32')
            self.n_classes = self.feature_prob_[0].shape[0]
            for i in range(self.n_features):
                self.n_categories[i] = self.feature_prob_[i].shape[1]               
        elif labels is not None: #initialize from labels or n_categories
            # check that labels is list of lists
            if not all(isinstance(elem, list) for elem in labels):
                self.labels = [labels]
            else: self.labels = labels
            self.n_features = len(self.labels)
            self.n_categories = np.zeros((self.n_features,),'int32')
            for i in range(self.n_features):
                self.n_categories[i] = len(self.labels[i])
        elif n_categories is not None: #initialize from n_categories
            self.n_categories = np.asarray(n_categories)
            self.n_features = len(n_categories)
        else:
            error("Discrete(): can not initialize; at least feature_probs, labels or n_categories needs to be specified")
           
        self.n_classes, self.classes_, self.priors = _init_classes_(n_classes=self.n_classes,classes=classes,priors=priors)   

            
    def lbl2indx(self,Xlabels):
        """
        converts observation labels to array of indices
        
        input and output should be arrays of shape (n_samples,n_features)
                                                or (n_features,)
        
        """
        return_1D = False
        if Xlabels.ndim ==1 : return_1D = True
        Xlabels = Xlabels.reshape(-1,self.n_features)
        n_samples = Xlabels.shape[0]
        Xindx = np.zeros(Xlabels.shape,'int32')
        for j in range(self.n_features):
            lbls_list = list(self.labels[j])
            for i in range(n_samples):
                Xindx[i,j] = lbls_list.index(Xlabels[i,j])

        if return_1D:
            return(Xindx.flatten())
        else:
            return(Xindx)            
            
    def predict_ftr_prob(self,X):
        """
        computes state observation likelihoods for discrete desity model
        !!! X should be an array of samples (also if single sample)
        !!! always returns a 2D array
        
        X:  array of shape (n_features,) or (n_samples,n_features)
        returns: array of shape (n_states,) or (n_samples,n_states)
        """
        return_1D = False
        if X.ndim == 1: return_1D = True
        X = X.reshape(-1,self.n_features)
        n_samples = X.shape[0]
        prob = np.ones((n_samples,self.n_classes))
        for j in range(self.n_features):
            featprobs = self.feature_prob_[j][:,X[:,j]].T
            prob = prob * featprobs
        return prob
    
    def predict_ftr_log_prob(self,X):
        return Spch.logf(self.predict_ftr_prob(X))

    def predict_prob(self,X):
        return self.predict_ftr_prob(X)
    def predict_log_prob(self,X):
        return self.predict_ftr_log_prob(X)
         
    def predict_proba(self,X):
        if self.priors is not None: priors = self.priors
        elif self.class_prior_ is not None: priors = self.class_prior_
        else: priors = 1.
        likeh = self.predict_ftr_prob(X)
        proba = likeh * priors
        return( proba / np.sum(proba,axis=1,keepdims=True) )

    def predict_log_proba(self,X):
        return Spch.logf(self.predict_ftr_prob(X))        
    
    def print_model(self,labels=None,style='lin'):
        """
        You can override the labels stored in the model with specifying labels
        If no labels are found, abstract indices will be displayed
        """
        if labels is None:
            labels = self.labels
        if labels is None: 
            labels = []
            for j in range(self.n_features):
                labels.append([str(i) for i in range(self.n_categories[j])])
        for j in range(self.n_features):
            print("++ Feature (%d) ++\n "%j)
            lbls = labels[j]
            if style == 'lin': ftr_prob = self.feature_prob_[j].T
            elif style == 'log': ftr_prob = Spch.logf(self.feature_prob_[j].T)
            featprob_df = pd.DataFrame(ftr_prob,columns=self.classes_,
                     index= ['P('+lbls[i]+'|.)' for i in range(0,self.n_categories[j])])
            display(featprob_df)

    def plot_model(self,figsize=(14,6)):
        barwidth = .2
        n_features = len(self.feature_prob_)
        f,ax = plt.subplots(1,n_features,figsize=figsize)
        for j in range(n_features):
            xs = np.arange(self.n_categories[j])
            for i in range(self.n_classes):
                ax[j].bar(xs+i*barwidth,self.feature_prob_[j][i,:],width=barwidth)
                ax[j].set_xticks([xx for xx in range(self.n_categories[j]) ])
            ax[j].legend(np.arange(self.n_classes))
            ax[j].set_title('Likelihoods for Feature %d'%j)

    def fit(self,X,y,floor=1.e-3,index=True):
        '''
        fit to discrete density observation for a set of observations and labels
        if index==True, then observations are assumed to be observation indices
        '''
        #print("sorry")
        if self.n_classes is None:  # you need to initialize from data
            self.classes = np.unique(y)
            self.n_classes = len(self.classes)
        if (index == True) or (self.labels is None):
            Xindx = X
        else: # expecting numeric labels in each category
            Xindx = self.lbl2indx(X)
        self.feature_prob_ = []
        self.counts = []
        n_obs, n_ftrs = Xindx.shape
        for j in range(self.n_features):
            XX = Xindx[:,j]
            counts = np.zeros((self.n_classes,self.n_categories[j]),dtype='int')
            for i in range(n_obs):
                counts[y[i],XX[i]] += 1
            sum_of_counts = np.sum(counts,axis=1).astype(float)
            self.feature_prob_.append(counts.astype(float)/sum_of_counts[:,np.newaxis])
            self.counts.append(sum_of_counts)
        return

        
class Gaussian(GaussianNB):
    """ Gaussian Estimator and Classifier, a wrapper around GaussianNB
    with added printing/plotting functionality

    Parameters
    ----------
    priors         array-like of shape (n_classes,)  default=None
        Prior probabilities of the classes. 
        If specified the priors are not adjusted according to the data.

    var_smoothing  float, default=1e-9
        Portion of the largest variance of all features that is added 
        to variances for calculation stability.
        
    Attributes
    ----------
    n_classes :    int,
        number of classes
    n_features:    int,
        number of features

    Attributes from GaussianNB
    --------------------------
    theta_ :       float,
        class means
    var_   :       float,
        class variances
    epsilon:       float64,
        added value to variance
        
    classes_ :     array of shape (n_classes,)
    class_prior_ : array, shape (n_classes,)
        probability of each class.
    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.   
           
    """               

    #
    # coding remark: 7/12/2021
    # using __init__() with additional parameters gives problems;
    # seems to compromise the __repr__() method of _BaseEstimator
    # as it expects all initialization parameters to be available under their name
    # in the class
    #
    #def __init__(self,var_smoothing=1.e-9,n_classes=None,classes=None,priors=None,n_features=1,mu=None,var=None):
    #

        
    def _validate(self):
        if (self.n_classes is None):
            raise ValueError("GAUSSIAN: No classes are defined yet")
        if (not hasattr(self,'theta_')) or (not hasattr(self,'var_')):
            raise ValueError("GAUSSIAN: NOT initialized yet")
            
    def init_model(self,mu=None,var=None,classes=None,class_prior=None):
        
        # initialize class means, n_classes and n_features_in_
        if (mu is None):
            if (classes is None):
                print("You must at least specify means to initialize a model")
                return
            else:  # initialize classes only
                #self.classes_ = np.array(classes)
                self.n_classes = len(classes)
                self.n_features_in_ = 1
                mu = np.zeros((self.n_classes,1))
        else:
            # initialize from model
            (self.n_classes,self.n_features_in_) = mu.shape
        self.theta_ = mu
        
        # initialize var from var (matrix), var(scalar) or default (1.0)
        if var is not None:
            if np.isscalar(var): var = var*np.ones(self.theta_.shape)
            self.var_ = var
        else:
            self.var_ = np.ones((self.n_classes,self.n_features_in_))
        
        # initialize class names and class priors
        if classes is None: self.classes_ = np.arange(self.n_classes)
        else:               self.classes_ = np.array(classes)
            
        if class_prior is None:
            self.class_prior_ = np.ones(self.n_classes)/self.n_classes
        else:
            self.class_prior_ = np.array(class_prior)
        
        
    def predict_ftr_prob(self,X):
        return np.exp(self.predict_log_prob(X))
        
    # sklearn computes the joint feature likelihood and class prior
    def predict_ftr_log_prob(self,X):
        return (self._joint_log_likelihood(X)-np.log(self.class_prior_))

    def predict_prob(self,X):
        return self.predict_ftr_prob(X)
    def predict_log_prob(self,X):
        return self.predict_ftr_log_prob(X)
    
    
    def print_model(self,precision=None):
        """ print the model """
        try: 
            if precision is not None:
                pd.set_option('display.precision',precision)
            df = pd.DataFrame(data={'prior':list(self.class_prior_),'mean':list(self.theta_),'var':list(self.var_)})
            display(df)
            #for k in range(0,self.n_classes):
            #    print("Class[%d] (%s) with prior=%.3f" % (k,self.classes[k],self.class_prior_[k]))
            #    print("-----------------------------------")
            #    df = pd.DataFrame(data={'mean':self.theta_[k],'var':self.var_[k]})
            #    print(df)
        except:
            print('Could not print the model')
            
    def plot_model(self):
        nclass, n_features = self.theta_.shape
        f,_ = plt.subplots(1,n_features,figsize=(14,5))
        ax = f.axes
        for j in range(n_features):
            for i in range(nclass):
                mu = self.theta_[i,j]
                sigma = np.sqrt(self.var_[i,j])
                xx = np.linspace(mu-2*sigma,mu+2*sigma,100)
                ax[j].plot(xx, norm.pdf(xx, mu, sigma))
            ax[j].legend(self.classes_)
            ax[j].set_title("Feature (%d)" %j)           
        

class GMM(BaseEstimator, ClassifierMixin):
    """ Gaussian Mixture Model Estimator and Classifier
    
    Attributes
    ----------
    
    n_classes :    int,
        number of classes
    n_components : int,
        number of components in each Mixture
    max_iter :     int, 
        maximum number of iterations in GMM updates
    tol :          float64,
        tolerance  for converging GaussianMixture
    
    class_prior_ : array, shape (n_classes,)
        probability of each class.
    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.   
        
    
    
    """   
    def __init__(self, n_components=1 ,classes=[0,1], max_iter=10, **kwargs):
        self.n_components = n_components
        self.max_iter = max_iter
        self.data_range_ = None
        self.classes = classes
        self.n_components=n_components
        self.n_classes = len(self.classes)
        self.class_count_ = np.zeros(self.n_classes,dtype='float64')
        self.class_prior_ = np.ones(self.n_classes,dtype='float64')/float(self.n_classes)

        
        gmm_kwargs=dict(covariance_type='diag',init_params='kmeans',random_state=1)
        gmm_kwargs.update(kwargs)
        self.gmm = [GaussianMixture(max_iter=self.max_iter,n_components=self.n_components, 
                        **gmm_kwargs) for k in range(0,self.n_classes)]   


        
    def _validate_params(self):
        if self.n_classes < 2:
            raise ValueError("Minimum Number of Classes is 2")
        if self.n_classes != len(self.gmm):
            raise ValueError("Initialized GMM shape(%d,_) does not match n_classes(%d) "
                % (len(self.gmm), self.n_classes) )
    
    def llscore(self, X, y):
        ''' 
        likelihood score per sample for given data
        '''
        ll = 0.
        for k in range(0,self.n_classes) :
            ll_k = self.gmm[k].score(X[y== self.classes[k],: ])
            ll += ll_k * sum(y==self.classes[k])
        return(ll/len(y))
    
    def bic(self,X,y):
        '''
        bic score for model given the data
        '''
        nparam = ((2*X.shape[1]+1)*self.n_components -1 ) * self.n_classes
        bic = -2*self.llscore(X,y)*len(y) + (np.log(X.shape[0])* nparam)
        return(bic)
    
    def ll_and_bic(self,X,y):
        '''
        bic score for model given the data
        '''
        ll = self.llscore(X,y)
        nparam = ((2*X.shape[1]+1)*self.n_components -1 ) * self.n_classes
        bic = -2*ll*len(y) + (np.log(X.shape[0])* nparam)
        return(ll,bic)
        
    def fit(self, X, y):
        """Fit GMM with EM for each class.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data
    
        y : numpy array, shape (n_samples,)
            Target values
        
        
        Returns
        -------
        self : returns an instance of self.
        """

        for k in range(0,self.n_classes):
            self.gmm[k]._check_parameters(X)

        classes = np.unique(y)
        if(len(classes) > len(self.gmm)):
            raise ValueError("More classes in data than in allocated GMMs")

        for k in range(0,self.n_classes) :
            selection = (y== self.classes[k])
            self.gmm[k].fit(X[selection,: ])
            self.class_count_[k] = np.sum(selection)     
              
        # keep the counts in the model for adaptation and incremental training purposes
        # keep the datarange for summary / plotting utilities
        self.class_prior_ = self.class_count_ / len(y)
        self.data_range_ = np.vstack((np.min(X,axis=0),np.max(X,axis=0)))
        
    def predict_ftr_log_prob(self,X):
        """ Log Likelihoods of  X  for each class
        
            Returns
            -------
            array, shape (n_samples, n_classes)
                Predicted likelihoods per class

        """
        Xprob = np.ndarray(dtype='float64',shape=(X.shape[0],self.n_classes))
        for k in range(0,self.n_classes):
            Xprob[:,k]= self.gmm[k].score_samples(X)      
        return Xprob

    def predict_ftr_prob(self,X):
        """ Compute Likelihoods per class
        """
        return np.exp(self.predict_ftr_log_prob(X))

    def predict_prob(self,X):
        return self.predict_ftr_prob(X)
    def predict_log_prob(self,X):
        return self.predict_ftr_log_prob(X)    
        
    def predict_log_proba(self, X, priors = None):
        """ 
        Computes the log posteriors (class probabilities) given the samples
        You can override the priors derived during training
        """
        return( np.log(self.predict_proba( X, priors) ) )
        
    def predict_proba(self, X, priors = None):
        """ 
        Computes the class probabilities (posteriors) given the samples
        You can override the priors derived during training
        """
        
        if priors is None:
            priors = self.class_prior_
            
        if (len(priors) != self.n_classes):
            raise ValueError("Dimensions of priors do not match number of classes")
            
        likelihoods = self.predict_ftr_prob(X)
        nsamples = X.shape[0]
        total_likelihood = np.zeros(nsamples)
        posteriors = np.zeros((nsamples,self.n_classes))
        for k in range(0,self.n_classes):
            likelihoods[:,k] =  priors[k]*likelihoods[:,k]
            total_likelihood += likelihoods[:,k]
        for k in range(0,self.n_classes):
            posteriors[:,k] = np.divide(likelihoods[:,k],total_likelihood)
                    
        return(posteriors)
    
    def predict(self, X,priors=None):
        indx = self.predict_proba(X,priors=priors).argmax(axis=1)
        return( [self.classes[k] for k in indx ] )

            
    def print_model(self):
        """ print the model """
        for k in range(0,self.n_classes):
            print("Class[%d] (%s) with prior=%.3f" % (k,self.classes[k],self.class_prior_[k]))
            print("-----------------------------------")
            if  self.gmm[k].means_.shape[1] == 1 :
                df = pd.DataFrame(data={'weights':self.gmm[k].weights_.reshape(-1), 
                                    'mean':self.gmm[k].means_.reshape(-1), 
                                    'cov':  (self.gmm[k].covariances_).reshape(-1)})
                print(df)
            else:
                for kk in range(self.n_components):
                    print( self.gmm[k].weights_[kk], self.gmm[k].means_[kk],(self.gmm[k].covariances_[kk]) ) 
            print("")
    #def print(self):
    #    print_model(self)
        
    def plot_ftr_prob(self):
        """ 
        plot the likelihood distributions 
        ... untested 
        """
        if( self.data_range_ == None ):
            raise ValueError("No data_range_ given for the plots")
        npts = 100
        ndim = len(self.data_range_,axis=1)
        if(ndim>2):
             raise ValueError("plot_prob only for 1-D or 2-D data")
        elif(ndim==2):
            x = np.linspace(self.data_range_[0,0],self.data_range_[1,0],num=npts)
            y = np.linspace(self.data_range_[0,1],self.data_range_[1,1],num=npts)
            xx,yy = np.meshgrid(x,y, sparse=True)
            z = self.predict_ftr_prob(np.vstack(xx,yy))
            h = plt.contourf(x,y,z)
            plt.show()
        else:
            x = np.linspace(self.data_range_[0,0],self.data_range_[1,0],num=npts)
            z = self.predict(np.vstack(x))
            h = plt.plot(x,z)
            plt.show()

            
#####################################
##### OBSOLETE
#####################################
class DiscreteDens_obsolete1(CategoricalNB):
    '''
    Probability Model for discrete densities for multiple classes, supporting multiple features based on categoricalNB
    
    Attributes from CategoricalNB
        classes_
        class_count_
        n_features_in
        category_count
        feature_log_prob_
        class_log_prior_
        alpha

    Attributes:

        n_classes       int       number of classes
        n_features      int       number of features
        n_categories    array of shape (n_features,) categories for each feature
        classes         array of shape (n_classes,)  known labels for categories
        feature_prob_   list of arrays of shape (n_classes,n_categories of feature)
        
    '''
    
    def __init__(self,feature_probs=None,labels=None,n_classes=None,classes=None,n_categories=None,n_features=None,priors=None,alpha=1.):
               
        self.n_features = n_features
            
        if feature_probs is None :
            print("ERROR(DiscreteDens): An emission probability matrix is required for initialization")
            exit(1)

        self.alpha = alpha
        self.feature_probs = feature_probs

            
        self.n_features = len(self.feature_probs)
        self.n_features_in_ = self.n_features
        self.n_categories = np.zeros((self.n_features,),'int32')
        self.n_classes = self.feature_probs[0].shape[0]
        for i in range(self.n_features):
            self.n_categories[i] = self.feature_probs[i].shape[1]   
            
        if priors is None:
            self.class_prior_ = np.ones(self.n_classes,dtype='float64')/(self.n_classes) 
        else: 
            self.class_prior_ = priors
            self.feature_log_prob_ = Spch.logf(_probs_to_posteriors_(self.feature_probs),priors=self.class_prior_)
        
        if labels is None:
            self.labels=[]
            for i in range(self.n_features):
                self.labels.append( np.array(['L'+str(i) for i in range(0,self.n_categories[i])]) )
        else:
            self.labels = labels
            
        if classes is None :
            self.classes = np.arange(self.n_classes)
        else:
            self.classes = classes        
        
        