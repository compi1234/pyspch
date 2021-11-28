# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:41:29 2019

@author: compi

"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
#from scipy.special import logsumexp


class DiscreteDens():
    '''
    Probability Model for discrete densities for multiple classes, supporting multiple features
    This is very similar to sklearn categoricalNB
    
    Important NOMENCLATURE used throughout:
         - emission probabilities: probabilities in the model
         - observation probabilities: probabilities of the observation stream for the respective states

    Attributes:

        n_classes       int       number of classes
        n_features      int       number of features
        n_categories    array of shape (n_features,) categories for each feature
        classes         array of shape (n_classes,)  known labels for categories
        feature_probs   list of arrays of shape (n_classes,n_categories of feature)
        
    '''
    
    def __init__(self,feature_probs=None,labels=None,classes=None,n_categories=None,n_features=None):
               
        self.n_features = n_features
            
        if feature_probs is None :
            print("ERROR(DiscreteDens): An emission probability matrix is required for initialization")
            exit(1)

        self.feature_probs = feature_probs
        self.n_features = len(self.feature_probs)
        self.n_categories = np.zeros((self.n_features,),'int32')
        self.n_classes = self.feature_probs[0].shape[0]
        for i in range(self.n_features):
            self.n_categories[i] = self.feature_probs[i].shape[1]           
        
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
            
    def lbl2indx(self,Xlabels):
        """
        converts observation labels to array of indices
        
        input and output have shape (n_samples,n_features)
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
    
    def predict_log_proba(self,X):
        """
        !!! X should be an array of samples (also if single sample)
        !!! always returns a 2D array
        computes state observation likelihoods for discrete desity model
        
        X:  array of shape (n_features,) or (n_samples,n_features)
        returns: array of shape (n_states,) or (n_samples,n_states)
        """
        return spchu.logf(self.predict_proba(X))
            
    def predict_proba(self,X):
        return_1D = False
        if X.ndim == 1: return_1D = True
        X = X.reshape(-1,self.n_features)
        n_samples = X.shape[0]
        proba = np.ones((n_samples,self.n_classes))
        for j in range(self.n_features):
            featprobs = self.feature_probs[j][:,X[:,j]].T
            proba = proba * featprobs
        return proba
        
    def print_model(self,labels=None):
        for j in range(self.n_features):
            print(" ++ Feature (%d) ++"%j)
            labels = self.labels[j]
            
            featprob_df = pd.DataFrame(self.feature_probs[j].T,columns=self.classes,
                     index= ['P('+labels[i]+'|.)' for i in range(0,self.n_categories[j])])
            display(featprob_df)





class GaussianMixtureClf(BaseEstimator, ClassifierMixin):
    """ Gaussian Mixture Model based Classifier
    
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
    def __init__(self, n_components=1 ,classes=[0,1], max_iter=10, tol=1.e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

        self.data_range_ = None

        self.classes = classes
        self.n_components=n_components
        self.n_classes = len(self.classes)
        self.class_count_ = np.zeros(self.n_classes,dtype='float64')
        self.class_prior_ = np.ones(self.n_classes,dtype='float64')/float(self.n_classes)
        
        self.gmm = [GaussianMixture(max_iter=self.max_iter,tol=self.tol,random_state=1,n_components=self.n_components, 
                        covariance_type='diag',init_params='kmeans') for k in range(0,self.n_classes)]   


        
    def _validate_params(self):
        if self.n_classes < 2:
            raise ValueError("Minimum Number of Classes is 2")
        if self.n_classes != len(self.gmm):
            raise ValueError("Initialized GMM shape(%d,_) does not match n_classes(%d) "
                % (len(self.gmm), self.n_classes) )
    
        
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
#            print("Model for class: ",self.gmm[k].means_,np.sqrt(self.gmm[k].covariances_))        
              
        # keep the counts in the model for adaptation and incremental training purposes
        # keep the datarange for summary / plotting utilities
        self.class_prior_ = self.class_count_ / len(y)
        self.data_range_ = np.vstack((np.min(X,axis=0),np.max(X,axis=0)))
        
    def predict_log_prob(self,X):
        """ Likelihoods of  X  for each class
        
            Returns
            -------
            array, shape (n_samples, n_classes)
                Predicted likelihoods per class
                
            array, shape (n_samples,)
                Total Likelihood per sample
        """
        Xprob = np.ndarray(dtype='float64',shape=(X.shape[0],self.n_classes))
        for k in range(0,self.n_classes):
            Xprob[:,k]= self.gmm[k].score_samples(X)      
        return Xprob

    def predict_prob(self,X):
        """ Log-Probability estimates (likelihoods) per class
        """
        return np.exp(self.predict_log_prob(X))

        

    def predict_proba(self, X, priors = None):
        """ 
        Computes the class probabilities (posteriors) given the samples
        You can override the priors derived during training
        """
        
        if priors == None:
            priors = self.class_prior_
            
        if (len(priors) != self.n_classes):
            raise ValueError("Dimensions of priors do not match number of classes")
            
        likelihoods = self.predict_prob(X)
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

            
    def print(self):
        """ print the model """
        for k in range(0,self.n_classes):
            print("Class[%d] (%s) with prior=%.3f" % (k,self.classes[k],self.class_prior_[k]))
            print("-----------------------------------")
            if  self.gmm[k].means_.shape[1] == 1 :
                df = pd.DataFrame(data={'weights':self.gmm[k].weights_.reshape(-1), 
                                    'mean':self.gmm[k].means_.reshape(-1), 
                                    'std_dev':np.sqrt(self.gmm[k].covariances_).reshape(-1)})
                print(df)
            else:
                for kk in range(self.n_components):
                    print( self.gmm[k].weights_[kk], self.gmm[k].means_[kk],np.sqrt(self.gmm[k].covariances_[kk]) ) 
            print("")
    
    def plot_prob(self):
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
            z = self.predict_prob(np.vstack(xx,yy))
            h = plt.contourf(x,y,z)
            plt.show()
        else:
            x = np.linspace(self.data_range_[0,0],self.data_range_[1,0],num=npts)
            z = self.predict(np.vstack(x))
            h = plt.plot(x,z)
            plt.show()
