# A number of 'sequence data sets'

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

###### some helper functions
# perturb with normal distributed noise
# optionally (default) mean and variance normalization which is done prior to noise addition
def perturb(x,norm=False,lev=1.):
    if norm: x = norm_mv(x)
    return(x + np.random.normal(scale=lev,size=x.shape))
# mean and variance normalization (per column)
def norm_mv(x):
    x0 = x - x.mean(axis=0,keepdims=True)
    return(x0/x0.std(axis=0,keepdims=True))
# maps [-range,range]  to nbins class labels with rather uniform distribution if std.norm input
def discretize(X,nbins=4,xrange=2.): 
    _,n_features = X.shape
    Xd = np.zeros(dtype='int32',shape=X.shape)
    if not isinstance(nbins,list):
        nbins = [nbins]*n_features
    if not isinstance(xrange,list):
        xrange = [xrange]*n_features
    for i in np.arange(n_features):
        x = X[:,i]
        x = (x+xrange[i])*(nbins[i]/(2.*xrange[i]))
        xd = np.clip(x.astype('int32'),0,nbins[i]-1)
        Xd[:,i] = xd
    return(Xd)
##########

def make_seq1(noise=0.1,return_X_y=True):
    """
    make_seq1 is a small sequence data set
    it has 4D feature vectors and data is drawn sequentially from 3 classes
    
    there are 2 sets with 75 data samples per set
    the data is returned as a tuple of tuples (X1,y1),(X2,y2)
    
    the noisiness of the data can be manipulated with the parameter
    noise  default=0.1    best in range(0.,1.)
    """ 
    
    # we actually use the Iris data set and canibalize it into a sequence data set
    X, y = load_iris(return_X_y=True)
    X = perturb(X,lev=noise) 
    X1 = X[0::2]
    y1 = y[0::2]
    X2 = X[1::2]
    y2 = y[1::2]
    
    return( (X1,y1), (X2,y2) )

def make_seq1d(nbins=[4,4,4,4],noise=0.1,return_X_y=True):
    """
    make_seq1d is a small sequence data set
    it has 4D feature vectors and data is drawn sequentially from 3 classes
    there are 2 sets with 75 data samples per set
    it is a discretized version of the continuous data generated in make_seq1
    
    the number of labels per class can be specified in the nbins as a list
    
    the data is returned as a tuple of tuples (X1,y1),(X2,y2)
    
    the noisiness of the data can be manipulated with the parameter
    noise  default=0.1    best in range(0.,1.)
    """  
    
    (X1,y1), (X2,y2) = make_seq1(noise=noise)

    X1d = discretize(norm_mv(X1),nbins=nbins)
    X2d = discretize(norm_mv(X2),nbins=nbins)
    
    return( (X1d,y1), (X2d,y2) )

def load_seq_iris_1():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    # sort train and test to make them look sequential
    indx = np.argsort(y_train)
    y_train = y_train[indx]
    X_train = X_train[indx]
    indx = np.argsort(y_test)
    y_test = y_test[indx]
    X_test = X_test[indx]

    # make also a noisy test set
    X_noisy = X_test + np.random.normal(loc=0.,scale=1,size=X_test.shape)

    df_train = pd.DataFrame(X_train)
    df_train['lbl'] = y_train
    df_test = pd.DataFrame(X_test)
    df_test['lbl'] = y_test
    df_noisy = pd.DataFrame(X_noisy)
    df_noisy['lbl'] = y_test
    return(df_train,df_test,df_noisy)
    
