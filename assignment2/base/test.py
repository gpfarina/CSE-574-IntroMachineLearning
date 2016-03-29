import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    Z=np.concatenate([X,y], axis=1)
    Z1=Z[Z[:, 2] == 1, :]
    m1=np.mean(Z1, axis=0)
    m1=m1[:2]
    Z2=Z[Z[:, 2] == 2, :]
    m2=np.mean(Z2, axis=0)
    m2=m2[:2]
    Z3=Z[Z[:, 2] == 3, :]
    m3=np.mean(Z3, axis=0)
    m3=m3[:2]
    Z4=Z[Z[:, 2] == 4, :]
    m4=np.mean(Z4, axis=0)
    m4=m4[:2]
    Z5=Z[Z[:, 2] == 5, :]
    m5=np.mean(Z5, axis=0)
    m5=m5[:2]
    means=np.array(m1,m2,m3,m4,m5)
    covmat=np.cov(Z[:2])
    prior[0]=np.float(Z1.shape[0])/np.float(Z.shape[0])
    prior[1]=np.float(Z2.shape[0])/np.float(Z.shape[0])
    prior[2]=np.float(Z3.shape[0])/np.float(Z.shape[0])
    prior[3]=np.float(Z4.shape[0])/np.float(Z.shape[0])
    prior[4]=np.float(Z5.shape[0])/np.float(Z.shape[0])
    return means,covmat




def delta(k, means, covmat, x, prior):
    np.dot(np.dot(x.T, np.linalg.inv(covmat)), means[k-1])-0.5*np.dot(np.dot(means[k-1].T,np.linalg.inv(covmat)), means[k-1])+np.log(prior[k-1])
    
def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    for j in range(Xtest.shape[0]):
        for i in range(1,6):
            d[i-1]=delta(i, means, covmat, Xtest[j], prior[i-1])
        ypred[j]=np.argmax(d)+1
    # IMPLEMENT THIS METHOD
    return acc,ypred


def main():
    if sys.version_info.major == 2:
        X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
    else:
        X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')
    prior=np.zeros(5)
    means,covmat = ldaLearn(X,y)
    ldaacc = ldaTest(means,covmat,Xtest,ytest)
    print('LDA Accuracy = '+str(ldaacc))
