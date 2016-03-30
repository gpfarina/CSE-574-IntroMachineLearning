import numpy as np
from scipy.optimize import minimize
from sklearn.lda import LDA #for comparison
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
    means=np.array([m1,m2,m3,m4,m5])
    covmat=np.cov(Z[:,0:2].T) #this is the real covmat
#    covmat=identity(2) #but with this we get better results
    prior[0]=np.float(Z1.shape[0])/np.float(Z.shape[0])
    prior[1]=np.float(Z2.shape[0])/np.float(Z.shape[0])
    prior[2]=np.float(Z3.shape[0])/np.float(Z.shape[0])
    prior[3]=np.float(Z4.shape[0])/np.float(Z.shape[0])
    prior[4]=np.float(Z5.shape[0])/np.float(Z.shape[0])
    return means.T,covmat


def delta(k, means, covmat, x, prior):
  return(np.dot(np.dot((x-means[:, k-1]).T,np.linalg.solve(covmat, np.identity(2))),(x-means[:, k-1])))
   # return(np.dot(means[:, k-1].T,x)-0.5*np.dot(means[:, k-1].T,means[:, k-1])+prior)
    
def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    d=np.zeros(5)
    ypred=np.zeros(ytest.shape[0])
    for j in range(Xtest.shape[0]):
        for i in range(1,6):
            d[i-1]=delta(i, means, covmat, Xtest[j], prior[i-1])
        ypred[j]=np.argmin(d)+1
    
    acc=100.0*np.sum((ytest.flatten() == ypred).astype(float))/ytest.shape[0]
    # IMPLEMENT THIS METHOD
    return acc,ypred


def main2():
    if sys.version_info.major == 2:
        X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
    else:
        X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')
    means,covmat = ldaLearn(X,y)
    ldaacc,ypred = ldaTest(means,covmat,Xtest,ytest)
    print('LDA Accuracy = '+str(ldaacc))
  
    # clf = LDA()
    # clf.fit(X, np.ravel(y))
    # ypred2=clf.predict(Xtest)
    # print('LDA Accuracy = '+str(100*np.mean((ytest.flatten() == ypred2).astype(float))))


prior=np.zeros(5)
main2()
