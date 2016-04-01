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
    means=np.array([m1,m2,m3,m4,m5])
    covmat=np.cov(Z[:,0:2].T) #this is the real covmat
    return means.T,covmat

def delta(mean, covmat, x):
    return(np.dot(np.dot((x-mean).T,np.linalg.solve(covmat, np.identity(2))),(x-mean)))


def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    Z=np.concatenate([X,y], axis=1)
    Z1=Z[Z[:, 2] == 1, :]
    m1=np.mean(Z1, axis=0)
    c1=np.cov(Z1[:,0:2].T)
    m1=m1[:2]
    Z2=Z[Z[:, 2] == 2, :]
    m2=np.mean(Z2, axis=0)
    c2=np.cov(Z2[:,0:2].T)
    m2=m2[:2]
    Z3=Z[Z[:, 2] == 3, :]
    m3=np.mean(Z3, axis=0)
    c3=np.cov(Z3[:,0:2].T)
    m3=m3[:2]
    Z4=Z[Z[:, 2] == 4, :]
    m4=np.mean(Z4, axis=0)
    c4=np.cov(Z4[:,0:2].T)
    m4=m4[:2]
    Z5=Z[Z[:, 2] == 5, :]
    m5=np.mean(Z5, axis=0)
    c5=np.cov(Z5[:,0:2].T)
    m5=m5[:2]
    means=np.array([m1,m2,m3,m4,m5])
    covmats=np.array([c1,c2,c3,c4,c5])
    return means.T,covmats
    
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
            d[i-1]=delta(means[:, i-1], covmat, Xtest[j])
        ypred[j]=np.argmin(d)+1 #we take the minimum melhanobis distance which is exactly the same as the maximum Log likelihood
        
        
    acc=100*np.mean((ytest.flatten() == ypred).astype(float))
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    d=np.zeros(5)
    ypred=np.zeros(ytest.shape[0])
    for j in range(Xtest.shape[0]):
        for i in range(1,6):
            d[i-1]=delta(means[:, i-1], covmats[i-1], Xtest[j])
        ypred[j]=np.argmin(d)+1 
    # IMPLEMENT THIS METHOD
    acc=100*np.mean((ytest.flatten() == ypred).astype(float))
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD
    return np.dot(np.dot(np.linalg.solve(np.dot(X.T, X), np.identity(X.shape[1])),X.T),y)

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD               
    w=np.dot(np.linalg.solve(lambd*np.identity(65)+np.dot(X.T, X),np.identity(65)),np.dot(X.T,y))
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    rmse=np.sqrt(np.dot((ytest-np.dot(Xtest,w)).T,(ytest-np.dot(Xtest,w)))/Xtest.shape[0])
    # IMPLEMENT THIS METHOD
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                     
    w=w.reshape(65,1)
    B=np.dot(np.dot(X.T, X),w).reshape(w.shape[0],1) 
    C=(np.dot(X.T, y)-lambd*w).reshape(w.shape[0],1)
    error_grad=B - C
    A=(y-np.dot(X,w)).reshape(y.shape[0],1)
    error=0.5*(np.dot(A.T, A )+lambd*np.dot(w.T, w))

    return (error, error_grad.flatten())

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    return Xd

# Main script

# Problem 1
# LDA
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
#we dont do qda fornow
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.show()
 #let's terminate here for now
zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.show()
# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))

for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    print(rmses3[i])
    i = i + 1
plt.plot(lambdas,rmses3)
plt.show()

print("-----------")
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    print(rmses4[i])
    i = i + 1
plt.plot(lambdas,rmses4)

plt.show()
# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
