import numpy as np
import pickle as pl
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt, exp


def initializeWeights(n_in,n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
def sigmoid(z):
    return(1/(1+np.exp(-1.0*z)))

def step(w, p):
    p=p.reshape((p.shape[0],1))
    return(sigmoid(((np.dot(w, np.vstack((p,[1.0])))))))

def ff(w1, w2, p):
    n_classes=10
    z=step(w1,p)
    o=step(w2, z)
    return(o)

def featureSelection(M):
    return(np.array(filter(lambda(x): min(x)!=max(x), zip(*M))).T)

def preprocess():
    mat = loadmat('mnist_all.mat')

    for i in range(10):
        m = mat.get('train'+str(i))
        m=(m/255.0)
        num_row=m.shape[0]
        label=np.ones((num_row,(1.0)))
        c=np.hstack((m,label*i))
        mat['train'+str(i)]=c

    for i in range(10):
        m = mat.get('test'+str(i))
        m = (m/255.0)
        num_row=m.shape[0]
        label=np.ones((num_row,(1.0)))
        c=np.hstack((m,label*i))
        mat['test'+str(i)]=c

    train_stack=mat.get('train0')

    for i in range(1,10):
        temp123=mat.get('train'+str(i))
        train_stack=np.concatenate((train_stack,temp123),axis=0)


    for i in range(10):
        temp=mat.get('test'+str(i))
        train_stack=np.concatenate((train_stack,temp),axis=0)
        
    train_stack=featureSelection(train_stack) #featureSelection

    split = range(train_stack.shape[0])
    aperm = np.random.permutation(split)

    train_stack_tdata = train_stack[aperm[0:50000],:]
    train_stack_vdata = train_stack[aperm[50000:60000],:]
    test_stack = train_stack[aperm[60000:],:]
    


    newFet=train_stack.shape[1]

    train_data = np.array(train_stack_tdata)[:,0:newFet-1]
    train_label = np.array(train_stack_tdata)[:,newFet-1:]
    validation_data = np.array(train_stack_vdata)[:,0:newFet-1]
    validation_label =  np.array(train_stack_vdata)[:,newFet-1:]
    test_data = np.array(test_stack)[:,0:newFet-1]
    test_label = np.array(test_stack)[:,newFet-1:]
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
def vectorize(v,nclasses):
    v=int(v)
    if(v<nclasses and v>=0):
        r=np.zeros(nclasses, int)
        r[v]=int(1)
        return(r)
    else:
        print("error in vectorize")
        


def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    grad_w1 =  np.zeros(n_hidden*(n_input + 1)).reshape( (n_hidden, (n_input + 1)) ) 
    grad_w2 =  np.zeros(n_class  * (n_hidden +1 )).reshape((n_class, (n_hidden + 1)))
    delta=np.zeros(10)

    for p in range(training_data.shape[0]):
        z=step(w1, training_data[p]).reshape((n_hidden,1))
        z1=np.vstack((z,[1.0]))
        o=step(w2, z)
        y=vectorize(training_label[p],10)
        y=y.reshape((y.shape[0],1))
        obj_val+=np.sum(np.square((y-o)))/2
        x=training_data[p].reshape((training_data[p].shape[0],1))
        x=np.vstack((x,1.0))
        delta=(y-o)*o*(1-o)
        grad_w2+=- np.dot(delta,z1.T)
        tmp=w2[:,0:n_hidden]
        A= np.dot(delta.T, tmp).T
        B= ((1-z)*z)
        C= A*B
        D= np.dot(C, x.T)
        grad_w1+= -D

    
    grad_w1/=training_data.shape[0]    
    grad_w2/=training_data.shape[0]
    grad_w1+=(lambdaval/training_data.shape[0])*(w1)
    grad_w2+=(lambdaval/training_data.shape[0])*(w2)
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_val/=training_data.shape[0]
    obj_val+=lambdaval*(((w1*w1).sum())+((w2*w2).sum()))/(2*training_data.shape[0] )
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    labels = np.zeros(data.shape[0]).reshape((data.shape[0], 1))
    for i in range(data.shape[0]):
        o=ff(w1,w2, data[i])
        labels[i]=np.argmax(o)

    return labels


def experiment():
    train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess()
    n_input = train_data.shape[1]
    n_class=10
    opts = {'maxiter' : 50}
    R={'00':(0.0, (np.array([[0,0],[0,0]]), np.array([[0,0],[0,0]])), 0.0, 0.0, 0.0)}
    for nH in range(4, 24, 4):
        initial_w1 = initializeWeights(n_input, nH)
        initial_w2 = initializeWeights(nH, n_class)
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
        i=0
        for lambdaval in np.linspace(0,1,10):
            print("nH: "+str(nH)+" lambdaval:  "+str(lambdaval)+" i: "+str(i))
            args = (n_input, nH, n_class, train_data, train_label, lambdaval)
            nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
            w1 = nn_params.x[0:nH* (n_input + 1)].reshape( (nH, (n_input + 1)))
            w2 = nn_params.x[(nH * (n_input + 1)):].reshape((n_class, (nH + 1)))
            predicted_label = nnPredict(w1,w2,train_data)
            accTrainData=100*np.mean((predicted_label == train_label).astype(float))
            predicted_label = nnPredict(w1,w2,validation_data)
            accValData=100*np.mean((predicted_label == validation_label).astype(float))
            predicted_label = nnPredict(w1,w2,test_data)
            accTestData=100*np.mean((predicted_label == test_label).astype(float))
            R[str(nH)+str(i)]=(lambdaval, (w1,w2), accTrainData, accValData, accTestData)
            print(R[str(nH)+str(i)])
            pl.dump(R, open("res.p", "wb")) #temporary save
            i=i+1
    return R

R=experiment()
pl.dump(R, open("res.p", "wb"))



