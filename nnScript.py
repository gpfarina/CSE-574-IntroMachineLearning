import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt, exp


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return(1/(1+np.exp(-1.0*z)))

def step(w, p):
    p=p.reshape((p.shape[0],1))
    return(sigmoid(((np.dot(w, np.vstack((p,[1.0])))))))

def ff(w1, w2, p):
    n_classes=10
    z=step(w1,p)
    o=step(w2, z)
    return(o)

#we remove from the matrix all and only the columns that contain the same digit
def featureSelection(M):
    return(np.array(filter(lambda x: min(x)!=max(x), zip(*M))).T)

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data

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
    
    #Your code here

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

    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
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
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_val/=training_data.shape[0]
    
    obj_val+=lambdaval*(((w1*w1).sum())+((w2*w2).sum()))/(2*training_data.shape[0] )

    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 

    
    
    
    labels = np.zeros(data.shape[0]).reshape((data.shape[0], 1))
    for i in range(data.shape[0]):
        o=ff(w1,w2, data[i])
        labels[i]=np.argmax(o)



    #Your code here
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""


print("start")
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();
print("preprocess done")

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 4;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.5;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.
print ("start minimize")
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
print("end minimize")
#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
