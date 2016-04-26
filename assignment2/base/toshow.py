from sklearn.qda import QDA
import numpy as np
import pickle
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
clf = QDA()
y=y.ravel()
clf.fit(X, y)
ypred=clf.predict(Xtest)
ypred=ypred.reshape(ypred.shape[0],1)
acc=100*np.mean((ytest == ypred).astype(float))
print(acc)
