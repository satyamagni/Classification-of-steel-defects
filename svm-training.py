#import timeit

#setup = '''
import numpy as np
from numpy.core.fromnumeric import argmin
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
#'''
#my_code = '''
X = pd.read_csv("training-steel.csv", header = None)
Y = pd.read_csv("testing-steel.csv", header = None)


E = np.zeros(100)

#cross validation to determine gamma 'g'
for g in range(100):
    err = np.zeros(4)
    for b in range(4):
        
        #centre and scaling (batch by batch, 4 batches for cross validation)
        Xnp = X.to_numpy()
        Xtesttemp = np.concatenate((Xnp[125*b:125*(b+1),:],Xnp[500+75*b:500+75*(b+1),:]))
        Xctesttemp = Xtesttemp[:,27]
        
        Xtraintemp = np.delete(Xnp,np.s_[125*b:125*(b+1)],0)
        Xtraintemp = np.delete(Xtraintemp,np.s_[375+75*b:375+75*(b+1)],0)
        Xctraintemp = Xtraintemp[:,27]
        
        Xtraintemp = pd.DataFrame(Xtraintemp)
        Xtesttemp = pd.DataFrame(Xtesttemp)
        
        mu = np.mean(Xtraintemp)
        Xtraintemp = Xtraintemp - mu
        std = np.std(Xtraintemp)
        Xtraintemp = Xtraintemp/std
        Xtesttemp = Xtesttemp -mu
        Xtesttemp = Xtesttemp/std
        
        Xtraintemp = Xtraintemp.to_numpy()
        Xtesttemp = Xtesttemp.to_numpy()
        
        x=Xtraintemp[:,0:27]
        y=Xtesttemp[:,0:27]
        
        #model fitting
        svm = SVC(kernel = 'rbf', gamma = (g+1)*0.01)
        svm.fit(x,Xctraintemp)
        Y_pred = svm.predict(y)
        err[b] = sum(abs(Y_pred-Xctesttemp))
    E[g] = np.sum(err)
#'''
#print (timeit.timeit(setup = setup, stmt = my_code, number = 1))
print(argmin(E))