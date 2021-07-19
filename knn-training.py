#import timeit

#setup = '''
import numpy as np
import pandas as pd
import timeit
#'''
#my_code = '''
X = pd.read_csv("training-steel.csv", header = None)
Y = pd.read_csv("testing-steel.csv", header = None)
X = X.to_numpy()
Y = Y.to_numpy()

E = np.zeros(50)
for k in range(50):
    err = np.zeros(4)
    for b in range(4):
        Xtesttemp = np.concatenate((X[125*b:125*(b+1),:],X[500+75*b:500+75*(b+1),:]))
        Xctesttemp = Xtesttemp[:,27]
        
        Xtraintemp = np.delete(X,np.s_[125*b:125*(b+1)],0)
        Xtraintemp = np.delete(Xtraintemp,np.s_[375+75*b:375+75*(b+1)],0)
        Xctraintemp = Xtraintemp[:,27]
        
        x=Xtraintemp[:,0:27]
        y=Xtesttemp[:,0:27]
        
        errors = np.zeros(200)
        for i in range(200):
            dist = np.sum((x - y[i,:])**2,axis = 1)**0.5
            sortIndex = np.argsort(dist)
            bestLabels = Xctraintemp[sortIndex[0:k]]
            prediction = (sum(bestLabels) > k/2.0)*1.0
            errors[i] = (Xctesttemp[i] != prediction)*1.0
        err[b] = np.sum(errors)
    E[k]=np.sum(err)
#    '''
    
#print (timeit.timeit(setup = setup, stmt = my_code, number = 1))