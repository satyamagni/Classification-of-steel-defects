#import timeit
#setup = '''
import numpy as np
import pandas as pd
#'''
#my_code = '''
X = pd.read_csv("training-steel.csv", header = None)
Y = pd.read_csv("testing-steel.csv", header = None)
X = X.to_numpy()
Y = Y.to_numpy()

x=X[:,0:27]
y=Y[:,0:27]
Xctrain=X[:,27]
Xctest=Y[:,27]

k = 16
errors = np.zeros(275)
score = np.zeros(275)
prediction = np.zeros(275)

for i in range(275):
    dist = np.sum((x - y[i,:])**2,axis = 1)**0.5
    sortIndex = np.argsort(dist)
    bestLabels = Xctrain[sortIndex[0:k]]
    prediction[i] = (sum(bestLabels) > k/2.0)*1.0
    errors[i] = (Xctest[i] != prediction[i])*1.0
    
    #score for ROC curve
    score[i] = sum(bestLabels)    
error = np.sum(errors)

#confusion matrix calculation
CM=np.zeros((2,2))
for n in range(275):
    if prediction[n]==1:
        if Xctest[n]==1:
                CM[0,1]=CM[0,1]+1
        else:
                CM[0,0]=CM[0,0]+1
    else:
        if Xctest[n]==1:
                CM[1,1]=CM[1,1]+1
        else:
                CM[1,0]=CM[1,0]+1
                
#'''
#print (timeit.timeit(setup = setup, stmt = my_code, number = 1))





