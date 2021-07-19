#import timeit

#setup = '''
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt

#'''
#my_code = '''
X = pd.read_csv("training-steel.csv", header = None)
Y = pd.read_csv("testing-steel.csv", header = None)

mu = np.mean(X)
X = X - mu
std = np.std(X)
X = X/std
Y = Y - mu
Y = Y/std

X = X.to_numpy()
Y = Y.to_numpy()

X[:,27] = np.concatenate((np.zeros(500),np.ones(300)))
Y[:,27]= np.concatenate((np.zeros(173),np.ones(102)))

x=X[:,0:27]
xc=X[:,27]
y=Y[:,0:27]
yc=Y[:,27]
        
#model fitting based on gamma from cross validation
svm = SVC(kernel = 'rbf', gamma = 0.03)
svm.fit(x,xc)
Y_pred = svm.predict(y)
err = sum(abs(Y_pred-yc))

#ROC curve, fpr stands for 'false positive rate', tpr for 'true positive rate'
fpr, tpr, thresholds = metrics.roc_curve(yc, svm.decision_function(y))
fpr1, tpr1, thresholds1 = metrics.roc_curve(xc, svm.decision_function(x)) 

#confusion matrix
CM=np.zeros((2,2))
for n in range(275):
    if Y_pred[n]==1:
        if yc[n]==1:
                CM[0,1]=CM[0,1]+1
        else:
                CM[0,0]=CM[0,0]+1
    else:
        if yc[n]==1:
                CM[1,1]=CM[1,1]+1
        else:
                CM[1,0]=CM[1,0]+1
#'''                
#print (timeit.timeit(setup = setup, stmt = my_code, number = 1))
                
plt.plot(fpr, tpr)
plt.plot(fpr1, tpr1)
plt.legend(['test', 'train'])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.grid()
plt.show()
print(CM)

