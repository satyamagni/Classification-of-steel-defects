#import timeit

#setup = '''
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


D_tr = pd.read_csv("training-steel.csv", header = None)
D_te = pd.read_csv("testing-steel.csv", header = None)
D_tr = D_tr.to_numpy()
D_te = D_te.to_numpy()
x_tr = D_tr[:,0:27]
y_tr = D_tr[:,27]
x_te = D_te[:,0:27]
y_te = D_te[:,27]

stdev = np.std(D_tr, axis=0)
print(stdev)

#scaler = StandardScaler()
#scaler.fit(x_tr)
#x_tr = scaler.transform(x_tr)
#x_te = scaler.transform(x_te)
#print(np.std(x_tr, axis=0))

#training
adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state = 0)
adaboost.fit(x_tr, y_tr)

#'''
#my_code = '''

#testing
y_pred = adaboost.predict(x_te)
err = sum(abs(y_pred-y_te))

fpr, tpr, thresholds = metrics.roc_curve(y_te, adaboost.decision_function(x_te))
fpr1, tpr1, thresholds1 = metrics.roc_curve(y_tr, adaboost.decision_function(x_tr))

#confusion matrix
CM=np.zeros((2,2))
for n in range(275):
    if y_pred[n]==1:
        if y_te[n]==1:
                CM[0,0]=CM[0,0]+1
        else:
                CM[0,1]=CM[0,1]+1
    else:
        if y_te[n]==1:
                CM[1,0]=CM[1,0]+1
        else:
                CM[1,1]=CM[1,1]+1
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