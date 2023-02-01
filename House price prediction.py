# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 09:37:30 2022

@author: fedib
"""

import pandas as pd
from sklearn import svm
import numpy as np

Dataset = pd.read_csv('C:/STUDY/SEMESTRE 2/machine learning/tp/tp2/house-prices.csv')

S= np.array(Dataset['SqFt'])
Reg= np.array(Dataset['Price'])

Size=S.reshape(-1,1)
Real_regression=Reg.reshape(-1,1)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

##Size_train, Size_test, Real_regression_train, Real_regression_test = train_test_split(Size, Real_regression, test_size = 0.3, random_state = 0)

lr = LinearRegression()
##lr.fit(Size_train, Real_regression_train)
import time
debut = time.time()
lr.fit(Size, Real_regression)
fin=time.time()-debut
fin
pred = lr.predict(Size)

"""Q8"""
predx = lr.predict([[1000]])
predx 
""""""
a = lr.coef_

b = lr.intercept_

label1=a*Size + b

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Real_regression,pred)
import math 
print("rmse",math.sqrt(mse)) 

import pandas as pd
Real_regression1=pd.DataFrame(Real_regression)
Real_regression1.describe()

from sklearn.metrics import explained_variance_score
EV=explained_variance_score(Real_regression,pred)
print("explained variance : %f" %(EV))


import matplotlib.pyplot as plt
plt.plot(Size, pred, 'r')
plt.scatter(Size, Real_regression)

"""Q6"""
### print('r squares value', model.score(x,y))
###ou bien
from sklearn.metrics import r2_score
r=r2_score(Real_regression,pred)
print(r)


import matplotlib.pyplot as plt
##plt.plot(Size, Real_regression, 'ro')
plt.scatter(Size, Real_regression)

"""""""""""""""""partie II """""""""""""""""

import pandas as pd
from sklearn import svm
import numpy as np

Dataset = pd.read_csv('C:/STUDY/SEMESTRE 2/machine learning/tp/tp2/house-prices.csv')

price= Dataset["Price"]
Y=np.array(price).reshape(-1,1)

data=Dataset.drop(["Price","Brick","Neighborhood","Home"],axis=1)

from sklearn.preprocessing import LabelEncoder
X1=Dataset["Brick"]
le1=LabelEncoder()
X1New=le1.fit_transform(X1)
X1New=X1New.reshape(-1,1)

X2=Dataset["Brick"]
le2=LabelEncoder()
X2New=le2.fit_transform(X2)
X2New=X2New.reshape(-1,1)

dataNew=np.concatenate((data,X1New,X2New),axis=1)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


lr = LinearRegression()
##lr.fit(Size_train, Real_regression_train)
import time
debut = time.time()
lr.fit(dataNew, Y)
fin=time.time()-debut
print(fin)
pred = lr.predict(dataNew)

a = lr.coef_
b = lr.intercept_

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,pred)
import math 
print("rmse",math.sqrt(mse)) 

import pandas as pd
Y1=pd.DataFrame(Y)
Y1.describe()

from sklearn.metrics import explained_variance_score
EV=explained_variance_score(Y,pred)
print("explained variance : %f" %(EV))


avPlots(pred)
##import matplotlib.pyplot as plt
##plt.plot(dataNew, pred, 'r')
##plt.scatter(dataNew, Y)

