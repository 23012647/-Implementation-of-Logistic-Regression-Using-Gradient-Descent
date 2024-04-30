# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the data file and import numpy, matplotlib and scipy.

2.Visulaize the data and define the sigmoid function, cost function and gradient descent.

3.Plot the decision boundary.

4.Calculate the y-prediction.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SIBIRAJ E
RegisterNumber: 212223080052 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset['gender'] = dataset['gender'].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def loss(theta ,X ,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iters):
    m=len(y)
    for i in range(num_iters):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -=alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iters=1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)



```

## Output:
dataset

![image](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568857/f0867f7f-7760-49c5-91a1-11651def2aee)

datatypes

![image](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568857/6abb2fa0-dae6-44b3-8165-64a8a30d1ac6)


dataset after printing only codes columns

![image](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568857/195b4be8-42e2-4b9a-b4cf-8c6ca544c315)

Accuracy

![image](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568857/8339e6d2-f639-4847-b399-7c6932ec427a)

Array values of Y prediction

![image](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568857/e35c024d-e8ec-4787-989c-ae522da7e3f6)

Array values of Y

![image](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568857/4ebd1fb5-2959-434f-bca7-31539316b414)

predicting with different values

![image](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568857/80d70188-2287-41d3-85a9-391bee4ea6e1)








## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

