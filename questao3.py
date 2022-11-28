#Yan D. R. Machado - MAT1305 - Resolução Lista 2
#questão3

import numpy as np
import matplotlib.pyplot as plt
# import statistics
# import math
# import pandas as pd
# from autograd import grad

def pontos_2d(w,P): #função para gerar os pontos aleatórios
    X = [np.ones(P), 2*np.random.random_sample((P,))-1, 2*np.random.random_sample((P,))-1]
    # X = [np.ones(P),np.random.random_sample(P,),np.random.random_sample((P,))]
    X = np.array(X)
    Y = []
    for i in range(P): 
        if X[2][i]**2 * w[2] + X[1][i]**2 * w[1] + X[0][i]*w[0] < 0:
            Y.append(-1)
        else:
            Y.append(1)
    return X, Y

# def pontos_2d(w,P):
#     X = [np.ones(P),np.random.random_sample(P,),np.random.random_sample((P,))]
#     X = np.array(X)
#     Y = []
#     for i in range(P):
#         if X[2][i]*w[2] + X[1][i]*w[1] + X[0][i]* w[0] < 0:
#             Y.append(1)
#             X[2][i] += 0.1
#         else:
#             Y.append(-1)
#             X[2][i] -= 0.05
#     return X,Y

#feature transformation (livro 10.2 - examples)

# def feature_transforms(x,w):
#     # calculate feature transform
#     f = np.sin(w[0] + np.dot(x.T,w[1:])).T
#     return f

# def model(x,w):    
#     # feature transformation 
#     f = feature_transforms(x,w[0])
    
#     # compute linear combination and return
#     a = w[1][0] + np.dot(f.T,w[1][1:])
#     return a.T

def f(x):
    return x**2

def modelx(w,x):
    return np.dot(f(x),w)

def model(w,x):
    return x[2]**2 * w[2] + x[1]**2 * w[1] + x[0]*w[0]

def g_lr_softmax(w,X,Y):
    # N = len(X[:,0])
    P = len(X[0,:])
    cost = 0
    for p in range(P):
        c = model(w,X[:,p])
        # print([c,Y[p]])
        cost = cost + np.log((1+np.exp(-Y[p]*c)))
        # print(cost)
    cost = cost/P
    return cost

def grad_lr_softmax(w,X,Y):
    N = len(X[:,0])
    P = len(X[0,:])
    grad = np.zeros(N)
    for p in range(P):
        c = model(w,X[:,p])
        k = Y[p] * np.exp(-Y[p]*c) / (1 + np.exp(-Y[p]*c))
        grad = grad + k * f(X[:,p])
    grad = -grad/P
    return grad

def gradient_descent(alpha, max_its, w, X,Y):
    weight_history = [w]
    cost_history = [g_lr_softmax(w,X,Y)]
    tol = alpha
    for k in range(max_its):
        grad_eval = grad_lr_softmax(w,X,Y)
        # print('grad = ', grad_eval)
        
        w = w - alpha*grad_eval
        # print('w = ',w)
        # print('custo = ', g_lr_softmax(w,X,Y))
        
        weight_history.append(w)
        cost_history.append(g_lr_softmax(w,X,Y))
        norm = np.linalg.norm(grad_eval)
        if norm < tol:
            print('alpha ', alpha)
            alpha = alpha/10
            tol = tol/10
    return weight_history, cost_history

P = 500
w = np.array([1,-1/0.64,-1/0.25])
[X,Y] = pontos_2d(w,P)
N = 2
max_its = 500
alpha = 0.01
[w,cost] = gradient_descent(alpha,max_its,w,X,Y)

# print(w[max_its-1])
# print(cost)

# for k in range(1,max_its,50):
#     plt.figure(figsize = (8,8))
#     for i in range(P):
#         if Y[i] > 0:
#             plt.scatter(X[1,i],X[2,i],c = 'b')
#         else:
#             plt.scatter(X[1,i],X[2,i],c = 'r')
#     theta = np.arange(0,6.5,0.1)
#     n = len(theta)
#     x = np.sqrt(-w[k][0]/w[k][1]) * np.cos(theta)
#     y = np.sqrt(-w[k][0]/w[k][2]) * np.sin(theta)
#     # print(x,y)
#     plt.plot(x,y,'g')
#     plt.show()
# plt.show()

#mudando as características do plot para o espaço de features conforme figura da lista

for k in range(1,max_its,500):
    plt.figure(figsize = (8,8))
    for i in range(P):
        if Y[i] > 0:
            plt.scatter(X[1,i]**2,X[2,i]**2,c = 'r')
        else:
            plt.scatter(X[1,i]**2,X[2,i]**2,c = 'b')
    x = np.linspace(0,3/5,P)
    y = -(w[k-1][1] * x + w[k-1][0]) / w[k-1][2]
    plt.plot(x, y, c = 'g')
    plt.show()
plt.show()