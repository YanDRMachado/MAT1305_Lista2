#Yan D. R. Machado - MAT1305 - Resolução Lista 2
#questão2

import numpy as np
import matplotlib.pyplot as plt
# import statistics
# import math
# import pandas as pd
# from autograd import grad

def pontos_2d(P): #função para gerar os pontos aleatórios
    np.random.seed(1)
    X = [5*np.random.random_sample(P,),5*np.random.random_sample(P,)]
    X = np.array(X)
    Y = np.sin(X[0,:]+X[1,:]) + 0.2 * np.random.random_sample(P,)
    return X,Y

#resolver o problema NL_regression que foi feito em sala adicionando uma dimensão aos elementos computados

def f(v,x):
    return np.sin(v[0] + x[0] * v[1] + x[1] * v[2])

def model(w,v,x):
    a = w[0] + np.sin(v[0] + x[0] * v[1] + x[1] * v[2]) * w[1]
    return a

def NL_regression(w,v,x,y):
    P = len(x)
    cost = 0
    for p in range(P):
        cost = cost + (model(w,v,x[:,p]) - y[p])**2
    cost /= P
    return cost

    
def grad_NL_regression(w,v,x,y):
    P = len(x)
    grad = np.zeros(5)
    for p in range(P):
        k = 2*(model(w,v,x[:,p]) - y[p])
        grad[0] += k
        grad[1] += k * np.sin(v[0] + x[0][p] * v[1] + x[1][p] * v[2])
        grad[2] += k * np.cos(v[0] + x[0][p] * v[1] + x[1][p] * v[2]) * w[1]
        grad[3] += k * np.cos(v[0] + x[0][p] * v[1] + x[1][p] * v[2]) * w[1] * x[0][p]
        grad[4] += k * np.cos(v[0] + x[0][p] * v[1] + x[1][p] * v[2]) * w[1] * x[1][p]
    grad /= P
    return grad

def gradient_descent(x,y,alpha,max_its):
    w_THETA = [1,1,1,1,1] # [w0,w1,v0,v1,v2]
    w_THETA = np.array(w_THETA)
    weight_history = [w_THETA]
    w = np.zeros(2)
    v = np.zeros(3)
    w[0] = w_THETA[0]
    w[1] = w_THETA[1]
    v[0] = w_THETA[2]
    v[1] = w_THETA[3]
    v[2] = w_THETA[4]
    
    cost_history = [NL_regression(w,v,x,y)]
    tol = 0.1
    for k in range(max_its):
        w_THETA = w_THETA - alpha * grad_NL_regression(w,v,x,y)
        weight_history.append(w_THETA)
        cost_history.append(NL_regression(w,v,x,y))
        g = grad_NL_regression(w,v,x,y)
        norm = np.linalg.norm(g)
        if norm < tol:
            alpha = alpha / 10
            tol = tol / 10
        w[0] = w_THETA[0]
        w[1] = w_THETA[1]
        v[0] = w_THETA[2]
        v[1] = w_THETA[3]
        v[2] = w_THETA[4]
    return weight_history, cost_history

P = 500
[X,Y] = pontos_2d(P)
max_its = 500
alpha = 0.01
[w_h,cost] = gradient_descent(X,Y,alpha,max_its)
ax = plt.axes(projection = '3d')
ax.view_init(15, 140)

#plot
for k in range(1,max_its,50):
    for i in range(P):
        ax.scatter(X[0,:], X[1,:],Y, c = Y, cmap='plasma', linewidth=1)
    x = np.linspace(0,5,P)
    y = np.linspace(0,5,P)
    xplot, yplot = np.meshgrid(x,y)
    w = np.zeros(2)
    v = np.zeros(3)
    w[0] = w_h[k][0]
    w[1] = w_h[k][1]
    v[0] = w_h[k][2]
    v[1] = w_h[k][3]
    v[2] = w_h[k][4]
    z = w[0] + np.sin(v[0] + xplot*v[1] + yplot*v[2]) * w[1]
    ax.plot_surface(xplot, yplot, z, alpha = 0.5, color='g')
    plt.show()