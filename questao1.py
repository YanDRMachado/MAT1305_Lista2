#Yan D. R. Machado - MAT1305 - Resolução Lista 2
#questão1

import numpy as np
import matplotlib.pyplot as plt
import statistics #usei para o cálculo da média e desvio padrão
# import math
# import pandas as pd
# from autograd import grad

def pontos_2d(P): #função para gerar os pontos aleatórios
    X = 5*np.random.random_sample(P,)
    X = np.array(X)
    Y = np.sin(X[:]) + 0.2 * np.random.random_sample(P,)
    meanX = statistics.mean(X)
    stdevX = statistics.stdev(X)
    return X, Y, meanX, stdevX #serão usados para a normalização conforme pedido

P = 100
[X,Y,mean,stdev] = pontos_2d(P)
print('mean, stdev: ', mean, ',', stdev)

#---------------------------------não normalizado-----------------------------

def f(v,x):
    return np.sin(v[0] + x * v[1])

def model(w,v,x_p):
    a = w[0] + f(v,x_p) * w[1]
    return a
    
def NL_regression(w,v,x,y):
    P = len(x)
    cost = 0
    for p in range(P):
        cost = cost + (model(w,v,x[p]) - y[p])**2
    cost /= P
    return cost

def grad_NL_regression(w,v,x,y):
    P = len(x)
    grad = np.zeros(4)
    for p in range(P):
        k = 2*(model(w,v,x[p]) - y[p])
        grad[0] += k
        grad[1] += k * f(v,x[p])
        grad[2] += k * np.cos(v[0] + x[p] * v[1])*w[1]
        grad[3] += k * np.cos(v[0] + x[p] * v[1])*w[1]*x[p]
    grad /= P
    return grad

def gradient_descent(x,y,alpha,max_its):
    w_theta = [1,1,1,1] #valores iniciais
    w_theta = np.array(w_theta)
    weight_history = [w_theta]
    w = np.zeros(2)
    v = np.zeros(2)
    w[0] = w_theta[0]
    w[1] = w_theta[1]
    v[0] = w_theta[2]
    v[1] = w_theta[3]
    cost_history = [NL_regression(w,v,x,y)]
    tol = 0.1
    for k in range(max_its):
        w_theta = w_theta - alpha * grad_NL_regression(w,v,x,y)
        weight_history.append(w_theta)
        cost_history.append(NL_regression(w,v,x,y))
        g = grad_NL_regression(w,v,x,y)
        norm = np.linalg.norm(g)
        if norm < tol:
            alpha = alpha/10
            tol = tol/10
        w[0] = w_theta[0]
        w[1] = w_theta[1]
        v[0] = w_theta[2]
        v[1] = w_theta[3]
    return weight_history, cost_history


max_its = 500
alpha = 0.01
[w_h, cost] = gradient_descent(X,Y,alpha,max_its)

# meanX = statistics.mean(X)
# stdevX = statistics.stdev(X)

# print(len(it),len(cost_h))
# print(it)

# it = list(range(0,max_its+1))
# plt.scatter(it, cost, marker='.', color='b')
# plt.scatter(it, cost_h, marker='1', color='r')
# plt.show()

# plot da curva se adaptando ao conjunto de pontos
# for k in range(1,max_its,50):
#     for i in range(P):
#         plt.scatter(X[i],Y[i],c='b')
#     x = np.linspace(0,5,100)
#     w = np.zeros(2)
#     v = np.zeros(2)
#     w[0] = w_h[k][0]
#     w[1] = w_h[k][1]
#     v[0] = w_h[k][2]
#     v[1] = w_h[k][3]
#     y = model(w,v,x)
#     # print(w,v)
#     plt.plot(x,y,c='r')
#     plt.show()
# # print(cost_h)

#---------------------------------normalizado----------------------------------

def f_norm(v,x):
    return f(v,x) - mean / stdev
    
def model_norm(w,v,x_p):
    a = w[0] + (f_norm(v,x_p) * w[1])
    return a
    
def NL_regression_norm(w,v,x,y):
    P = len(x)
    cost = 0
    for p in range(P):
        cost = cost + (model_norm(w,v,x[p]) - y[p])**2
    cost /= P
    return cost

def grad_NL_regression_norm(w,v,x,y):
    P = len(x)
    grad = np.zeros(4)
    for p in range(P):
        k = 2*(model_norm(w,v,x[p]) - y[p])
        grad[0] += k
        grad[1] += k * f_norm(v,x[p])
        grad[2] += k * np.cos(v[0] + x[p] * v[1])*w[1]
        grad[3] += k * np.cos(v[0] + x[p] * v[1])*w[1]*x[p]
    grad /= P
    return grad

def gradient_descent_norm(x,y,alpha,max_its):
    w_theta = [1,1,1,1] #valores iniciais
    w_theta = np.array(w_theta)
    weight_history = [w_theta]
    w = np.zeros(2)
    v = np.zeros(2)
    w[0] = w_theta[0]
    w[1] = w_theta[1]
    v[0] = w_theta[2]
    v[1] = w_theta[3]
    cost_history = [NL_regression_norm(w,v,x,y)]
    tol = 0.1
    for k in range(max_its):
        w_theta = w_theta - alpha * grad_NL_regression_norm(w,v,x,y)
        weight_history.append(w_theta)
        cost_history.append(NL_regression_norm(w,v,x,y))
        g = grad_NL_regression(w,v,x,y)
        norm = np.linalg.norm(g)
        if norm < tol:
            alpha = alpha/10
            tol = tol/10
        w[0] = w_theta[0]
        w[1] = w_theta[1]
        v[0] = w_theta[2]
        v[1] = w_theta[3]
    return weight_history, cost_history

[w_h_norm, cost_norm] = gradient_descent_norm(X,Y,alpha,max_its)


#-------------plot para comparar normalizado x sem normalizar (custo x steps)

it = list(range(0,max_its+1)) #gerando 500 pontos para o eixo X do gráfico
plt.figure(figsize=(14,8), dpi=80)
plt.scatter(it, cost, marker='o', color='b', label='Cost')
plt.scatter(it, cost_norm, marker='.', color='r', label='Normalized Cost')
plt.title("Cost Function x Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost Value")
plt.legend()
plt.show()
