# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 18:37:50 2021

@author: eliaswolf
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(36)

# Define Process y
def y(N, gamma, sigma):
    
    mu = np.random.normal(loc=0, scale=gamma)
    
    eps = np.random.normal(size=N, loc=0, scale=sigma)
    
    y_i_t = mu + eps
    
    return(y_i_t, mu)


# Set Parameters
gamma = 2.5
sigma = 1
N = 100


# Plot one realization of y
y_i, mu_i = y(N, gamma, sigma)

plt.plot(y_i, label="$y_t^{(i)}$", color="b")
plt.hlines(mu_i, xmin=0, xmax=100, color="k", 
           label="$\mu^{(i)}$ = " + str(np.round(mu_i, 2)))
plt.grid()
plt.legend()
plt.title("One Realization $y_t^{(i)} of $y_t$")
plt.xlabel("T")
plt.ylabel("$y_t^{(i)}$")
plt.show()


# Plot 4 realizations of y
color_list = ["green", "red", "blue", "purple"]

for i in range(4):
    y_i, mu_i = y(N, gamma, sigma)
    
    plt.plot(y_i, color=color_list[i])
    plt.hlines(mu_i, xmin=0, xmax=100, color=color_list[i], linestyle="--")

plt.grid()
plt.title("4 Realizations $y_t^{(i)}$ with i = (1,...,4) of $y_t$")
plt.xlabel("T")
plt.ylabel("$y_t^{(i)}$")
plt.show()



    