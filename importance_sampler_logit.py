# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:35:08 2020

@author: Elias Wolf
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import bernoulli
import statsmodels.api as sm


# Simulate Data of a simple logit model

# Set parameters and generate x variable
theta_0 = 1
theta_1 = 2
N = 200

x_vals = norm.rvs(scale = 1, size = N)

lin_comb = theta_0 + theta_1*x_vals

def sigmoid(x):
    
    return(1/(1+np.exp(-x)))
    
prob = sigmoid(lin_comb)

def labels(x):
    return(bernoulli.rvs(prob))
    
y = labels(prob)

X = np.stack((np.ones(N), x_vals), axis = 1)

# Estimate Data with package
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary())


# Define Likelihood Function
def Lik(theta, feats, labels):

    #N = len(labels)

    x = feats
    y = labels
    
    scores = np.dot(x, theta)
    
    pi = sigmoid(scores)
    
    probs = (pi**y)*(1-pi)**(1-y)
    
    joint_prob = np.prod(probs)
    
    return(joint_prob)



# Define Gaussian Prior
def prior(theta, mu, sigma):
    
    sigma = sigma*np.eye(2)
    mu = mu*np.array([1, 1])
    
    return(multivariate_normal.pdf(theta, mu, sigma))
   
# Draw from Gaussian Proposal Distribution                 
def proposal(theta, mu, sigma):
    
    sigma = sigma*np.eye(2)
    mu = mu*np.array([1, 1])
    
    return(multivariate_normal.pdf(theta, mu, sigma))


# Approximate Posterior via Importance Sampling
w = []
theta_draws = []

sigma_prior = 5
mu_prior = [0, 0]

sigma_prop = 10
mu_prop = 2

N_draws = 50000

for i in range(N_draws):

    theta_i = multivariate_normal.rvs(size = 1, 
                                      mean = mu_prop*np.ones(2), 
                                      cov = sigma_prop*np.eye(2))


    nom = Lik(theta_i, test_feats, test_labels)*prior(theta_i, mu_prior, sigma_prior)

    denom = proposal(theta_i, mu_prop, sigma_prop)

    w_theta = nom/denom

    w.append(w_theta)
    theta_draws.append(theta_i)
    
    
# Normalize the weights   
w_norm = np.asarray(w)/sum(w)

# Resample 
theta_draws = np.asarray(theta_draws)

resample_index = np.random.choice(theta_draws.shape[0], size=50000, p=w_norm)

posterior_joint = theta_draws[resample_index]


theta_0, theta_1 = posterior_joint.reshape(-1, 2).T

plt.hist(theta_0, bins=15)
plt.title("Posterior of Theta 0")
plt.grid()
plt.show()

plt.hist(theta_1, bins=15)
plt.title("Posterior of Theta 1")
plt.grid()
plt.show()

print(np.mean(theta_0))
print(np.mean(theta_1))



