# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 19:06:09 2020

@author: Elias Wolf
"""

import numpy as np
import scipy as scp
import matplotlib.pyplot as plt


"""
Example for the implementaiton of a simple bootstrap particle filter based on
simulated data for the popular non-linear state-space model of the from

     z_t = x_t^2/20 + e, with e~N(0,R)
     x_t = x_{t-1}/2 + 25*x_{t-1}/(1+x_{t-1}^2) + 8*cos(t) + u, with u~N(0,Q) 

Weights are resampled if the efficienty ration drops below a threshold of 20.
"""

# simulate data

x_0 = 0
Q = 10
R = 1

x_t = [x_0]
z_t = []


def f_t(x, t):
    
    x_t = x/2 + 25*x/(1+x**2) + 8*np.cos(1.2*t)
    
    return(x_t)
    

for i in range(1, 102):
    
    x_i = f_t(x_t[i-1], i)
    
    x_i = x_i + np.random.normal(0, np.sqrt(Q), 1)
    
    x_t.append(x_i)
    
    z_t.append((x_i**2)/20 + np.random.normal(0, np.sqrt(R), 1))
    
x_t = x_t[1:]

# Plot simulated Series
plt.plot(x_t, color="k", label="$x_t$")
plt.plot(z_t, color="b", label="$z_t$")
plt.title("Simulated States and Measurements")
plt.legend()
plt.grid()
plt.show()


# State densitiy at t=0 (prior beliefs about the state)

n_weights = 1000

prior_init = np.random.normal(loc=0, scale=np.sqrt(Q), size=n_weights)


posterior = [prior_init]
posterior_mean = []
posterior_max = []
weights_init = np.ones(n_weights)/n_weights

for i in range(101):
    
    #print(i)
    
    # sample x_t+1
    state_update = np.random.normal(loc=f_t(posterior[i], i+1), 
                                            scale=np.sqrt(Q))
    """
    plt.hist(state_update)
    plt.title('Distribution of State')
    plt.show()
    """
    
    # observation update
    obs_update = norm.pdf(z_t[i], 
                          loc=(state_update**2)/20,
                          scale=np.sqrt(R))
    """
    plt.hist(obs_update)
    plt.title('Probability of Observations')
    plt.show()
    """
    
    # weights
    weights_update = weights_init*obs_update
    
    weights_norm = weights_update/sum(weights_update)
    
    """
    plt.hist(weights_norm)
    plt.title('Normalized Weights')
    plt.show()
    """
    
    state_posterior = np.random.choice(state_update, size=n_weights, 
                                       p=weights_norm)
    """
    plt.hist(state_posterior)
    plt.title('State Posterior')
    plt.show()
    """
    # Compute effective sample size
    effective_size = 1/sum(weights_norm**2)
        
    if effective_size < 20:
        weights_init = np.ones(n_weights)/n_weights
        #print('weights degenerated!')
    else:
        weights_init = weights_update        
    
    posterior.append(state_posterior)
    posterior_mean.append(np.mean(state_posterior))
    posterior_max.append(np.max(state_posterior))

    

plt.plot(x_t, color='k', label='true state $x_t$', linestyle ='--')
plt.plot(posterior_mean, color='red', label='filtered posterior mean $\hat{x}_t$')
plt.legend()
plt.title("Filtered Series")
plt.grid()
plt.show()
    
    





