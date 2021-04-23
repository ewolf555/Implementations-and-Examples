# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:22:25 2021

@author: Elias Wolf
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import norm

"""
Implementation of the Bootstrap Particle Filter and the Auxilliary Particle
Filter for a stochastic volatility model of the form

              y_t = gamma_0 + eps_t*sigma_t^2
              log(sigma_t^2) = delta_0 + delta_1*log(sigma_{t-1}^2) + v_t

Weights og the bootsstrap particle filter are resampled once the 
effective sample size drops below a certain threshold. 
Both filters are compared by their MSEs at the end of the code. 
"""

# Params
gamma_0 =  2.74

delta_0 = 0.5
delta_1 = 0.7

# Observation Noise
R = 1
# State Noise
Q = 1

# State equation
def mean_state_t(sigma_lagged, delta_0, delta_1):
    
    state_mean = delta_0 + delta_1*sigma_lagged
    
    return(state_mean)

# Simulate some data
y_t = []
ln_sigma_2_t = [0]

for i in range(0, 200):
    
    eps_t = np.random.normal(0, np.sqrt(R), 1)
    v_t = np.random.normal(0, np.sqrt(Q), 1)

    ln_sigma_2 = mean_state_t(ln_sigma_2_t[i], delta_0, delta_1) + v_t
    
    y = gamma_0 + eps_t*np.sqrt(np.exp(ln_sigma_2))
    
    y_t.append(y[0])
    ln_sigma_2_t.append(ln_sigma_2[0])


ln_sigma_2_t = np.asarray(ln_sigma_2_t)[1:]
y_t = np.asarray(y_t)


plt.plot(y_t, color="k", label="$y_{t+1}$")
plt.title("Plot for $y_{t+1}$")
plt.legend()
plt.grid()
plt.show()


plt.plot(ln_sigma_2_t, color="r", label="$\log(\sigma_t^2)$")
plt.title("Plot for $\log(\sigma_t^2)$")
plt.legend()
plt.grid()
plt.show()

# Bootstrap Particle Filter
M = 10000

prior_init = np.random.normal(loc=0, scale=np.sqrt(R), size=M)

posterior_bpf = [prior_init]
posterior_mean_bpf = []

weights_init = np.ones(M)/M

for i in range(200):


    # State Update
    epsilon = np.random.normal(size=M, loc=0, scale=np.sqrt(Q))

    states_update = mean_state_t(posterior_bpf[i], 
                                 delta_0, delta_1) + epsilon

    # Compute Incremental Weights
    weights = norm.pdf(y_t[i], 
                      loc=gamma_0,
                      scale=np.sqrt(np.exp(states_update)*R))

    # Compute normalized weights Weights
    weights_update = weights_init*weights
    
    weights_norm = weights_update/sum(weights_update)
    
    
    # Resample the particles
    state_index = np.random.choice(np.arange(0,M), 
                                   size=M, 
                                   p=weights_norm)
    
    states_posterior = states_update[state_index]

    # Compute effective sample size
    effective_size = 1/sum(weights_norm**2)
        
    if effective_size < 10:
        weights_init = np.ones(M)/M
    else:
        weights_init = weights_update    
      
    posterior_bpf.append(states_posterior)
    posterior_mean_bpf.append(np.mean(states_posterior))
    

# Auxilliary Particle Filter
# State densitiy at t=0 (prior beliefs about the state)
M = 10000
P = 30000

prior_init = np.random.normal(loc=0, scale=np.sqrt(R), size=M)

posterior_aux = [prior_init]
posterior_mean_aux = []

for i in range(200):
    
    #print(i)
    
    # Step 1: Sample indizes and resample states
    # State Update    
    states_mu = mean_state_t(posterior_aux[i], delta_0, delta_1)
    
    # Compute Incremental Weights
    weights_idx = norm.pdf(y_t[i], loc=gamma_0,
                           scale=np.sqrt(np.exp(states_mu)*R))

    # Compute Weights
    weights_idx_norm = weights_idx/sum(weights_idx)

    # Resample indices
    lambda_index = np.random.choice(np.arange(0,M), 
                                    size=P, 
                                    p=weights_idx_norm)

    # Associate with states
    states_resample_mu = states_mu[lambda_index]
    posterior_resample = posterior_aux[i][lambda_index]

    # Second Step: Update states
    # simulate states foreward
    epsilon = np.random.normal(size=P, loc=0, scale=np.sqrt(Q))

    states_sim = mean_state_t(posterior_resample, delta_0, delta_1) + epsilon
    
    # compute weights
    weights_2 = norm.pdf(y_t[i], loc=gamma_0,
                         scale=np.sqrt(np.exp(states_sim)*R))

    weights_1 = norm.pdf(y_t[i], loc=gamma_0,
                         scale=np.sqrt(np.exp(states_resample_mu)*R))
    
    weights_states = weights_2/weights_1

    weights_states_norm = weights_states/sum(weights_states)
    
    # resample states
    state_index = np.random.choice(np.arange(0,P), size=M, 
                                   p=weights_states_norm)

    states_posterior = states_sim[state_index]

    posterior_aux.append(states_posterior)
    posterior_mean_aux.append(np.mean(states_posterior))



# Plot filtered states
plt.plot(ln_sigma_2_t, color="b", label="$\log(\sigma_t^2)$")
plt.plot(posterior_mean_bpf, color='k', label="Filtered state BPF")
plt.title("Filtered State of $\log(\sigma_t^2)$ - Bootstrap Particle Filter")
plt.legend()
plt.grid()
plt.show()

plt.plot(ln_sigma_2_t, color="b", label="$\log(\sigma_t^2)$")
plt.plot(posterior_mean_aux, color='k', label="Filtered state AuxPF")
plt.title("Filtered State of $\log(\sigma_t^2)$ - Auxilliary Particle Filter")
plt.legend()
plt.grid()
plt.show()

# Compare MSEs
print("MSE(Auxilliary Particle Filter): ", sum((ln_sigma_2_t - posterior_mean_aux)**2))
print("MSE(Bootstrap Particle Filter): ", sum((ln_sigma_2_t - posterior_mean_bpf)**2))
