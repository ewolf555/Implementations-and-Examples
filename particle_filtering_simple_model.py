# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 23:16:02 2021

@author: Elias Wolf
"""

import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# simulate data

x_0 = 0
Q = 10
R = 1

x_t = [x_0]
y_t = []


def f_t(x, t):
    
    x_t = x/2 + 25*x/(1+x**2) + 8*np.cos(1.2*t)
    
    return(x_t)
    

for i in range(1, 102):
    
    x_i = f_t(x_t[i-1], i)
    
    x_i = x_i + np.random.normal(0, np.sqrt(Q), 1)
    
    x_t.append(x_i)
    
    y_t.append((x_i**2)/20 + np.random.normal(0, np.sqrt(R), 1))
    
x_t = np.concatenate(x_t[1:])
y_t = np.concatenate(y_t)

# Plot simulated Series
plt.plot(x_t, label='State', color='r')
plt.plot(y_t, label='Measurement', color='k')
plt.title('Simulated Series')
plt.legend()
plt.grid()
plt.show()

# Bootstrap Particle Filter


# State densitiy at t=0 (prior beliefs about the state)

M = 50000

prior_init = np.random.normal(loc=0, scale=np.sqrt(Q), size=M)

posterior = [prior_init]
posterior_mean = []
posterior_max = []
weights_init = np.ones(M)/M
for i in range(101):
    
    #print(i)
    epsilons = np.random.normal(size=M, loc=0, scale=np.sqrt(Q))
    
    states = f_t(posterior[i], i+1) + epsilons
        
    # observation update
    p_n = norm.pdf(y_t[i], 
                   loc=(states**2)/20,
                   scale=np.sqrt(R))
    
    # weights
    weights_update = weights_init*p_n
    
    Weights_norm = weights_update/sum(weights_update)
    
    
    # Resample the particles (Bootstrap Particle Filter if phi_init=1)
    state_index = np.random.choice(np.arange(0,M), 
                                   size=M, 
                                   p=Weights_norm)
    
    states_resample = states[state_index]
    epsilons_resample = epsilons[state_index]
    
        
    # Compute effective sample size
    effective_size = 1/sum(Weights_norm**2)
        
    if effective_size < 20:
        weights_init = np.ones(M)/M
        #print('weights degenerated!')
    else:
        weights_init = weights_update        
    
    posterior.append(states_resample)
    posterior_mean.append(np.mean(states_resample))
    posterior_max.append(np.max(states_resample))



plt.plot(x_t, color='k', label='true state', linestyle ='--')
plt.plot(posterior_mean, color='red', label='filtered posterior mean')
plt.legend()
plt.grid()
plt.show()
    


###############################################################################
###############################################################################
        

# Tempered Particle Filter Implementation

M = 50000
r_star = 2
c_init = 2


prior_init = np.random.normal(loc=0, scale=np.sqrt(Q), size=M)

posterior_tpf = [prior_init]
posterior_mean_tpf = []
posterior_max_tpf = []

for i in range(101):
    
    print("------------------------------------------\n")
    print("\n Particle Filtering t = " + str(i) + "\n")
    
    # State Update
    epsilons = np.random.normal(size=M, loc=0, scale=np.sqrt(Q))
    
    states = f_t(posterior_tpf[i], i+1) + epsilons 

    # Function for the initial inefficiency Ratio and rootfinding procedure
    def init_InEff(phi_1, state, measurement, Sigma_u):
    
        N = len(state)
    
        p_n = norm.pdf(measurement, 
                       loc=(state**2)/20,
                       scale=np.sqrt(Sigma_u/phi_1))
    
        # Stable implementation for normalized weights using log weights
        p_n = np.log(p_n) - np.max(np.log(p_n))
    
        new_Weights = np.exp(p_n)/np.mean(np.exp(p_n))
    
        init_InEff = sum(new_Weights**2)/N
    
        return(init_InEff)
    
    def optim_func_init(x, r):
        return(init_InEff(x, states, y_t[i], R) - r)
    
    # Calculate Boundaries
    upper_bound = optim_func_init(1, r_star)
    lower_bound = optim_func_init(0.01, r_star)
    
    print("Upper bound: " + str(upper_bound))
    print("Lower bound: " + str(lower_bound))
    
    # Check if Tempering Condition is satisfied
    if np.sign(upper_bound)*np.sign(lower_bound) == 1:
    
        phi_init = 1  # set phi_new = 1 to skip tempering
    
    # Find new value for phi if condition is not satisfied.
    else:  
        
        phi_init = brentq(optim_func_init, a=0.01, b=1, args=(r_star))
           
    # Compute Incremental Weights
    p_1 = norm.pdf(y_t[i], loc=(states**2)/20,
                   scale=np.sqrt(R/phi_init))
    
    # Compute Weights
    p_1 = np.log(p_1) - np.max(np.log(p_1))
    
    Weights_norm = np.exp(p_1)/sum(np.exp(p_1))
    
  
    # Resample the particles (Bootstrap Particle Filter if phi_init=1)
    state_index = np.random.choice(np.arange(0,M), 
                                   size=M, 
                                   p=Weights_norm)
    
    states_resample = states[state_index]
    states_lagged_resample = posterior_tpf[i][state_index]
    epsilons_resample = epsilons[state_index]

    
    count = 0
    phi_old = phi_init
    c = c_init
    
    print("Initial Proposal Variance: " + str(c))
    print("Initial Phi: " + str(phi_init), "\n ----------------- \n" )
    
    
    while phi_old < 1:
        
        count += 1
    
        # Function for the inefficiency ratio
        def inEff(phi_new, phi_old, state, measurement, Sigma_u):
        
            N = len(state)
          
            p_new = norm.pdf(measurement, loc=(state**2)/20,
                             scale=np.sqrt(Sigma_u/phi_new))
        
            p_old = norm.pdf(measurement, loc=(state**2)/20,
                             scale=np.sqrt(Sigma_u/phi_old))
        
            weights_phi = np.log(p_new) - np.log(p_old)
        
            weights_phi = weights_phi - np.max(weights_phi)
        
            new_Weights = np.exp(weights_phi)/np.mean(np.exp(weights_phi))
        
            inEff = sum(new_Weights**2)/N
        
            return(inEff)
        
        def optim_func_temp(x, phi_old, r, states_resample, measurement, R):
            return(inEff(x, phi_old, states_resample, measurement, R) - r)
        
        # Calculate Boundaries
        upper_bound = optim_func_temp(1, phi_old, r_star, states_resample, y_t[i], R)
        lower_bound = optim_func_temp(phi_old, phi_old, r_star, states_resample, y_t[i], R)
        
        # Check if Tempering Condition is satisfied
        if np.sign(upper_bound)*np.sign(lower_bound) == 1:
        
            print("Tempering Ended after " + str(count) + " Iterations \n")
            #print(inEff(1, phi_old, states_resample, x_t[i], y_t[i], R))
        
            phi_new = 1  # set phi_new = 1 to exit loop
        
        # Find new value for phi if condition is not satisfied.
        else:  
        
            print("Tempering step: " + str(count))
            # Find new value to satisfy inefficiency ratio = r_star
        
            phi_new = brentq(optim_func_temp, a=phi_old, b=1, args=(phi_old, r_star, states_resample, y_t[i], R))
               
        print("Phi_new: " + str(phi_new))
        
        # Calculate the normalized weights
        def Weights_phi(phi_new, phi_old, state, measurement, Sigma_u):

            p_new = norm.pdf(measurement, loc=(state**2)/20,
                             scale=np.sqrt(Sigma_u/phi_new))
        
            p_old = norm.pdf(measurement, loc=(state**2)/20,
                             scale=np.sqrt(Sigma_u/phi_old))
            
            weights_phi = np.log(p_new) - np.log(p_old)
                
            weights_phi = weights_phi - np.max(weights_phi)
            
            new_Weights = np.exp(weights_phi)/sum(np.exp(weights_phi))
    
            return(new_Weights)
        
        # Weights in the updating step, could potentially be 
        # overritten to be Weights_norm based on phi_init to 
        # save space. 
        Weights_phi_star = Weights_phi(phi_new, phi_old,
                                       states_resample, y_t[i], R)
        
        # Selection
            
        # Resample the particles based on Weights_phi_star
        state_index = np.random.choice(np.arange(0,M), 
                                       size=M, 
                                       p=Weights_phi_star)
           
        states_resample = states_resample[state_index]
        states_lagged_resample = states_lagged_resample[state_index]
        epsilons_resample = epsilons_resample[state_index]
        
        
        # Save new states, new epsilons and set new phi to old phi
        
        transition_check = states_resample == (f_t(states_lagged_resample, i+1) + epsilons_resample)
        
        if all(transition_check):
            print("Selection: Transition Equation statisfied")

        # Mutation Step
            
        # Function for Mutation Step
        def mutation_step(eps, state, c, state_lagged, measurement, phi, Sigma_u):
            
            M = len(eps)
         
            # Draw from proposal distribution
            e = np.random.normal(loc=eps, scale=c, size=M)

            # Compute states
            mean_state =  f_t(state_lagged, i+1)
         
            state_proposal = mean_state + e
            
            state_old = state
            
            # Proposal
            p_proposal = norm.pdf(measurement, loc=(state_proposal**2)/20,
                                  scale=np.sqrt(Sigma_u/phi))*norm.pdf(e)
            
            p_old = norm.pdf(measurement, loc=(state_old**2)/20,
                             scale=np.sqrt(Sigma_u/phi))*norm.pdf(eps)
                  
            # Acceptance Ratio
            ratio = np.fmin(1, np.exp(np.log(p_proposal) - np.log(p_old)))
        
            rand = np.random.uniform(size=M)
            
            eps_update = np.where(rand < ratio, e, eps)
            state_update = np.where(rand < ratio, state_proposal, state_old)
            
            rejection_ratio = np.count_nonzero(ratio < rand)/M
            
            return(eps_update, state_update, rejection_ratio)
        
        # Empty list for Rejection Rates
        rej_ratio_all = []
        eps_up = epsilons_resample
        state_up = states_resample
        

        for j in range(2):
            
            eps_up, state_up, rej_ratio = mutation_step(eps_up, state_up, c,  
                                                        states_lagged_resample, 
                                                        y_t[i], phi_new, R)
            
            rej_ratio_all.append(rej_ratio)
       
        
        # Adjust c based on rejection Rate
            
        # Function for adaptive step-size based on rejection rate
        def f(x):
        
            logistic_comp = np.exp(20*(x-0.4))/(1+np.exp(20*(x-0.4)))
        
            return(0.95 + 0.1*logistic_comp)
            
        c = c*f(1-np.mean(rej_ratio_all))
            
        print("Proposal Variance: " + str(c), "\n")
            
        
        transition_check = states_resample == (f_t(states_lagged_resample, i+1) + epsilons_resample)
        
        if all(transition_check):
            print("Mutation: Transition Equation statisfied")
            
        # Save new states, new epsilons and set new phi to old phi
        epsilons_resample = eps_up
        states_resample = state_up
        
        # Adjust c based on rejection Rate
            
        # Function for adaptive step-size based on rejection rate
        def f(x):
        
            logistic_comp = np.exp(20*(x-0.4))/(1+np.exp(20*(x-0.4)))
        
            return(0.95 + 0.1*logistic_comp)
            
        c = c*f(1-np.mean(rej_ratio_all))
        
        print("Avrg. Acceptance Ratio: " + str(1 - np.mean(rej_ratio_all)))
        print("New Proposal Variance: " + str(c), "\n")
        
        phi_old = phi_new
            
    posterior_tpf.append(states_resample)
    posterior_mean_tpf.append(np.mean(states_resample))
    posterior_max_tpf.append(np.max(states_resample))


plt.plot(x_t, color='k', label='true state', linestyle ='--')
plt.plot(posterior_mean, color="red", label='filtered posterior mean (bpf)')
plt.plot(posterior_mean_tpf, color='blue', label='filtered posterior mean (tpf)')
plt.title("Filtered States - Bootstrap vs. Tempered Particle Filter")
plt.legend()
plt.grid()
plt.show()



# RMSE

rmse_bpf = np.sqrt(sum((x_t - np.asarray(posterior_mean))**2))
rmse_tpf = np.sqrt(sum((x_t - np.asarray(posterior_mean_tpf))**2))

print("rmse_bpf = ", rmse_bpf)
print("rmse_tpf = ", rmse_tpf)




