# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:30:58 2021

@author: eliaswolf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.stats import multivariate_normal
from scipy.stats import uniform
from scipy.stats import gamma

# simulate data
x_t = np.random.exponential(scale=1, size=201)-1

# Params
gamma_0 =  2.74
gamma_1 = -1.19

delta_0 = 0.5
delta_1 = 0.8 #0.782533
delta_2 = 0.5 #-0.034730


# Observation Noise
R = 1
# State Noise
Q = 1

def mean_state_t(x, sigma_lagged, delta_0, delta_1, delta_2):
    
    state_mean = delta_0 + delta_1*x + delta_2*sigma_lagged
    
    return(state_mean)
    

def mean_obs_t(x, gamma_0, gamma_1):
    
    obs_mean = gamma_0 + gamma_1*x

    return(obs_mean)    
    

y_t = []

ln_sigma_2 = [0]

for i in range(1, 201):
    
    eps_t = np.random.normal(0, np.sqrt(R), 1)
    v_t = np.random.normal(0, np.sqrt(Q), 1)

    ln_sigma_sqrd = mean_state_t(x_t[i], ln_sigma_2[i-1], delta_0, delta_1, delta_2) + v_t
    
    sigma_sqrd = np.exp(ln_sigma_sqrd)
    
    y_forecast = mean_obs_t(x_t[i], gamma_0, gamma_1) + eps_t*np.sqrt(sigma_sqrd)
    
    y_t.append(y_forecast[0])
    ln_sigma_2.append(np.log(sigma_sqrd[0]))

ln_sigma_2 = np.asarray(ln_sigma_2)[1:]
x_t = x_t[1:]
y_t = np.asarray(y_t)


# Plots of realizations
plt.plot(x_t, color="r", label="$x_{t}$")
plt.plot(y_t, color="k", label="$y_{t+1}$")
plt.title("Plot for $y_{t+1}$ and $x_{t}$")
plt.legend()
plt.grid()
plt.show()


plt.plot((np.asarray(y_t)-mean_obs_t(x_t, gamma_0, gamma_1))**2, color="k", label="$(y_{t+1}-E[{y}_{t+1}])^2$")
plt.plot(np.exp(np.asarray(ln_sigma_2)), color="r", label="$\sigma_t^2$")
plt.title("Plot for $y_{t+1}^2$")
plt.legend()
plt.grid()
plt.show()



plt.plot(np.asarray(ln_sigma_2), color="r", label="$\log(\sigma_t^2)$")
plt.plot(x_t, color='b', label="$x_t$")
#plt.plot(x_t, color="k", label="$x_{t}$")
plt.title("Plot for $\log(\sigma_t^2)$ and $x_t$")
plt.legend()
plt.grid()
plt.show()

plt.hist(x_t)
plt.grid()
plt.title('Histogram of Exogenous Variable')
plt.show()

plt.hist(y_t)
plt.grid()
plt.title('Histogram of Endogenous Variable')
plt.show()



params_t = [delta_0, delta_1, delta_2, Q]
params_m = [gamma_0, gamma_1, R]
c_init = 2
r_star = 1.5
M = 20000


prior_init = np.random.normal(loc=0, scale=np.sqrt(1), size=M)

posterior_tpf = [prior_init]
posterior_mean_tpf = []
loglik_tpf = []


for i in range(200):
    

    #print("------------------------------------------\n")
    #print("\n Particle Filtering t = " + str(i) + "\n")
    
    # State Update
    epsilons = np.random.normal(size=M, loc=0, scale=np.sqrt(params_t[3]))
    
    states = mean_state_t(x_t[i], posterior_tpf[i], 
                          params_t[0], params_t[1], params_t[2]) + epsilons

    # Function for the initial inefficiency Ratio and rootfinding procedure
    def init_InEff(phi_1, state, exog, measurement, Sigma_u):
    
        N = len(state)
    
        p_n = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                       scale=np.sqrt(np.exp(state)*Sigma_u/phi_1))
    
        # Stable implementation for normalized weights using log weights
    
        p_n = np.log(p_n) - np.max(np.log(p_n))
    
        new_Weights = np.exp(p_n)/np.mean(np.exp(p_n))
    
        init_InEff = sum(new_Weights**2)/N
    
        return(init_InEff)
    
    def optim_func_init(x, r):
        return(init_InEff(x, states, x_t[i], y_t[i], params_t[2]) - r)
    
    upper_bound = optim_func_init(1, r_star)
    lower_bound = optim_func_init(0.0001, r_star)
    
    #print("Upper bound: " + str(upper_bound))
    #print("Lower bound: " + str(lower_bound))
    
    # Check if Inefficiency Ratio is below threshold
    if np.sign(upper_bound)*np.sign(lower_bound) == 1:
    
        phi_init = 1  # set phi_new = 1 to skip tempering
    
    # Find new value for phi if condition is not satisfied.
    else:  
        
        phi_init = brentq(optim_func_init, a=0.0001, b=1, args=(r_star))
    
    # Compute Incremental Weights
    p_1 = norm.pdf(y_t[i], loc=mean_obs_t(x_t[i], params_m[0], params_m[1]),
                   scale=np.sqrt(np.exp(states)*params_m[2]/phi_init))
    
    loglik_increment = np.log(np.mean(p_1))
    #print("LogLik Increment: ", loglik_increment)
    
    # Compute Weights
    p_1 = np.log(p_1) - np.max(np.log(p_1))
    
    Weights_norm = np.exp(p_1)/sum(np.exp(p_1))
    
    # Resample the particles (Bootstrap Particle Filter if phi_init=1)
    state_index = np.random.choice(np.arange(0,M), 
                                   size=M, 
                                   p=Weights_norm)
    
    states_resample = states[state_index]
    epsilons_resample = epsilons[state_index]
    states_lagged_resample = posterior_tpf[i][state_index]
    
    
    
    count = 0
    phi_old = phi_init
    c = c_init
    
    #print("Initial Phi: " + str(phi_init), "\n ----------------- \n" )
   
    # Tempering 
    while phi_old < 1:
        
        count += 1
    
        # Function for the inefficiency ratio
        def inEff(phi_new, phi_old, state, exog, measurement, Sigma_u):
        
            N = len(state)
          
            p_new = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                             scale=np.sqrt(np.exp(state)*Sigma_u/phi_new))
        
            p_old = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                             scale=np.sqrt(np.exp(state)*Sigma_u/phi_old))
        
            weights_phi = np.log(p_new) - np.log(p_old)
        
            weights_phi = weights_phi - np.max(weights_phi)
        
            new_Weights = np.exp(weights_phi)/np.mean(np.exp(weights_phi))
        
            inEff = sum(new_Weights**2)/N
        
            return(inEff)
        
        # Find new value to satisfy inefficiency ratio = r_star
        def optim_func_temp(x, phi_old, r):
            return(inEff(x, phi_old, states_resample, x_t[i], y_t[i], params_m[2]) - r)
        
        # Calculate Boundaries
        upper_bound = optim_func_temp(1, phi_old, r_star)
        lower_bound = optim_func_temp(phi_old, phi_old, r_star)
        
        
        # Check if Tempering Condition is satisfied
        if np.sign(upper_bound)*np.sign(lower_bound) == 1:
        
            #print("Tempering Ended after " + str(count) + " Iterations \n")
            #print(inEff(1, phi_old, states_resample, x_t[i], y_t[i], R))
        
            phi_new = 1  # set phi_new = 1 to exit loop
        
        # Find new value for phi if condition is not satisfied.
        else:  
        
            #print("Tempering step: " + str(count))
            # Find new value to satisfy inefficiency ratio = r_star
        
            phi_new = brentq(optim_func_temp, a=phi_old, b=1, args=(phi_old, r_star))
               
        #print("Phi_new: " + str(phi_new))
        
        # Calculate the normalized weights
        def Weights_phi(phi_new, phi_old, state, exog, measurement, Sigma_u):

            p_new = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                             scale=np.sqrt(np.exp(state)*Sigma_u/phi_new))
            
            p_old = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                             scale=np.sqrt(np.exp(state)*Sigma_u/phi_old))
            
            log_weights_phi = np.log(p_new) - np.log(p_old)
                
            return(log_weights_phi)
        
        # Weights in the updating step, could potentially be 
        # overwritten to be Weights_norm based on phi_init to 
        # save space. 
        
        Weights_phi_star = Weights_phi(phi_new, phi_old,
                                       states_resample, 
                                       x_t[i], y_t[i], params_m[2])
        
        # Add Likelihood Increment
        loglik_increment = loglik_increment + np.log(np.mean(np.exp(Weights_phi_star))) 
        
        weights_phi = Weights_phi_star - np.max(Weights_phi_star)
            
        weights_phi_star_norm = np.exp(weights_phi)/sum(np.exp(weights_phi))
        
        
        # Selection
            
        # Resample the particles based on Weights_phi_star
        state_index = np.random.choice(np.arange(0,M), 
                                       size=M, 
                                       p=weights_phi_star_norm)
        
        
        states_resample = states_resample[state_index]
        epsilons_resample = epsilons_resample[state_index]
        states_lagged_resample = states_lagged_resample[state_index]
        
        # Mutation Step
            
        # Function for Mutation Step
        def mutation_step(eps, state, s_lagged, exog, measurement, phi, Sigma_u, c):
            
            M = len(eps)
         
            # Draw from proposal distribution
            e = np.random.normal(loc=eps, scale=c, size=M)

            # Compute states
            mean_state = mean_state_t(exog, s_lagged, 
                                      params_t[0], params_t[1], params_t[2]) 
         
            state_proposal = mean_state + e
            
            state_old = state
            
            # Proposal
            p_proposal = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                                  scale=np.sqrt(np.exp(state_proposal)*Sigma_u/phi))*norm.pdf(e)
            
            p_old = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                             scale=np.sqrt(np.exp(state_old)*Sigma_u/phi))*norm.pdf(eps)
                  
            # Acceptance Ratio
            ratio = np.fmin(1, np.exp(np.log(p_proposal) - np.log(p_old)))
        
            rand = np.random.uniform(size=M)
            
            eps_update = np.where(ratio < rand, eps, e)
            state_update = np.where(ratio < rand, state_old, state_proposal)
            
            rejection_ratio = np.count_nonzero(ratio < rand)/M
            
            return(eps_update, state_update, rejection_ratio)
        
        
        # Empty list for Rejection Rates
        rej_ratio_all = []
        eps_up = epsilons_resample
        states_up = states_resample
        
        for j in range(4):
            
            eps_up, states_up, rej_ratio = mutation_step(eps_up, states_up, 
                                                         states_lagged_resample, 
                                                         x_t[i], y_t[i], 
                                                         phi_new, params_m[2], c)
            rej_ratio_all.append(rej_ratio)  

        
        #print("Avrg. Acceptance Ratio: " + str(1 - np.mean(rej_ratio_all)))

        # Adjust c based on rejection Rate
            
        # Function for adaptive step-size based on rejection rate
        def f(x):
        
            logistic_comp = np.exp(20*(x-0.4))/(1+np.exp(20*(x-0.4)))
        
            return(0.95 + 0.1*logistic_comp)
            
        c = c*f(1-np.mean(rej_ratio_all))
            
        #print("Proposal Variance: " + str(c), "\n")
            
        # Save new states, new epsilons and set new phi to old phi
        epsilons_resample = eps_up
        states_resample = states_up
      
        phi_old = phi_new
            
    posterior_tpf.append(states_resample)
    posterior_mean_tpf.append(np.mean(states_resample))
    loglik_tpf.append(loglik_increment)



plt.plot(ln_sigma_2, color='k', label='true state', linestyle ='--')
plt.plot(posterior_mean_tpf, color = 'blue')
plt.grid()
plt.show()

plt.plot(np.exp(np.asarray(ln_sigma_2)/2), 
         color="k", label='true state', linestyle ='--')
plt.plot(np.exp(np.asarray(posterior_mean_tpf)/2), color = 'blue')
plt.legend()
plt.grid()
plt.show()


def z(params_m, params_t, x, y, M, c_init, r_star):
    
    prior_init = np.random.normal(loc=0, scale=np.sqrt(1), size=M)

    posterior_tpf = [prior_init]
    loglik_tpf = 0
    
    for i in range(200):

        #print("------------------------------------------\n")
        #print("\n Particle Filtering t = " + str(i) + "\n")
        
        # State Update
        epsilons = np.random.normal(size=M, loc=0, scale=np.sqrt(params_t[3]))
        
        states = mean_state_t(x_t[i], posterior_tpf[i], 
                              params_t[0], params_t[1], params_t[2]) + epsilons
    
        # Function for the initial inefficiency Ratio and rootfinding procedure
        def init_InEff(phi_1, state, exog, measurement, Sigma_u):
        
            N = len(state)
        
            p_n = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                           scale=np.sqrt(np.exp(state)*Sigma_u/phi_1))
        
            # Stable implementation for normalized weights using log weights
        
            p_n = np.log(p_n) - np.max(np.log(p_n))
        
            new_Weights = np.exp(p_n)/np.mean(np.exp(p_n))
        
            init_InEff = sum(new_Weights**2)/N
        
            return(init_InEff)
        
        def optim_func_init(x, r):
            return(init_InEff(x, states, x_t[i], y_t[i], params_t[2]) - r)
        
        upper_bound = optim_func_init(1, r_star)
        lower_bound = optim_func_init(0.001, r_star)
        
        #print("Upper bound: " + str(upper_bound))
        #print("Lower bound: " + str(lower_bound))
        
        # Check if Inefficiency Ratio is below threshold
        if np.sign(upper_bound)*np.sign(lower_bound) == 1:
        
            phi_init = 1  # set phi_new = 1 to skip tempering
        
        # Find new value for phi if condition is not satisfied.
        else:  
            
            phi_init = brentq(optim_func_init, a=0.001, b=1, args=(r_star))
        
        # Compute Incremental Weights
        p_1 = norm.pdf(y_t[i], loc=mean_obs_t(x_t[i], params_m[0], params_m[1]),
                       scale=np.sqrt(np.exp(states)*params_m[2]/phi_init))
        
        loglik_increment = np.log(np.mean(p_1))
        #print("LogLik Increment: ", loglik_increment)
        
        # Compute Weights
        p_1 = np.log(p_1) - np.max(np.log(p_1))
        
        Weights_norm = np.exp(p_1)/sum(np.exp(p_1))
        
        # Resample the particles (Bootstrap Particle Filter if phi_init=1)
        state_index = np.random.choice(np.arange(0,M), 
                                       size=M, 
                                       p=Weights_norm)
        
        states_resample = states[state_index]
        epsilons_resample = epsilons[state_index]
        states_lagged_resample = posterior_tpf[i][state_index]
        
        
        
        count = 0
        phi_old = phi_init
        c = c_init
        
        #print("Initial Phi: " + str(phi_init), "\n ----------------- \n" )
       
        # Tempering 
        while phi_old < 1:
            
            count += 1
        
            # Function for the inefficiency ratio
            def inEff(phi_new, phi_old, state, exog, measurement, Sigma_u):
            
                N = len(state)
              
                p_new = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                                 scale=np.sqrt(np.exp(state)*Sigma_u/phi_new))
            
                p_old = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                                 scale=np.sqrt(np.exp(state)*Sigma_u/phi_old))
            
                weights_phi = np.log(p_new) - np.log(p_old)
            
                weights_phi = weights_phi - np.max(weights_phi)
            
                new_Weights = np.exp(weights_phi)/np.mean(np.exp(weights_phi))
            
                inEff = sum(new_Weights**2)/N
            
                return(inEff)
            
            # Find new value to satisfy inefficiency ratio = r_star
            def optim_func_temp(x, phi_old, r):
                return(inEff(x, phi_old, states_resample, x_t[i], y_t[i], params_m[2]) - r)
            
            # Calculate Boundaries
            upper_bound = optim_func_temp(1, phi_old, r_star)
            lower_bound = optim_func_temp(phi_old, phi_old, r_star)
            
            
            # Check if Tempering Condition is satisfied
            if np.sign(upper_bound)*np.sign(lower_bound) == 1:
            
                #print("Tempering Ended after " + str(count) + " Iterations \n")
          
                phi_new = 1  # set phi_new = 1 to exit loop
            
            # Find new value for phi if condition is not satisfied.
            else:  
            
                #print("Tempering step: " + str(count))
                # Find new value to satisfy inefficiency ratio = r_star
            
                phi_new = brentq(optim_func_temp, a=phi_old, b=1, args=(phi_old, r_star))
                   
            #print("Phi_new: " + str(phi_new))
            
            # Calculate the normalized weights
            def Weights_phi(phi_new, phi_old, state, exog, measurement, Sigma_u):
    
                p_new = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                                 scale=np.sqrt(np.exp(state)*Sigma_u/phi_new))
                
                p_old = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                                 scale=np.sqrt(np.exp(state)*Sigma_u/phi_old))
                
                log_weights_phi = np.log(p_new) - np.log(p_old)
                    
                return(log_weights_phi)
            
            # Weights in the updating step, could potentially be 
            # overwritten to be Weights_norm based on phi_init to 
            # save space. 
            
            Weights_phi_star = Weights_phi(phi_new, phi_old,
                                           states_resample, 
                                           x_t[i], y_t[i], params_m[2])
            
            # Add Likelihood Increment
            loglik_increment = loglik_increment + np.log(np.mean(np.exp(Weights_phi_star))) 
            
            weights_phi = Weights_phi_star - np.max(Weights_phi_star)
                
            weights_phi_star_norm = np.exp(weights_phi)/sum(np.exp(weights_phi))
            
            
            # Selection
                
            # Resample the particles based on Weights_phi_star
            state_index = np.random.choice(np.arange(0,M), 
                                           size=M, 
                                           p=weights_phi_star_norm)
            
            
            states_resample = states_resample[state_index]
            epsilons_resample = epsilons_resample[state_index]
            states_lagged_resample = states_lagged_resample[state_index]
            
            # Mutation Step
                
            # Function for Mutation Step
            def mutation_step(eps, state, s_lagged, exog, measurement, phi, Sigma_u, c):
                
                M = len(eps)
             
                # Draw from proposal distribution
                e = np.random.normal(loc=eps, scale=c, size=M)
    
                # Compute states
                mean_state = mean_state_t(exog, s_lagged, 
                                          params_t[0], params_t[1], params_t[2]) 
             
                state_proposal = mean_state + e
                
                state_old = state
                
                # Proposal
                p_proposal = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                                      scale=np.sqrt(np.exp(state_proposal)*Sigma_u/phi))*norm.pdf(e)
                
                p_old = norm.pdf(measurement, loc=mean_obs_t(exog, params_m[0], params_m[1]),
                                 scale=np.sqrt(np.exp(state_old)*Sigma_u/phi))*norm.pdf(eps)
                      
                # Acceptance Ratio
                ratio = np.fmin(1, np.exp(np.log(p_proposal) - np.log(p_old)))
            
                rand = np.random.uniform(size=M)
                
                eps_update = np.where(ratio < rand, eps, e)
                state_update = np.where(ratio < rand, state_old, state_proposal)
                
                rejection_ratio = np.count_nonzero(ratio < rand)/M
                
                return(eps_update, state_update, rejection_ratio)
            
            
            # Empty list for Rejection Rates
            rej_ratio_all = []
            eps_up = epsilons_resample
            states_up = states_resample
            
            for j in range(2):
                
                eps_up, states_up, rej_ratio = mutation_step(eps_up, states_up, 
                                                             states_lagged_resample, 
                                                             x_t[i], y_t[i], 
                                                             phi_new, params_m[2], c)
                rej_ratio_all.append(rej_ratio)  
    
            
            #print("Avrg. Acceptance Ratio: " + str(1 - np.mean(rej_ratio_all)))
    
            # Adjust c based on rejection Rate
                
            # Function for adaptive step-size based on rejection rate
            def f(x):
            
                logistic_comp = np.exp(20*(x-0.4))/(1+np.exp(20*(x-0.4)))
            
                return(0.95 + 0.1*logistic_comp)
                
            c = c*f(1-np.mean(rej_ratio_all))
                
            #print("Proposal Variance: " + str(c), "\n")
                
            # Save new states, new epsilons and set new phi to old phi
            epsilons_resample = eps_up
            states_resample = states_up
          
            phi_old = phi_new
                
        posterior_tpf.append(states_resample)
        loglik_tpf += loglik_increment
        
        
    return(loglik_tpf) 
        
print(z(params_m, params_t, x_t, y_t, M, c_init, r_star))



theta_init = np.array([2, -0.9, 0.6, 0.6, 0.3, 0.2, 0.1])
params_m = [theta_init[0], theta_init[1], np.exp(theta_init[5])]
params_t = [theta_init[2], theta_init[3], np.tanh(theta_init[4]), np.exp(theta_init[6])]
loglik_init = z(params_m, params_t, x_t, y_t, M, c_init, r_star)


theta = [theta_init]
loglik = [loglik_init]

for i in range(2000):
    
    print(i)
    
    theta_draw = multivariate_normal.rvs(mean=theta[i], cov=0.004*np.eye(7))
    
    delta_2 = np.tanh(theta_draw[4])
    sigma_R = np.exp(theta_draw[5])
    sigma_Q = np.exp(theta_draw[6])
    
    #print(delta_2, sigma_R, sigma_Q)
    
    #print(theta_draw)
    params_m = [theta_draw[0], theta_draw[1], sigma_R]
    params_t = [theta_draw[2], theta_draw[3], delta_2, sigma_Q]
    
    z_t = z(params_m, params_t, x_t, y_t, M, c_init, r_star)
    
    # Prior for gamma_0, gamma_1, delta_0 and delta_1    
    prior = multivariate_normal.logpdf(theta_draw[:4], 
                                        mean=np.array([2.74, -1.19, 0.5, 0.8]), 
                                        cov=2*np.eye(4))
    
    prior -= multivariate_normal.logpdf(theta[i][:4], 
                                        mean=np.array([2.74, -1.19, 0.5, 0.8]), 
                                        cov=2*np.eye(4))  
    # Prior for delta_1
    prior += uniform.logpdf(delta_2, -1, 2)
    prior -= uniform.logpdf(np.tanh(theta[i][4]), -1, 2)
    
    # Prior Sigma_R
    prior += gamma.logpdf(sigma_R, a=5, scale=1/5) 
    prior -= gamma.logpdf(np.exp(theta[i][5]), a=5, scale=1/5)
    
    # Prior Sigma_Q
    prior += gamma.logpdf(sigma_Q, a=5, scale=1/5)
    prior -= gamma.logpdf(np.exp(theta[i][6]), a=5, scale=1/5)
    
    # Correct for Jacobian 
    logJacob = np.log(np.abs(1-delta_2**2)) - np.log(np.abs(1-np.tanh(theta[i][4])**2))
    logJacob +=  np.log(np.abs(sigma_R)) - np.log(np.abs(np.exp(theta[i][5])))
    logJacob += np.log(np.abs(sigma_Q)) - np.log(np.abs(np.exp(theta[i][6])))
    
    
    p_new = z_t 
    
    p_old = loglik[i] 
    
    #print(p_old, p_new)
    
    alpha = np.min(np.asarray([1, np.exp(p_new-p_old + prior + logJacob)]))
    
    #print(alpha)
    
    omega_m = np.random.uniform(0, 1, 1)   
    
    if omega_m < alpha:
        theta.append(theta_draw)
        loglik.append(z_t)
        #print("Move")
    else:
        theta.append(theta[i])
        loglik.append(loglik[i])
        #print("Stay")


theta = np.transpose(theta)
burn_in = 650   

plt.plot(theta[0][burn_in:])
plt.title("$\gamma_0$")
plt.grid()
plt.show()

plt.plot(theta[1][burn_in:])
plt.title("$\gamma_1$")
plt.grid()
plt.show()

plt.plot(theta[2][burn_in:])
plt.title("$\delta_0$")
plt.grid()
plt.show()

plt.plot(theta[3][burn_in:])
plt.title("$\delta_1$")
plt.grid()
plt.show()

plt.plot(np.tanh(theta[4])[burn_in:])
plt.title("$\delta_2$")
plt.grid()
plt.show()

plt.plot(np.exp(theta[5])[burn_in:])
plt.title("R")
plt.grid()
plt.show()


plt.plot(np.exp(theta[6])[burn_in:])
plt.title("Q")
plt.grid()
plt.show()



print("gamma_0: ", np.mean(theta[0][burn_in:]), 
      "\ngamma_1: ", np.mean(theta[1][burn_in:]), 
      "\ndelta_0: ", np.mean(theta[2][burn_in:]),
      "\ndelta_1: ", np.mean(theta[3][burn_in:]),
      "\ndelta_2: ", np.mean(np.tanh(theta[4])[burn_in:]),
      "\nR: ", np.mean(np.exp(theta[5])[burn_in:]),
      "\nQ: ", np.mean(np.exp(theta[6])[burn_in:]))
   


Sigma_prop = np.cov(theta[:,burn_in:])
theta_init_2 = np.array([np.mean(theta[0][burn_in:]),
                       np.mean(theta[1][burn_in:]),
                       np.mean(theta[2][burn_in:]),
                       np.mean(theta[3][burn_in:]),
                       np.mean(theta[4][burn_in:]),
                       np.mean(theta[5][burn_in:]),
                       np.mean(theta[6][burn_in:])])


params_m = [theta_init_2[0], theta_init_2[1], np.exp(theta_init_2[5])]
params_t = [theta_init_2[2], theta_init_2[3], np.tanh(theta_init_2[4]), np.exp(theta_init_2[6])]
loglik_init = z(params_m, params_t, x_t, y_t, M, c_init, r_star)


theta = [theta_init_2]
loglik = [loglik_init]


for i in range(2000):
    
    print(i)
    
    theta_draw = multivariate_normal.rvs(mean=theta[i], cov=Sigma_prop)
    
    delta_2 = np.tanh(theta_draw[4])
    sigma_R = np.exp(theta_draw[5])
    sigma_Q = np.exp(theta_draw[6])
    
    #print(delta_2, sigma_R, sigma_Q)
    
    #print(theta_draw)
    params_m = [theta_draw[0], theta_draw[1], sigma_R]
    params_t = [theta_draw[2], theta_draw[3], delta_2, sigma_Q]
    
    z_t = z(params_m, params_t, x_t, y_t, M, c_init, r_star)
    
    # Prior for gamma_0, gamma_1, delta_0 and delta_1    
    prior = multivariate_normal.logpdf(theta_draw[:4], 
                                        mean=np.array([2.74, -1.19, 0.5, 0.8]), 
                                        cov=2*np.eye(4))
    
    prior -= multivariate_normal.logpdf(theta[i][:4], 
                                        mean=np.array([2.74, -1.19, 0.5, 0.8]), 
                                        cov=2*np.eye(4))  
    # Prior for delta_1
    prior += uniform.logpdf(delta_2, -1, 2)
    prior -= uniform.logpdf(np.tanh(theta[i][4]), -1, 2)
    
    # Prior Sigma_R
    prior += gamma.logpdf(sigma_R, a=5, scale=1/5) 
    prior -= gamma.logpdf(np.exp(theta[i][5]), a=5, scale=1/5)
    
    # Prior Sigma_Q
    prior += gamma.logpdf(sigma_Q, a=5, scale=1/5)
    prior -= gamma.logpdf(np.exp(theta[i][6]), a=5, scale=1/5)
    
    # Correct for Jacobian 
    logJacob = np.log(np.abs(1-delta_2**2)) - np.log(np.abs(1-np.tanh(theta[i][4])**2))
    logJacob +=  np.log(np.abs(sigma_R)) - np.log(np.abs(np.exp(theta[i][5])))
    logJacob += np.log(np.abs(sigma_Q)) - np.log(np.abs(np.exp(theta[i][6])))
    
    
    p_new = z_t 
    
    p_old = loglik[i] 
    
    #print(p_old, p_new)
    
    alpha = np.min(np.asarray([1, np.exp(p_new-p_old + prior + logJacob)]))
    
    #print(alpha)
    
    omega_m = np.random.uniform(0, 1, 1)   
    
    if omega_m < alpha:
        theta.append(theta_draw)
        loglik.append(z_t)
        #print("Move")
    else:
        theta.append(theta[i])
        loglik.append(loglik[i])
        #print("Stay")


theta = np.transpose(theta)


















































for i in range(2000):
    
    print(i)
    
    theta_draw = multivariate_normal.rvs(mean=theta[i], cov=Sigma_prop)
    
    #print(theta_draw)
    params_m = [theta_draw[0], theta_draw[1], R]
    params_t = [theta_draw[2], theta_draw[3], 0.5, Q]
    
    z_t = z(params_m, params_t, x_t, y_t, M, c_init, r_star)
    
    
    p_new = z_t + np.log(multivariate_normal.pdf(theta_draw, 
                                        mean=np.zeros(4), 
                                        cov=np.eye(4)))
    
    p_old = loglik[i] + np.log(multivariate_normal.pdf(theta[i], 
                                              mean=np.zeros(4), 
                                              cov=np.eye(4)))
    
    #print(p_old, p_new)
    
    alpha = np.min(np.asarray([1, np.exp(p_new-p_old)]))
    
    #print(alpha)
    
    omega_m = np.random.uniform(0, 1, 1)   
    
    if omega_m < alpha:
        theta.append(theta_draw)
        loglik.append(z_t)
        #print("Move")
    else:
        theta.append(theta[i])
        loglik.append(loglik[i])
        #print("Stay")


theta = np.transpose(theta)
    
   
plt.plot(theta[0][300:])
plt.title("$\gamma_0$")
plt.grid()
plt.show()

plt.plot(theta[1][300:])
plt.title("$\gamma_1$")
plt.grid()
plt.show()

plt.plot(theta[2][300:])
plt.title("$\delta_0$")
plt.grid()
plt.show()

plt.plot(theta[3][300:])
plt.title("$\delta_1$")
plt.grid()
plt.show()


print("gamma_0: ", np.mean(theta[0][300:]), 
      "\ngamma_1: ", np.mean(theta[1][300:]), 
      "\ndelta_0: ", np.mean(theta[2][300:]),
      "\ndelta_1: ", np.mean(theta[3][300:]))
