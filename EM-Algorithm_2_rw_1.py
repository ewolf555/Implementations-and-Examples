# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 01:08:51 2020

@author: Elias Wolf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os as os
from scipy.optimize import minimize

os.chdir('C:/Users/Elias Wolf/Documents/Uni/Theory of Machine Learning/Implementation')

data_all = pd.read_csv('capm_data.csv')



def Kalman_filter(theta, observations, endog, state_vec_init, P_init):
    """
        Loglikelihood function
    ____
    Input:
    - theta: values for the unknown parameters sigma_u_1,sigma_u_2, sigma_eps
             in a list named "estim"
    - observations: exogenous variable
    - endogenous. endogeous variable
    - state_vec_init: initial mean values for the Kalman filter recursion
    - P_init: intial values for the variance covariance matrix 
              of the state variables
    
    ____
    Returns:
    the loglikelihood that the given values for the parameters 
    """
    
    mle_sigma_u_1, mle_sigma_u_2, mle_sigma_eps = theta
    
    mle_sigma_u_1 = np.exp(mle_sigma_u_1)
    mle_sigma_u_2 = np.exp(mle_sigma_u_2)
    mle_sigma_eps = np.exp(mle_sigma_eps)
    
    C_state = np.array([0, 0])
    F = np.eye(2)
    Q = np.diag([mle_sigma_u_1, mle_sigma_u_2]) 
    R = mle_sigma_eps
    
    N_states = state_vec_init.shape[0]
    
    def Kalman_alg(state_vec, obs_vec, obs_endog, F, P, Q, R, C_state):
        
        # Correction Step
        
        H = obs_endog
        
        #print(obs_endog)
        # forecasting observation equation
        # y_t|t-1 = H*xi_t|t-1
        obs_vec_pred = np.dot(H, state_vec)
        #print('Forecast:', obs_vec_pred)
        
        # forecast error
        # y_t - y_t|t-1
        e = obs_vec - obs_vec_pred
        #print('Forecast error:', e)
        
        # MSE of observation Equation    
        # E[(y_t - y_t|t-1)(y_t - y_t|t-1)'] = H*P_t|t-1*H' + R
        MSE_obs_vec = np.dot(H, P).dot(H.T) + R
        #print('MSE(y):', MSE_obs_vec)
        
        # Inverse of E[(y_t - y_t|t-1)(y_t - y_t|t-1)'] required for Kalman Gain
        MSE_obs_vec_inv = 1/MSE_obs_vec
        
        # calculating Kalman Gain 
        # P_t|t-1*H'(H*P_t|t-1*H' + R)^(-1)
        K = np.dot(P, H.T).dot(MSE_obs_vec_inv)
        #print('Kalman Gain:', K)
        
        # State vector at time t
        # xi_t|t = x_t|t-1 + P_t|t-1*H'(H*P_t|t-1*H' + R)^(-1)
        state_vec_tt = C_state + state_vec + np.dot(K, e)
        
        # MSE of State Equation Update t
        # P_t|t = (I - K*H)P_t|t-1 
        P_tt = (np.eye(N_states) - K.dot(H)).dot(P)
        
        # Prediction Step
        
        # Forecast of the state vector
        # xi_t+1|t = C + F*xi_t|t
        state_vec = C_state + F.dot(state_vec_tt)
        
        # Forecast MSE of State Equation 
        # P_t+1|t = F*P_t|t*F' + Q
        P = np.dot(np.dot(F, P_tt), F.T) + Q
        
        return(state_vec, P, e, MSE_obs_vec, state_vec_tt, obs_vec_pred)
         
    state_vec = state_vec_init
    P = P_init

    states = [[] for i in range(state_vec_init.shape[0])]
    forecast_error = []
    forecast_var = []
    forecast_y = []
              
    for i in range(observations.shape[0]):
        
        obs_vec = np.asarray(observations[i])    
        
        obs_endog = endog[i]
        
        state_vec, P, e, MSE_obs_vec, state_vec_tt, obs_vec_pred = Kalman_alg(state_vec, obs_vec, obs_endog, F, P, Q, R, C_state)
        
        for j in range(N_states):
            states[j].append(state_vec_tt[j])
        
        forecast_error.append(e)
        forecast_var.append(MSE_obs_vec)
        forecast_y.append(obs_vec_pred) 
        
    return(states, forecast_error, forecast_var, forecast_y)



def loglik(theta, observations, endog, state_vec_init, P_init): #the likelihood of the given parameters
    """
    Loglikelihood function
    ____
    Input:
    - theta: values for the unknown parameters c , phi, sigma_u_1,sigma_u_2, sigma_eps
    in a list named "estim"
    - observations: exogenous variable
    - endogenous. endogeous variable
    - state_vec_init: initial mean values for the Kalman filter recursion
    - P_init: intial values for the variance covariance matrix of the state variables
    
    ____
    Returns:
    the loglikelihood that the given values for the parameters 

    """
    
    def square(argument):
        return([x ** 2 for x in argument])
    
    
    states, forecast_error, forecast_var, forecast_y = Kalman_filter(theta, observations, endog, state_vec_init, P_init)

    term_1 = - len(observations) * 0.5 * np.log(2*np.pi)
    term_2 = - 0.5 * sum(np.log(forecast_var))
    term_3 = - 0.5 * sum([square(forecast_error)[i]/forecast_var[i] for i in range(len(observations))])
    
    loglik = sum([term_1,term_2,term_3])
    #calculation of likelihood
    
    #return a negative value because the optimization algorithm minimizes the funtion
    return(-loglik) 


def MaxLikEstim(theta_init, observations, endog):
    """
    Maximum Likihood Maximization
    ____
    Input:
    - theta_init: initial values for the hyper paramters of the model 
    - observations: exogenous variable
    - endogenous. endogeous variable
    ____
    Returns:
    maximum likihood estimates for the hyper paramters of the state space model
    optimizing the likelihood function via the EM-Algorithm
    """
    
    P_init = np.diag([5,5]) 
    state_vec_init = np.array([1,1]) 
    
    print("Optimizing likelihood for hyper parameters \n")
    estim = minimize(loglik, init_vals, 
                     args = (observations, endog, state_vec_init, P_init),
                     method = 'BFGS', tol=1e-6, options = {'disp': True})    
    
    return(estim.x)

# Estimate time-varying CAPM-Model
PN = 7
r_asset = data_all.iloc[:, PN]
r_market = data_all.iloc[:, -1]

timerange = range(len(r_asset))

observations = np.asarray(r_asset)
endog = [np.array([1, r_market[t]]) for t in timerange]

P_init = np.diag([1,1]) 
state_vec_init = np.array([1,1]) 


init_vals = [5 for i in range(3)]

optim_theta = MaxLikEstim(init_vals, observations, endog)


states, forecast_error, forecast_var, forecast_y = Kalman_filter(optim_theta, observations, endog, state_vec_init, P_init)


# Plot 
plt.plot(states[0])
plt.grid()
plt.title(r'Evolution of $\alpha_t = \alpha_{t-1} + u_{t,1}$')
plt.xticks([0, 239, 479, 729, 969], ['1934', '1954', '1974', '1994', '2014'])
plt.xlabel('Years')
plt.show()

# Plot 
plt.plot(states[1])
plt.grid()
plt.title(r'Evolution of $\beta_t = \beta_{t-1} + u_{t,2}$')
plt.xticks([0, 239, 479, 729, 969], ['1934', '1954', '1974', '1994', '2014'])
plt.xlabel('Years')
plt.show()

print('Portfolio Nr.' + str(PN) + '\n', optim_theta)


plt.plot(forecast_y, label='forecast', color='black')
plt.plot(observations, label='observed returns', color='red')
plt.grid()
plt.title('Portfolio Return')
plt.xticks([0, 239, 479, 729, 969], ['1934', '1954', '1974', '1994', '2014'])
plt.xlabel('Years')
plt.legend()
plt.show()