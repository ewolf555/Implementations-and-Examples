# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:29:11 2019

@author: eliaswolf
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.api import OLS
from statsmodels.regression.linear_model import OLSResults
from linearmodels.system import SUR
from sklearn.metrics import mean_squared_error
from math import sqrt
from pykalman import KalmanFilter

retail_data_full = pd.read_csv('...')


def Kalman_filter(state_vec_init, observations, F, A, H, P_init, Q, R, C_obs, C_state):
    
    N_states = state_vec_init.shape[0]
    
    def Kalman_alg(state_vec, obs_vec, F, A, H, P, Q, R, C_obs, C_state):
        # forecasting observation equation
        
        obs_vec_pred = C_obs + np.dot(A, obs_vec) + np.dot(H, state_vec)
        
        
        # MSE of observation Equation    
        MSE_obs_vec = np.dot(H, P).dot(H.T) + R
        MSE_obs_vec_inv = np.linalg.inv(MSE_obs_vec)
     
        # calculating Kalman Gain
        K = np.dot(P, H.T).dot(MSE_obs_vec_inv)
        
        # Predict state vector at time t
        state_vec_tt = C_state + state_vec + np.dot(K, (obs_vec - obs_vec_pred))
        
        # MSE of State Equation Update t
        P_tt = (np.eye(N_states) - K.dot(H)).dot(P)
        
        state_vec = C_state + F.dot(state_vec_tt)
        
        # MSE of State Equation Forecast
        P = np.dot(np.dot(F, P_tt), F.T) + Q

        return(state_vec, P, state_vec_tt, P_tt)
         
    state_vec = C_state + F.dot(state_vec_init)
    P = F.dot(P_init.dot(F.T)) + Q

    states = [[] for i in range(N_states)]
    state_var = []
              
    for i in range(observations.shape[1]):
        
        obs_vec = np.asarray([observations[j][i] for j in range(observations.shape[0])])

        
        state_vec, P, state_vec_tt, P_tt = Kalman_alg(state_vec, obs_vec, 
                                                      F, A, H, P, Q, R, C_obs, C_state)
        
        for j in range(N_states):
            states[j].append(state_vec_tt[j])
            state_var.append(P_tt)

    return(states, state_var)


T =  [[1, 0.5], 
      [0.2, 0.4]]
M = [[0.1, 0.3], 
     [0.5, 0.0]]

s_init = [0, 0]

c_init = [[1, 0], 
          [0, 1]]

V_m = [[1, 0], 
       [0, 1]]

V_s = [[1, 0], 
       [0, 1]]

measurements = np.asarray([retail_data_full.X1,
                           retail_data_full.X2])

col_1 = np.asarray(retail_data_full.X1)
col_2 = np.asarray(retail_data_full.X2)
    
measurements_0 = np.stack((col_1, col_2), axis=-1)
                          
    
kf = KalmanFilter(transition_matrices = T, 
                  observation_matrices = M,
                  initial_state_mean = s_init, 
                  initial_state_covariance= c_init,
                  transition_covariance = V_s, 
                  observation_covariance = V_m)

filtered_state_means, filtered_state_covariances = kf.filter(measurements_0)

T_1 = np.asarray(T)
M_1 = np.asarray(M)

s_init_1 = np.asarray(s_init)

c_init_1 = np.asarray(c_init)

V_m_1 = np.asarray(V_m)

V_s_1 = np.asarray(V_s)

A_0 = np.asarray([[0, 0], 
                  [0, 0]])

measurements_1 = measurements

cons = np.asarray([0, 0])

filtered_state_means_1, filtered_state_covariances_1 = Kalman_filter(state_vec_init = s_init_1, 
                                                                     observations = measurements_1, 
                                                                     F = T_1, 
                                                                     A = A_0, 
                                                                     H = M_1,
                                                                     P_init = c_init_1, 
                                                                     Q = V_s_1, 
                                                                     R = V_m_1, 
                                                                     C_obs = cons, 
                                                                     C_state = cons)


plt.plot(filtered_state_means_1[0], label = 'filtered state 0')
plt.plot(np.hsplit(filtered_state_means, 2)[0], label = 'Package, filtered states 0')
plt.title('State 0')
plt.legend()
plt.grid()
plt.show()

plt.plot(filtered_state_means_1[1], label = 'filtered state 1')
plt.plot(np.hsplit(filtered_state_means, 2)[1], label = 'Package, filtered states 1')
plt.title('State 1')
plt.legend()
plt.grid()
plt.show()



    
    
    

    
