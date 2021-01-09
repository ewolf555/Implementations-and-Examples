# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:07:58 2020

@author: Elias Wolf
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Aproximate pi using Monte Carlo Simulations: Randomly draw from two uniform
Distributions gove coordinates of a square with sidelentgh 1. The probability
to draw a coordiante that lies within the circle with radius 0.5 is given by 
                  
                    Pr(in circle) = 0.5^2*pi/1 = pi/4
                
Solvinf for pi gives 

                          pi = Pr(in circle)/4
                          
Pr(in circle) is simulated by Monte Carlo Simulations as number of drawn 
coordinates of a sample of increasing size that lie within the circle.                 
"""


N_sim = 100000

pi_approx = []

for i in range(100, N_sim, 100):
    
    # Draw coordinates
    x_coord = np.random.uniform(-0.5, 0.5, size=i)
    y_coord = np.random.uniform(-0.5, 0.5, size=i)
    
    # Calculate euclidean distance to origin
    norm_vec = np.sqrt(x_coord**2 + y_coord**2)
    
    # Check if point lies within cirlce of radius 0.5
    in_circle = np.sum(norm_vec <= 0.5)

    pi_approx.append(in_circle/i*4)
    

# Plot results
plt.plot(range(100, N_sim, 100), pi_approx)
plt.grid()
plt.show()

# Print approximation from largest sample size
print(pi_approx[-1])


