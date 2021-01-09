# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:36:24 2020

@author: Elias Wolf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

prob = [0.8, 0.2]
loc = [3, -2]
scale = [1, 1]


def mix_pdf(x, loc, scale, weights):
    d = np.zeros_like(x)
    for mu, sigma, pi in zip(loc, scale, weights):
        d += pi * norm.pdf(x, loc=mu, scale=sigma)
    return d

x_vals = np.linspace(-6, 8, 500)

n_draws = 50000

proposal_draws = np.random.normal(0, 20, n_draws)
proposal_dens = norm.pdf(proposal_draws, 0, 5)

weights = mix_pdf(proposal_draws, loc, scale, prob)/proposal_dens

weights_norm = weights/np.sum(weights)

target = np.random.choice(proposal_draws, size=n_draws, 
                          p=weights_norm)


plt.hist(target, density=True, bins=30, color="grey")
plt.plot(x_vals, norm.pdf(x_vals, 0, 5), color="k", linestyle="--", 
         label="Proposal: $q(x) \sim N(0,5)$")
plt.plot(x_vals, mix_pdf(x_vals, loc, scale, weights=prob), color="r",
         label="Target: $\pi(x)$")
plt.title("Example: Importance Sampling")
plt.legend()
plt.grid()
plt.show()