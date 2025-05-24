# Code demonstrated using Python 3.13.2
# Bayesian Update: Posterior Distribution of Coin's Bias

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

# Data: 30 heads in 50 tosses
n_trials = 10
observed_heads = 3

# Create a grid for p (probability of heads)
p_grid = np.linspace(0, 1, 500)

# Prior: Beta distribution centered at 0.5 (Beta(10,10))
# This assumes that we have a prior belief that the coin is fair.
# Any changes in the posterior will be due to the observed data (fair or biased coin).
alpha = 10
beta = 10
prior = sts.beta.pdf(p_grid, alpha, beta)

# Likelihood: probability of 30 heads in 50 tosses, for each p in grid
likelihood = sts.binom.pmf(observed_heads, n_trials, p_grid)

# Unnormalized Posterior: product of likelihood and prior
unnorm_post = likelihood * prior

# Normalized Posterior: divide by sum to make it a proper density over grid
norm_post = unnorm_post / unnorm_post.sum()

# Plot all curves on one figure (scaled for visual comparison)
plt.figure(figsize=(8, 6))
plt.plot(p_grid, prior / prior.max(), label=f"Prior Beta({alpha},{beta})", linestyle='--')
plt.plot(p_grid, likelihood / likelihood.max(), label="Likelihood (scaled)", linestyle='-.')
###plt.plot(p_grid, unnorm_post / unnorm_post.max(), label="Unnormalized Posterior (scaled)", linestyle=':')
plt.plot(p_grid, norm_post / norm_post.max(), label="Normalized Posterior", linewidth=2)

plt.xlabel("p (Probability of heads)")
plt.ylabel("Scaled Density")
plt.title("Bayesian Update: Posterior Distribution of Coin's Bias")
plt.legend()
plt.grid(True)
plt.show()
