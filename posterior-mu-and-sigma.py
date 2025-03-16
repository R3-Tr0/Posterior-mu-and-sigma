import numpy as np
import matplotlib.pyplot as plt
import statistics as s
from scipy.stats import invgamma

 
# Generate some synthetic data

# Assume true average Filipino age is 29.36 and standard deviation is 0.78 years
# https://www.researchgate.net/figure/Profile-characteristics-in-healthy-Filipino-population_tbl1_351048220

np.random.seed(42)
true_mu = 29.36
true_sigma = 0.78
data = np.random.normal(true_mu, true_sigma, size=100)


# Define the prior hyperparameters
prior_mu_mean = 29.36 # Mean of the prior distribution of the mean
prior_mu_precision = 1 / (true_sigma*true_sigma)
# Variance = 1 / precision
# Assuming std is 0.78, then variance is 0.78^2 = 0.6084
# Assuming variance is 0.6084, then precision is 1 / 0.6084 = 1.6438


# Define the prior hyperparameters for sigma^2 (inverse gamma prior)
# Estimate  the prior mean for true variance to be ~0.78^2 = 0.6084.
# For InvGamma(alpha, beta), mean = beta/(alpha-1) (if alpha > 1).
# Set alpha = 7.8, then beta should be ≈ 0.6084*(7.8-1) ≈ 4.141.

prior_sigma_alpha = 7.8 # Alpha = shape parameter of inverse gamma distribution for standard deviation (sigma) prior distribution
prior_sigma_beta = 4.141 # Beta = rate parameter of inverse gamma distribution for standard deviation (sigma) prior distribution

# Update the prior hyperparameters with the data
posterior_mu_precision = prior_mu_precision + len(data) / true_sigma**2
posterior_mu_mean = (prior_mu_precision * prior_mu_mean + np.sum(data)) / posterior_mu_precision
 
posterior_sigma_alpha = prior_sigma_alpha + len(data) / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data))**2) / 2
 
# Calculate the posterior parameters
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)

# Sample from the inverse gamma posterior for sigma^2 and then compute sigma
posterior_sigma = invgamma.rvs(a=posterior_sigma_alpha, scale=posterior_sigma_beta, size=10000)
posterior_sigma = np.sqrt(posterior_sigma)
 
# Plot the posterior distributions
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='skyblue', edgecolor='black')
plt.title('Posterior distribution of $\mu$ of Filipino ages')
plt.xlabel('$\mu$')
plt.ylabel('Density')
 
plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='lightgreen', edgecolor='black')
plt.title('Posterior distribution of $\sigma$ of Filipino ages')
plt.xlabel('$\sigma$')
plt.ylabel('Density')
 
plt.tight_layout()
plt.show()
 
# Calculate summary statistics
mean_mu = np.mean(posterior_mu)
std_mu = np.std(posterior_mu)
print("Mean of mu:", mean_mu)
print("Standard deviation of mu:", std_mu)
 
mean_sigma = np.mean(posterior_sigma)
std_sigma = np.std(posterior_sigma)
print("Mean of sigma:", mean_sigma)
print("Standard deviation of sigma:", std_sigma)