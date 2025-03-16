import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Sample Problem based on cloud observation and rain prediction
# https://www2.stat.duke.edu/~jerry/sta101/extraproblems/bayesian.htm

# Problem:
# It's a typically hot morning in June in Durham.  You look outside and see some dark clouds rolling in.  Is it going to rain?
# Historically, there is a 30% chance of rain on any given day in June.  
# Furthermore, on days when it does in fact rain, 95% of the time there are dark clouds that roll in during the morning.   
# But, on days when it does not rain, 25% of the time there are dark clouds that roll in during the morning.
# Given that there are dark clouds rolling in, what is the chance that it will rain?


# Prior Setup
# Beta prior with parameters rain_prior = 3 and no_rain_prior = 7,
# which gives a prior mean of 0.3. This reflects the belief that it rains on 30% of days.
rain_prior = 3
no_rain_prior = 7

# Calculate prior mean and standard deviation (theoretical values)
prior_mean = rain_prior / (rain_prior + no_rain_prior)
prior_std = np.sqrt(rain_prior * no_rain_prior / ((rain_prior + no_rain_prior)**2 * (rain_prior + no_rain_prior + 1)))
# (These values are computed via integration ; discrete samples are shown below.)

# Generates raw discrete samples from the Beta prior distribution
num_samples = 10000
prior_samples = np.random.beta(rain_prior, no_rain_prior, size=num_samples)

# Likelihood Function
# The likelihood of observing dark clouds (B) given a probability of rain (θ or P(R)) is defined as:
# L(B | R) = 0.95 * P(R) + 0.25 * P(~R) or L(B | θ) = 0.95θ + 0.25(1-θ)
# where P(~R) = [1 - P(R)] = (1 - θ).
# P(R) is the probability of rain and P(~R) is the probability of no rain.
# This means that on rainy days (with probability θ) dark clouds are observed 95% of the time,
# and on non-rainy days (with probability 1-θ) dark clouds are observed 25% of the time.
def likelihood(theta):
    return 0.95 * theta + 0.25 * (1 - theta)

# Discretize the parameter space
theta = np.linspace(0, 1, 1000)

# Computes the prior density using the Beta distribution
prior_density = beta.pdf(theta, rain_prior, no_rain_prior)

# Compute the likelihood for each value of theta
like = likelihood(theta)

# Compute the Posterior
# Unnormalized posterior density is the product of the prior density and the likelihood.
unnormalized_posterior = prior_density * like

# Normalize the posterior density using numerical integration
posterior_density = unnormalized_posterior / np.trapezoid(unnormalized_posterior, theta)

# Generate raw discrete samples from the posterior distribution
dtheta = theta[1] - theta[0]
posterior_prob = posterior_density * dtheta
posterior_prob = posterior_prob / np.sum(posterior_prob)  # ensure normalization
posterior_samples = np.random.choice(theta, size=num_samples, p=posterior_prob)

# To obtain a distribution for the summary statistics (mean and std) we perform bootstrapping.
# For each bootstrap iteration we sample a subset from the raw prior/posterior samples and
# compute the sample mean and standard deviation.
B = 1000              # number of bootstrap iterations
boot_sample_size = 100  # size of each bootstrap sample

prior_means = []
prior_stds = []
posterior_means = []
posterior_stds = []

for i in range(B):
    boot_prior = np.random.choice(prior_samples, size=boot_sample_size, replace=True)
    boot_post = np.random.choice(posterior_samples, size=boot_sample_size, replace=True)
    prior_means.append(np.mean(boot_prior))
    prior_stds.append(np.std(boot_prior))
    posterior_means.append(np.mean(boot_post))
    posterior_stds.append(np.std(boot_post))

prior_means = np.array(prior_means)
prior_stds = np.array(prior_stds)
posterior_means = np.array(posterior_means)
posterior_stds = np.array(posterior_stds)

# Plot the discrete distributions for the summary statistics in a 2x2 grid.
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Distribution of Prior Mean
axs[0, 0].hist(prior_means, bins=30, density=True, color='skyblue', edgecolor='black')
axs[0, 0].axvline(np.mean(prior_means), color='red', linestyle='dashed', linewidth=2,
                  label=f'Mean: {np.mean(prior_means):.3f}')
axs[0, 0].axvline(np.mean(prior_means) + np.std(prior_means), color='green', linestyle='dashed', linewidth=2,
                  label=f'Std: {np.std(prior_means):.3f}')
axs[0, 0].axvline(np.mean(prior_means) - np.std(prior_means), color='green', linestyle='dashed', linewidth=2,
                  label=f'Std: {np.std(prior_means):.3f}')
axs[0, 0].set_title('Distribution of Prior Mean')
axs[0, 0].set_xlabel('Prior Mean')
axs[0, 0].set_ylabel('Density')
axs[0, 0].legend()

# Distribution of Prior Std
axs[0, 1].hist(prior_stds, bins=30, density=True, color='lightgreen', edgecolor='black')
axs[0, 1].axvline(np.mean(prior_stds), color='red', linestyle='dashed', linewidth=2,
                  label=f'Mean: {np.mean(prior_stds):.3f}')
axs[0, 1].axvline(np.mean(prior_stds) + np.std(prior_stds), color='green', linestyle='dashed', linewidth=2,
                  label=f'Std: {np.std(prior_stds):.3f}')
axs[0, 1].axvline(np.mean(prior_stds) - np.std(prior_stds), color='green', linestyle='dashed', linewidth=2,
                  label=f'Std: {np.std(prior_stds):.3f}')
axs[0, 1].set_title('Distribution of Prior Std')
axs[0, 1].set_xlabel('Prior Std')
axs[0, 1].set_ylabel('Density')
axs[0, 1].legend()

# Distribution of Posterior Mean
axs[1, 0].hist(posterior_means, bins=30, density=True, color='salmon', edgecolor='black')
axs[1, 0].axvline(np.mean(posterior_means), color='red', linestyle='dashed', linewidth=2,
                  label=f'Mean: {np.mean(posterior_means):.3f}')
axs[1, 0].axvline(np.mean(posterior_means) + np.std(posterior_means), color='green', linestyle='dashed', linewidth=2,
                  label=f'Std: {np.std(posterior_means):.3f}')
axs[1, 0].axvline(np.mean(posterior_means) - np.std(posterior_means), color='green', linestyle='dashed', linewidth=2,
                  label=f'Std: {np.std(posterior_means):.3f}')
axs[1, 0].set_title('Distribution of Posterior Mean')
axs[1, 0].set_xlabel('Posterior Mean')
axs[1, 0].set_ylabel('Density')
axs[1, 0].legend()

# Distribution of Posterior Std
axs[1, 1].hist(posterior_stds, bins=30, density=True, color='plum', edgecolor='black')
axs[1, 1].axvline(np.mean(posterior_stds), color='red', linestyle='dashed', linewidth=2,
                  label=f'Mean: {np.mean(posterior_stds):.3f}')
axs[1, 1].axvline(np.mean(posterior_stds) + np.std(posterior_stds), color='green', linestyle='dashed', linewidth=2,
                  label=f'Std: {np.std(posterior_stds):.3f}')
axs[1, 1].axvline(np.mean(posterior_stds) - np.std(posterior_stds), color='green', linestyle='dashed', linewidth=2,
                  label=f'Std: {np.std(posterior_stds):.3f}')
axs[1, 1].set_title('Distribution of Posterior Std')
axs[1, 1].set_xlabel('Posterior Std')
axs[1, 1].set_ylabel('Density')
axs[1, 1].legend()

plt.tight_layout()
plt.show()
