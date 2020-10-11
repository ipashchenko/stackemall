import numpy as np
from scipy.integrate import quad, dblquad, nquad
from scipy.stats import beta as beta_dist
from scipy.stats import rice as rice_dist
import dlib


def alpha(m_0, mod_m):
    return ((1 - m_0) / (m_0 * mod_m ** 2) - 1) * m_0


def beta(m_0, mod_m):
    return ((1 - m_0) / (m_0 * mod_m ** 2) - 1) * (1 - m_0)


def P(m_i, m_obs, m_0, mod_m, sigma_m):
    alpha_loc = alpha(m_0, mod_m)
    beta_loc = beta(m_0, mod_m)
    return beta_dist(alpha_loc, beta_loc).pdf(m_i) * rice_dist(m_i / sigma_m, scale=sigma_m).pdf(m_obs)


def loglik_j(m_obs_j, m_0, m_mod, sigma_j):
    result = quad(P, a=0, b=1, args=(m_obs_j, m_0, m_mod, sigma_j), epsabs=1e-15, epsrel=1e-15, limit=1000, limlst=100)[0]
    print("Lik = ", result)
    return np.log(result)


def loglik(m_0, mod_m, ms_obs, sigmas):
    print("calculating for ", m_0, mod_m)
    return np.sum([loglik_j(m_obs_j, m_0, mod_m, sigma_j) for m_obs_j, sigma_j in zip(ms_obs, sigmas)])


def lik(m_0, mod_m, ms_obs, sigmas):
    return np.exp(loglik(m_0, mod_m, ms_obs, sigmas))


# Find measurements
n = 10
# True mean intrinsic FPOL
m_0_true = 0.2
# True modulation index
mod_m_true = 0.4
# True variable FPOL values
ms_i_true = beta_dist(alpha(m_0_true, mod_m_true), beta(m_0_true, mod_m_true)).rvs(n)
# Some errors
sigmas = 0.03*np.ones(n)
# Observed variable FPOL values
ms_obs = rice_dist(ms_i_true / sigmas, scale=sigmas).rvs(n)


def l(m_0, mod_m_0):
    return np.exp(loglik(m_0, mod_m_0, ms_obs, sigmas))


# Function must accept only optimized arguments. Others getting from outer scope.
def ll(m_0, mod_m_0):
    return loglik(m_0, mod_m_0, ms_obs, sigmas)



# print(ll(0.2, 0.4))

# lower_bounds = [0.001, 0.001]
# upper_bounds = [0.8, 0.8]
# n_fun_eval = 50
# (m_0_ml, mod_m_0_ml), _ = dlib.find_max_global(ll, lower_bounds, upper_bounds, n_fun_eval)


# def L(mod_0):
#     return quad(lik, a=0, b=1, args=(mod_0, ms_obs, sigmas), epsabs=1e-15, epsrel=1e-15, limit=1000, limlst=100)[0]

# A = quad(L, a=0, b=+np.inf, epsabs=1e-15, epsrel=1e-15, limit=1000, limlst=100)[0]

# A = dblquad(lambda m_0, mod_m_0: np.exp(loglik(m_0, mod_m_0, ms_obs, sigmas)), 0, +np.inf, lambda x: 0, lambda x: +1)

# A = nquad(lik, [[0, 1], [0, +np.inf]], args=(ms_obs, sigmas))