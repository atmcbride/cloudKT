import numpy as np
import emcee
import logging
from functools import partial


from utilities import load_module

logger = logging.getLogger(__name__)

# def functions_must_exist(func):
#     def wrapper(self, *args, **kwargs):
#         """
#         Check that the log likelihood and log prior functions have been assigned
#         """
#         if not hasattr(self, 'log_likelihood'):
#             raise Exception("Log likelihood function has not been assigned!")
#         if not hasattr(self, 'log_priors'):
#             raise Exception("Log prior functions have not been assigned!")
#         return func(self, *args, **kwargs)
#     return wrapper


# class MCMC_Framework:
#     def intake_log_prior(self, log_prior_function, **log_prior_kwargs):
#         """
#         Intake a specified log prior function and its kwargs. 
#         Note that you can add many log prior functions! As long as they all take the form log_prior(theta, **log_prior_kwargs)
#         and the kwargs are specified to this function, they will be passed to the appropriate log_prior when run in the log-probability fn
#         """
#         if not hasattr(self, 'log_priors'):
#             self.log_priors = []
#         self.log_priors.append((log_prior_function, log_prior_kwargs))
#         pass

#     def intake_log_likelihood(self, log_likelihood_function, **log_likelihood_kwargs):
#         """
#         Intake a specified log likelihood function and its kwargs
#         By design, there can only be one log likelihood function!
#         """
#         self.log_likelihood = log_likelihood_function
#         self.log_likelihood_kwargs = log_likelihood_kwargs
    
#     def log_prior(self, theta):
#         log_prior_value = 0
#         for item in self.log_priors:
#             log_prior_fn, log_prior_kwargs = item
#             log_prior_value += log_prior_fn(theta, **log_prior_kwargs)
#         return log_prior_value

#     @functions_must_exist
#     def log_probability(self, theta):
#         log_likelihood = self.log_likelihood(theta, **self.log_likelihood_kwargs)
#         log_prior = self.log_prior(theta)
#         return log_likelihood + log_prior
    



def log_probability(theta, sightline = None, log_likelihood = None, log_priors = None, **kwargs):
    ll = log_likelihood(theta, sightline = sightline, **kwargs)
    lp = 0
    for item in log_priors:
        log_prior_fn, log_prior_kwargs = item
        lp += log_prior_fn(theta, sightline = sightline, **log_prior_kwargs)
    return ll + lp

def run_mcmc(sightline, mcmc_config, log_likelihood, log_priors, steps = 1000, nwalkers = 100, pool = None, filename = None):
    """"
    Run the MCMC
    """
    ndim = len(sightline.voxel_dAVdd) 
    nstar = len(sightline.stars)
    ndim_amp = int(ndim + ndim * nstar)

    if nwalkers < 2 * ndim:
        nwalkers = 2 * ndim + 5
        print('WARNING: nwalkers updated to', nwalkers)

    if filename is not None:
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim_amp)
    else:
        backend = None
        logger.warning('NO BACKEND')

    priors = []
    for item in mcmc_config:
        pass


    sampler = emcee.EnsembleSampler(nwalkers, ndim_amp, log_probability, pool = pool, backend = backend, kwargs = {
                                    'sightline': sightline, 'log_likelihood': log_likelihood, 'log_priors': log_priors})

    init = 10 *  (np.random.random((nwalkers, ndim_amp)) - 0.5)
    init[:, ndim:] = np.abs(sightline.dAVdd.ravel()[np.newaxis, :] + 0.1*(np.random.random(init[:, ndim:].shape)-0.5))
    init[:, ndim:][(init[:, ndim:] <= 0.1)] = 0.11 + 0.05 * np.random.random(np.sum(init[:, ndim:]<= 0.1))

    print('NDIM:', ndim, 'NSTAR:', nstar, 'INITSHAPE:', init.shape)
    # init = 10 *  (np.random.random((nwalkers, ndim_amp)) - 0.5)

    sampler.run_mcmc(init, steps, progress = True, store = True);

    # INITIAL TESTS WITH UN-CONFIGURED PRIORS

    # WITH POOL(8), 6 VARIABLES, 100 WALKERS, THIS TAKES 05:37
    # WITHOUT POOL(8), 6 VARIABLES, 100 WALKERS, THIS TAKES 01:08

    # WITH POOL(8), 12 VARIABLES, 100 WALKERS, THIS TAKES 05:28
    # WITHOUT POOL(8), 12 VARIABLES, 100 WALKERS, THIS TAKES 01:22

    # I FORGOT TO TURN ON 
    # import os
    # os.environ['OMP_NUM_THREADS'] = "1"
    # ADDED TO THIS THIS FILE

    # WITH POOL(8), 12 VARIABLES, 100 WALKERS, THIS TAKES 04:54

    # ADDED TO cloudKT.py

    # WITH POOL(8), 12 VARIABLES, 100 WALKERS, THIS TAKES 04:52

    # WITH POOL(20), 12 VARIABLES, 100 WALKERS, THIS TAKES 04:06



    return sampler