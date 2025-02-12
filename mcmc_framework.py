import numpy as np
import emcee
import logging

logger = logging.getLogger(__name__)

def functions_must_exist(func):
    def wrapper(self, *args, **kwargs):
        """
        Check that the log likelihood and log prior functions have been assigned
        """
        if not hasattr(self, 'log_likelihood'):
            raise Exception("Log likelihood function has not been assigned!")
        if not hasattr(self, 'log_priors'):
            raise Exception("Log prior functions have not been assigned!")
        return func(self, *args, **kwargs)
    return wrapper


class MCMC_Framework:
    def intake_log_prior(self, log_prior_function, **log_prior_kwargs):
        """
        Intake a specified log prior function and its kwargs. 
        Note that you can add many log prior functions! As long as they all take the form log_prior(theta, **log_prior_kwargs)
        and the kwargs are specified to this function, they will be passed to the appropriate log_prior when run in the log-probability fn
        """
        if not hasattr(self, 'log_priors'):
            self.log_priors = []
        self.log_priors.append((log_prior_function, log_prior_kwargs))
        pass

    def intake_log_likelihood(self, log_likelihood_function, **log_likelihood_kwargs):
        """
        Intake a specified log likelihood function and its kwargs
        By design, there can only be one log likelihood function!
        """
        self.log_likelihood = log_likelihood_function
        self.log_likelihood_kwargs = log_likelihood_kwargs
    
    def log_prior(self, theta):
        log_prior_value = 0
        for item in self.log_priors:
            log_prior_fn, log_prior_kwargs = item
            log_prior_value += log_prior_fn(theta, **log_prior_kwargs)
        return log_prior_value

    @functions_must_exist
    def log_probability(self, theta):
        log_likelihood = self.log_likelihood(theta, **self.log_likelihood_kwargs)
        log_prior = self.log_prior(theta)
        return log_likelihood + log_prior

    def run_MCMC(self, sightline, steps = 1000, nwalkers = 100, pool = None, filename = None):
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
            backend.reset(nwalkers, ndim)
        else:
            backend = None
            logger.warning('NO BACKEND')

        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, pool = pool, backend = backend)
        init[:, ndim:] = np.abs(sl.dAVdd.ravel()[np.newaxis, :] + 0.1*(np.random.random(init[:, ndim:].shape)-0.5))
        init[:, ndim:][(init[:, ndim:] <= 0.1)] = 0.11 + 0.05 * np.random.random(np.sum(init[:, ndim:]<= 0.1))
        init[:, ndim:] 
        print('NDIM:', ndim, 'NSTAR:', nstar, 'INITSHAPE:', init.shape)
        init = 10 *  (np.random.random((nwalkers, ndim_amp)) - 0.5)
        self.sampler.run_mcmc(init,  steps, progress = True, store = True);

"""

def MCMC_fg(sl, steps = 1000, nwalkers = 100, pool = None, filename = None):
    ndim = len(sl.voxel_dAVdd) 
    nstar = len(sl.stars)
    ndim_amp = int(ndim + ndim * nstar)

    if nwalkers < 2 * ndim_amp:
        nwalkers = 2 * ndim_amp + 5
        print('WARNING: nwalkers updated to', nwalkers)

    lp_foreground = Logprior_Foreground(sl.l, sl.b)

    
    if filename is not None:
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim_amp)
    else:
        backend = None
    # dAVdd_prior = sl.dAVdd[:]
    # dAVdd_prior[dAVdd_prior == 0] = np.nan 
    # dAVdd_prior_med = np.nanmedian(dAVdd_prior, axis = 1)
    # dAVdd_prior_std = np.nanstd(dAVdd_prior, axis = 1, ddof = 1)
    # gaussparams = (dAVdd_prior_med, dAVdd_prior_std)
    # print(gaussparams)

    # with Pool(15) as pool:

    sampler = emcee.EnsembleSampler(nwalkers, ndim_amp , logprob_fg, 
                                    kwargs={'sl': sl, 'lp_fore': lp_foreground}, pool = pool, backend = backend) # OKAY SO I FORGOT TO CHANGE THIS, WAS LOGPROB_2
    # init = 12.5 *(np.random.random((nwalkers, ndim_amp)) - 0.5)
    init = 10 *  (np.random.random((nwalkers, ndim_amp)) - 0.5)

    init[:, ndim:] = np.abs(sl.dAVdd.ravel()[np.newaxis, :] + 0.1*(np.random.random(init[:, ndim:].shape)-0.5))
    init[:, ndim:][(init[:, ndim:] <= 0.1)] = 0.11 + 0.05 * np.random.random(np.sum(init[:, ndim:]<= 0.1))
    init[:, ndim:] 
    print('NDIM:', ndim, 'NSTAR:', nstar, 'INITSHAPE:', init.shape)
    
    sampler.run_mcmc(init,  steps, progress = True, store = True);
    
    return sampler, ndim, ndim_amp
"""

def mcmc_framework(log_likelihood, log_likelihood_kwargs, log_priors, log_priors_kwargs):
    def log_probability(theta):
        log_likelihood_value = log_likelihood(theta, **log_likelihood_kwargs)
        log_prior_value = 0
        for item in log_priors:
            log_prior_fn, log_prior_kwargs = item
            log_prior_value += log_prior_fn(theta, **log_prior_kwargs)
        log_prior_value = log_priors(theta)
        return log_likelihood_value + log_prior_value

    return log_probability