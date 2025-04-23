import numpy as np
import emcee
import logging
from functools import partial


from utilities import load_module


logger = logging.getLogger(__name__)

def run_mcmc(sightline, mcmc_config, *args, steps = 1000, nwalkers = 100, pool = None, filename = None, **kwargs):
    """"
    Run the MCMC
    """
    ndim = len(sightline.voxel_dAVdd) 
    nstar = len(sightline.stars)
    ndim_amp = int(ndim + ndim * nstar)

    if nwalkers < 2 * ndim_amp:
        nwalkers = 2 * ndim_amp + 5
        logger.info("N walkers updated to " + str(nwalkers))

    if filename is not None:
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim_amp)
    else:
        backend = None
        logger.warning('NO BACKEND')

    ll_config = mcmc_config["LOG_LIKELIHOOD"]
    ll_module = load_module(ll_config["MODULE"])
    ll_fn = getattr(ll_module, ll_config["FUNCTION"])
    ll_params = ll_config["PARAMETERS"]
    log_likelihood = (ll_fn, ll_params) # Pass as tuple (fn, fn_kwargs)
    

    log_priors = []
    lp_config = mcmc_config["LOG_PRIOR"]
    for lp_entry in lp_config:
        lp_module = load_module(lp_entry["MODULE"])
        if "OBJECT" in lp_entry.keys():
            lp_object = getattr(lp_module, lp_entry["OBJECT"])(sightline, *args, **lp_entry["INIT_KWARGS"])
            lp_fn = getattr(lp_object, lp_entry["FUNCTION"])
            lp_params = lp_entry["PARAMETERS"]
            log_prior = (lp_fn, lp_params)
            log_priors.append(log_prior)

        else:
            lp_fn = getattr(lp_module, lp_entry["FUNCTION"])
            lp_params = lp_entry["PARAMETERS"]
            log_prior = (lp_fn, lp_params)
            log_priors.append(log_prior) # Pass as list of tuples (fn, fn_kwargs)


    sampler = emcee.EnsembleSampler(nwalkers, ndim_amp, log_probability, pool = pool, backend = backend, kwargs = {
                                    'sightline': sightline, 'log_likelihood': log_likelihood, 'log_priors': log_priors})

    # init = 10 *  (np.random.random((nwalkers, ndim_amp)) - 0.5)
    init = 30 *  (np.random.random((nwalkers, ndim_amp)) - 0.5)

    init[:, ndim:] = np.abs(sightline.dAVdd.ravel()[np.newaxis, :] + 0.1*(np.random.random(init[:, ndim:].shape)-0.5))
    # init[:, ndim:][(init[:, ndim:] <= 0.1)] = 0.11 + 0.05 * np.random.random(np.sum(init[:, ndim:]<= 0.1))
    init[:, ndim:][(init[:, ndim:] <= 0.0)] = 0.11 + 0.05 * np.random.random(np.sum(init[:, ndim:]<= 0.0))


    print('NDIM:', ndim, 'NSTAR:', nstar, 'INITSHAPE:', init.shape)

    sampler.run_mcmc(init, steps, progress = False, store = True);

    return sampler

def log_probability(theta, sightline = None, log_likelihood = None, log_priors = None, **kwargs):
    """
    For a given input vector theta, populated Sightline object (for data and modeling functions), log-likelihood function with inputs,
    and a list of log-prior functions with inputs, calculate and return the log-probability of theta 
    """
    ll_fn, ll_kwargs = log_likelihood
    ll = ll_fn(theta, sightline = sightline, **ll_kwargs)
    lp = evaluate_log_prior(theta, log_priors = log_priors, sightline = sightline, **kwargs)

    return ll + lp


def evaluate_log_prior(theta, log_priors = None, sightline = None, **kwargs):
    lp = 0
    for lp_entry in log_priors:
        lp_fn, lp_kwargs = lp_entry
        lp_fn_val = lp_fn(theta, sightline = sightline, **lp_kwargs)
        lp += lp_fn_val
    return lp

def load_from_hdf5(h5_fname):
    reader = emcee.backends.HDFBackend(h5_fname)
    chain = reader.get_chain()
    prob = reader.get_log_prob()
    return reader