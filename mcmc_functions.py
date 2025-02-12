import numpy as np

def log_likelihood(theta, sightline=None, **kwargs):
    v = theta[ :len(sightline.voxel_dAVdd)]
    av = theta[len(sightline.voxel_dAVdd):].reshape(-1, len(sightline.voxel_dAVdd))
    signal = sightline.signals
    sigma = sightline.signal_errs
    val = - 0.5 * np.nansum((signal - sightline.model_signals(v, dAVdd = av))**2 / (sigma**2)) # IS THIS WRONG
    if np.isnan(val):
        # print('fail loglikely')
        return -np.inf
    else:
        return val
    
def log_prior_v(theta, sightline = None, **kwargs):
    v = theta[ :len(sightline.voxel_dAVdd)]

    if (np.any(v < -8.5)) or (np.any(v > 17.5)):
        return -np.inf
    return 0.0


def log_prior(theta, sightline = None, log_priors = [], **kwargs):
    log_prior_value = 0
    for item in log_priors:
        log_prior_fn, log_prior_kwargs = item
        log_prior_value += log_prior_fn(theta, sightline = sightline, **log_prior_kwargs)
    return log_prior_value
