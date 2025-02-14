import numpy as np

def log_likelihood(theta, sightline = None, **kwargs):
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
    v = theta[ :sightline.ndim]
    if (np.any(v < -8.5)) or (np.any(v > 17.5)):
        return -np.inf
    return 0.0

def log_prior_davdd(theta, sightline = None, AV_base = 5, AV_max = 10, **kwargs):
    av = theta[sightline.ndim:].reshape(-1, sightline.ndim)
    if ((np.any(av < 0))):
        return -np.inf
    return 0.0

def log_prior_davdd_reg(theta, sightline = None, width_factor = 10, **kwargs):
    av = np.copy(theta[sightline.ndim:].reshape(-1, sightline.ndim)) # needs copy in order to not throw a NaN
    mask = sightline.dAVdd_mask
    av[mask] = np.nan
    avmed = sightline.voxel_dAVdd
    avstd = sightline.voxel_dAVdd_std * width_factor # should be 10
    # lp_val = -np.nansum(np.log(np.sqrt(2 * np.pi))) + np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * avstd**2)))# first part might not be needed
    lp_val =  np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * avstd**2)))
    return lp_val

def log_prior_davdd_reg_group(theta, sightline = None, width_factor = 3, **kwargs):
    av = np.copy(theta[sightline.ndim:].reshape(-1, sightline.ndim))
    mask = sightline.dAVdd_mask
    av[mask] = np.nan
    avmed = np.nanmedian(av, axis = 0,)
    avstd = sightline.voxel_dAVdd_std
    lp_val = np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * (width_factor * avstd)**2)))# first part might not be needed
    return lp_val


def log_prior(theta, sightline = None, log_priors = [],  **kwargs):
    log_prior_value = 0
    for item in log_priors:
        log_prior_fn, log_prior_kwargs = item
        log_prior_value += log_prior_fn(theta, sightline = sightline, **log_prior_kwargs)
    return log_prior_value
