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


class Logprior_Foreground:
    def __init__(self, sightline, **kwargs):
        l = sightline.l
        b = sightline.b
        self.distance = sightline.stars("DIST")
        self.pointfit = self.polynomial2d(l, b)
        self.pointfit_width = 2.404363059339516

    def polynomial2d(self, x1, x2, theta = None, uncert = None):  
        if theta is None:
            theta = np.array([5.03964666, -1.04129592, -0.72842925, -0.20292219,  0.0206567,  -0.14442016])
        if uncert is None:
            uncert = 2.404363059339516
        if np.array(x1).ndim != 1:
            x1 = np.array([x1])
            x2 = np.array([x2])
        x1 = x1 - 160 # FOR CA CLOUD SPECIFICIALLY
        x2 = x2 + 8 # DITTO
        X = np.array([[np.ones(np.array(x1).shape), x1, x2, x1 * x2, x1**2, x2**2]]).T
        matrix = X * theta[:, np.newaxis]
        return np.nansum(matrix, axis =1).item()
    
    def logprior_foreground_v(self, theta, sightline = None, foreground_distance = 400, **kwargs):    
        v = np.copy(theta[:sightline.ndim])
        foreground = self.distance <= foreground_distance
        prior_val = np.zeros(self.distance.shape)
        prior_val[foreground] = np.nansum(- 0.5 * np.nansum((v - self.pointfit.item)**2 / (self.pointfit_width**2)))
        return prior_val.item()
        
    def logprior_foreground_av(self, theta, sightline = None, foreground_distance = 400):
        av = np.copy(theta[sightline.ndim]).reshape(-1, sightline.ndim)
        foreground = self.distance <= foreground_distance
        prior_val = np.zeros(self.distance.shape)
        ampfit = (0.01928233, 0.01431857)
        avf = lambda x, mu, sigma :  -(x - mu)**2 / (2 * sigma**2)
        prior_val[foreground] = - 0.5 * np.nansum((av - ampfit[0])**2 / (ampfit[1]**2))
        return prior_val.item()

def log_prior(theta, sightline = None, log_priors = [],  **kwargs):
    log_prior_value = 0
    for item in log_priors:
        log_prior_fn, log_prior_kwargs = item
        log_prior_value += log_prior_fn(theta, sightline = sightline, **log_prior_kwargs)
    return log_prior_value
