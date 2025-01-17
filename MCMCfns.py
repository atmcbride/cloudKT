import numpy as np

def loglikely_2(v, av, sl, **kwargs):
    signal = sl.signals
    sigma = sl.signal_errs
    val = - 0.5 * np.nansum((signal - sl.model_signals(v, dAVdd = av))**2 / (sigma**2)) # IS THIS WRONG
    if np.isnan(val):
        # print('fail loglikely')
        return -np.inf
    else:
        return val


def logprior_v(v, v_max = 5, prior_mult = 1, **kwargs):
    if (np.any(v < -8.5)) or (np.any(v > 17.5)):
        return -np.inf
    return 0.0


def logprior_davdd(av, AV_base = 5, AV_max = 10):   
    if ((np.any(av < 0))):
        return -np.inf
    return 0.0

def logprior_davdd_reg(av,sl, mask = None, **kwargs):
    av = np.copy(av)
    mask = sl.dAVdd_mask
    av[mask] = np.nan
    avmed = sl.voxel_dAVdd
    avstd = sl.voxel_dAVdd_std * 10 # should be 10
    # lp_val = -np.nansum(np.log(np.sqrt(2 * np.pi))) + np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * avstd**2)))# first part might not be needed
    lp_val =  np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * avstd**2)))# first part might not be needed

    return lp_val

def logprior_davdd_reg_group(av,sl, mask = None,  width_factor = 3, **kwargs):
    av = np.copy(av)
    mask = sl.dAVdd_mask
    av[mask] = np.nan
    avmed = np.nanmedian(av, axis = 0,)
    avstd = sl.voxel_dAVdd_std
    # lp_val = - np.nansum(np.log(np.sqrt(2 * np.pi))) + np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * (width_factor * avstd)**2)))# first part might not be needed
    lp_val =  np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * (width_factor * avstd)**2)))# first part might not be needed

    return lp_val

def logprior_davdd_min(av):
    if np.any(av < 0.075):
        return -np.inf
    else:
        return 0.0


def logprob_2(p, sl, logprior = logprior_v, loglikely = loglikely_2, **kwargs):
    ndim = len(sl.voxel_dAVdd)
    v = p[ :ndim]
    av = p[ndim:].reshape(-1, ndim)
    lp = logprior(v, **kwargs)
    lp_davdd = logprior_davdd(av, AV_base = sl.dAVdd)
    lp_davdd_reg = logprior_davdd_reg(av, sl, **kwargs)
    lp_davdd_reg_group = logprior_davdd_reg_group(av, sl)
    if (not np.isfinite(lp)) | (not np.isfinite(lp_davdd)) | (not np.isfinite(lp_davdd_reg)):
        return -np.inf
    return lp + lp_davdd + lp_davdd_reg +  loglikely_2(v, av, sl = sl, **kwargs) + lp_davdd_reg_group # group term added 10.13

def logprob_avfix(p,sl, av = None,  logprior = logprior_v, loglikely = loglikely_2, **kwargs):
    ndim = len(sl.voxel_dAVdd)
    v = p[:ndim]
    lp = logprior(v, **kwargs)
    if (not np.isfinite(lp)):
        return -np.inf
    return lp + loglikely_2(v, av, sl = sl, **kwargs)

class Logprior_Foreground:
    def __init__(self, l, b):
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
    
    def logprior_foreground_v(self, v, distance, foreground_distance = 400, **kwargs):    
        foreground = distance <= foreground_distance
        prior_val = np.zeros(distance.shape)
        prior_val[foreground] = np.nansum(- 0.5 * np.nansum((v - self.pointfit.item)**2 / (self.pointfit_width**2)))
        return prior_val.item()
        
    def logprior_foreground_av(self, av, distance, foreground_distance = 400):
        foreground = distance <= foreground_distance
        prior_val = np.zeros(distance.shape)
        ampfit = (0.01928233, 0.01431857)
        avf = lambda x, mu, sigma :  -(x - mu)**2 / (2 * sigma**2)
        prior_val[foreground] = - 0.5 * np.nansum((av - ampfit[0])**2 / (ampfit[1]**2))
        return prior_val.item()


class BayesFramework:
    def __init__(self, **kwargs):
        self.log_likelihood = None
        self.log_priors = None
        return

    def add_logprior(self):
        return
    
    def logprob(self, p, **kwargs):
        return
    
############################################### 
    
class Logprior_Foreground:
    def __init__(self, l, b):
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
    
    def logprior_foreground_v(self, v, distance, foreground_distance = 401, **kwargs):    
        foreground = distance <= foreground_distance
        prior_val = np.zeros(distance.shape)

        prior_val[foreground] = (- 0.5 * (v - self.pointfit)**2 / (self.pointfit_width**2))[foreground]
        return np.nansum(prior_val)
        
    # def logprior_foreground_av(self, av, distance, foreground_distance = 401):
    #     foreground = distance <= foreground_distance
    #     prior_val = np.zeros(distance.shape)
    #     ampfit = [0.5 * 0.01928233, 0.5 * 0.01431857]
    #     avf = lambda x, mu, sigma :  -(x - mu)**2 / (2 * sigma**2)
    #     prior_val[foreground] = - 0.5 * np.nansum((av[:, foreground] - ampfit[0])**2 / (ampfit[1]**2))
    #     return np.nansum(prior_val)

    def logprior_foreground_av(self, av, distance, foreground_distance = 401):
        foreground = distance <= foreground_distance
        if np.any(av[:, foreground] > 0.8):
            return -np.inf
        return 0.0



def logprob_2(p, sl, logprior = logprior_v, loglikely = loglikely_2, **kwargs): ## NEW AS OF 05.16LIke.
    ndim = len(sl.voxel_dAVdd)
    v = p[ :ndim]
    av = p[ndim:].reshape(-1, ndim)
    lp = logprior(v, **kwargs)
    lp_davdd = logprior_davdd(av, AV_base = sl.dAVdd)
    lp_davdd_reg = logprior_davdd_reg(av, sl, **kwargs)
    lp_davdd_reg_group = logprior_davdd_reg_group(av, sl)
    if (not np.isfinite(lp)) | (not np.isfinite(lp_davdd)) | (not np.isfinite(lp_davdd_reg)):
        return -np.inf
    return lp + lp_davdd  + lp_davdd_reg + loglikely_2(v, av, sl = sl, **kwargs) + lp_davdd_reg_group # group term added 10.13


def logprob_fg(p, sl, lp_fore = None, **kwargs):
    ndim = len(sl.voxel_dAVdd)
    
    lprob = logprob_2(p, sl, **kwargs)
    v = p[ :ndim]
    av = p[ndim:].reshape(-1, ndim) #what shape is dAVddd? 

    ### Added 05.08 ###
    lprior_av_min = logprior_davdd_min(av)
    lprob = lprob + lprior_av_min

    lp_fore_v = lp_fore.logprior_foreground_v(v, sl.bins[1:])
    # lp_fore_av = lp_fore.logprior_foreground_av(av, sl.bins[1:])
    return lprob + lp_fore_v #+ lp_fore_av

# def logprob_2(p, sl, logprior = logprior_v, loglikely = loglikely_2, **kwargs): ## NEW AS OF 05.16LIke.
#     ndim = len(sl.voxel_dAVdd)
#     v = p[ :ndim]
#     av = p[ndim:].reshape(-1, ndim)
#     # lp = logprior(v, **kwargs)
#     lp_davdd = logprior_davdd(av, AV_base = sl.dAVdd)
#     lp_davdd_reg = logprior_davdd_reg(av, sl, **kwargs)
#     lp_davdd_reg_group = logprior_davdd_reg_group(av, sl)
#     if (not np.isfinite(lp)) | (not np.isfinite(lp_davdd)) | (not np.isfinite(lp_davdd_reg)):
#         return -np.inf
#     return  lp_davdd  + lp_davdd_reg + loglikely_2(v, av, sl = sl, **kwargs) + lp_davdd_reg_group # group term added 10.13

# def loglikely_2_example(av, data = data, sigma = err, **kwargs):
#     val = - 0.5 * np.nansum((data - model(av))**2 / (sigma**2)) 
#     return val


# def logprob(av **kwargs): 
#     lp = logprior(av, **kwargs)
#     ll = loglikelihood(av, **kwargs) 
#     return lp  + ll
