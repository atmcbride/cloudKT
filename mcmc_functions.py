import numpy as np
from scipy.signal import correlate, correlation_lags

def log_likelihood(theta, sightline = None, **kwargs):
    v, av = theta
    signal = sightline.signals
    sigma = sightline.signal_errs
    val = - 0.5 * np.nansum((signal - sightline.model_signals(v, dAVdd = av))**2 / (sigma**2)) # IS THIS WRONG
    if np.isnan(val):
        # print('fail loglikely')
        return -np.inf
    else:
        return val


# def log_likelihood(theta, sightline = None, **kwargs):
#     v = theta[ :len(sightline.voxel_dAVdd)]
#     av = theta[len(sightline.voxel_dAVdd):].reshape(-1, len(sightline.voxel_dAVdd))
#     signal = sightline.signals
#     sigma = sightline.signal_errs
#     val = - 0.5 * np.nansum((signal - sightline.model_signals(v, dAVdd = ))**2 / (sigma**2)) # IS THIS WRONG
#     if np.isnan(val):
#         # print('fail loglikely')
#         return -np.inf
#     else:
#         return val
    
def log_prior_v(theta, sightline = None, vmin = -8.5, vmax = 17.5, **kwargs):
    v, av = theta
    if (np.any(v < vmin)) or (np.any(v > vmax)):
        return -np.inf
    return 0.0

def log_prior_davdd(theta, sightline = None, **kwargs):
    v, av = theta
    if ((np.any(av < 0))):
        return -np.inf
    return 0.0

def log_prior_davdd_reg(theta, sightline = None, width_factor = 10, **kwargs):
    v, av = theta
    mask = sightline.dAVdd_mask
    av[mask] = np.nan
    avmed = sightline.voxel_dAVdd
    avstd = sightline.voxel_dAVdd_std * width_factor # should be 10
    # lp_val = -np.nansum(np.log(np.sqrt(2 * np.pi))) + np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * avstd**2)))# first part might not be needed
    lp_val =  np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * avstd**2)))
    return lp_val

def log_prior_davdd_reg_group(theta, sightline = None, width_factor = 3, **kwargs):
    v, av = theta
    mask = sightline.dAVdd_mask
    av[mask] = np.nan
    avmed = np.nanmedian(av, axis = 0,)
    avstd = sightline.voxel_dAVdd_std
    lp_val = np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * (width_factor * avstd)**2)))# first part might not be needed
    return lp_val



class Logprior_Foreground:
    def __init__(self, sightline, *args, **kwargs):
        l = sightline.l
        b = sightline.b
        self.distance = sightline.bins[1:]
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
        v, av = theta
        foreground = self.distance <= foreground_distance
        prior_val = np.zeros(self.distance.shape)
        prior_val[foreground] = (- 0.5 * (v - self.pointfit)**2 / (self.pointfit_width**2))[foreground]
        return np.nansum(prior_val)
        
    def logprior_foreground_av(self, theta, sightline = None, foreground_distance = 401):
        v, av = theta
        foreground = self.distance <= foreground_distance
        if np.any(av[:, foreground] > 0.8):
            return -np.inf
        return 0.0
    
class Logprior_Average_Extinction_OLD:
    def __init__(self, sightline, dust, emission, threshold = 0.03, ref_point = (167.4, -8.3)):
        b_em, l_em = emission.world[0, :, :][1:]
        b_em, l_em = b_em[:, 0], l_em[0, :]
        em_i, em_j = np.argmin(np.abs(l_em.value - ref_point[0])), np.argmin(np.abs(b_em.value - ref_point[1]))
        reference_point = emission.unmasked_data[:, em_j, em_i]
        corr_lags = correlation_lags(emission.shape[0], emission.shape[0])
        zpoint = corr_lags == 0

        correlation_image = np.zeros((emission.shape[1], emission.shape[2]))
        for i in range(emission.shape[1]):
            for j in range(emission.shape[2]):
                correlation_image[i, j] = correlate(emission.unmasked_data[:, i, j] / np.nansum(np.abs(emission.unmasked_data[:, i, j])), 
                                                    reference_point / np.nansum(np.abs(reference_point)))[zpoint]
        correlation_selection = np.where(correlation_image > threshold)
        correlation_l, correlation_b = l_em[correlation_selection[1]], b_em[correlation_selection[0]]

        dust_indices = np.array([dust.find_nearest_angular(correlation_l[i].value, correlation_b[i].value) for i in range(len(correlation_l))])
        # dust_coordinates = (dust.l_1d[dust_indices[:, 0]], dust.b_1d[dust_indices[:, 1]]) # for debuging 
        
        dust_profiles = dust.dustmap[dust_indices[:, 1], dust_indices[:, 0]] # remember that the dustmap is in b, l, d
        avg_dust_profile = np.nanmedian(dust_profiles, axis = 0)
        std_dust_profile = np.nanstd(dust_profiles, axis = 0, ddof = 1)

        distance = dust.distance
        n_bins = len(sightline.bins) - 1
        avg_dAVdd = np.zeros(n_bins)
        std_avg_dAVdd = np.zeros(n_bins)
        for i in range(len(avg_dAVdd)):
            bin_min, bin_max = sightline.bins[i], sightline.bins[i + 1]
            bin_profiles = dust_profiles[:, (distance > bin_min) & (distance <= bin_max)]
            # avg_dAVdd[i] = np.nansum(avg_dust_profile[(distance > bin_min) & (distance <= bin_max)])
            # std_avg_dAVdd[i]  = np.sqrt(np.nansum(std_dust_profile[(distance > bin_min) & (distance <= bin_max)]**2)) / np.sum((distance > bin_min) & (distance <= bin_max))
            avg_dAVdd[i] = np.nanmedian(np.nansum(bin_profiles, axis = 1))
            std_avg_dAVdd[i] = np.nanstd(np.nansum(bin_profiles, axis = 1), ddof = 1)

        self.avg_dAVdd = avg_dAVdd
        self.std_dAVdd = std_avg_dAVdd
        pass
        
    def log_prior_avg_av(self, theta, sightline=None, width_factor= 10):
        v, av = theta
        mask = sightline.dAVdd_mask
        av[mask] = np.nan
        avmed = np.nanmedian(av, axis = 0)
        avstd = sightline.voxel_dAVdd_std
        lp_val = np.nansum(- 0.5 * np.nansum((av - self.avg_dAVdd)**2 / ((width_factor * self.std_dAVdd)**2)))
        return lp_val

class Logprior_Average_Extinction_:
    def __init__(self, sightline, dust, emission, threshold = 0.03, ref_point = (167.4, -8.3)):
        b_em, l_em = emission.world[0, :, :][1:]
        b_em, l_em = b_em[:, 0], l_em[0, :]
        em_i, em_j = np.argmin(np.abs(l_em.value - ref_point[0])), np.argmin(np.abs(b_em.value - ref_point[1]))
        reference_point = emission.unmasked_data[:, em_j, em_i]
        corr_lags = correlation_lags(emission.shape[0], emission.shape[0])
        zpoint = corr_lags == 0

        correlation_image = np.zeros((emission.shape[1], emission.shape[2]))
        for i in range(emission.shape[1]):
            for j in range(emission.shape[2]):
                correlation_image[i, j] = correlate(emission.unmasked_data[:, i, j] / np.nansum(np.abs(emission.unmasked_data[:, i, j])), 
                                                    reference_point / np.nansum(np.abs(reference_point)))[zpoint]
        correlation_selection = np.where(correlation_image > threshold)
        correlation_l, correlation_b = l_em[correlation_selection[1]], b_em[correlation_selection[0]]

        dust_indices = np.array([dust.find_nearest_angular(correlation_l[i].value, correlation_b[i].value) for i in range(len(correlation_l))])
        # dust_coordinates = (dust.l_1d[dust_indices[:, 0]], dust.b_1d[dust_indices[:, 1]]) # for debuging 
        
        dust_profiles = dust.dustmap[dust_indices[:, 1], dust_indices[:, 0]] # remember that the dustmap is in b, l, d
        avg_dust_profile = np.nanmedian(dust_profiles, axis = 0)
        std_dust_profile = np.nanstd(dust_profiles, axis = 0, ddof = 1)

        distance = dust.distance
        n_bins = len(sightline.bins) - 1
        avg_dAVdd = np.zeros(n_bins)
        std_avg_dAVdd = np.zeros(n_bins)
        for i in range(len(avg_dAVdd)):
            bin_min, bin_max = sightline.bins[i], sightline.bins[i + 1]
            bin_profiles = dust_profiles[:, (distance > bin_min) & (distance <= bin_max)]
            avg_dAVdd[i] = np.nansum(avg_dust_profile[(distance > bin_min) & (distance <= bin_max)])
            # std_avg_dAVdd[i]  = np.sqrt(np.nansum(std_dust_profile[(distance > bin_min) & (distance <= bin_max)]**2)) / np.sum((distance > bin_min) & (distance <= bin_max))
            avg_dAVdd[i] = np.nanmedian(np.nansum(bin_profiles, axis = 1))
            std_avg_dAVdd[i] = np.nanstd(np.nansum(bin_profiles, axis = 1), ddof = 1)

        self.avg_dAVdd = avg_dAVdd
        self.std_dAVdd = std_avg_dAVdd
        pass
        
    def log_prior_avg_av(self, theta, sightline=None, width_factor= 10):
        v, av = theta
        mask = sightline.dAVdd_mask
        av[mask] = np.nan
        avmed = np.nanmedian(av, axis = 0)
        avstd = sightline.voxel_dAVdd_std
        lp_val = np.nansum(- 0.5 * np.nansum((av - self.avg_dAVdd)**2 / ((width_factor * self.std_dAVdd)**2)))
        return lp_val
    
class Logprior_Average_Extinction:
    def __init__(self, sightline, dust, emission, threshold = 0.03, ref_point = (167.4, -8.3)):
        b_em, l_em = emission.world[0, :, :][1:]
        b_em, l_em = b_em[:, 0], l_em[0, :]
        em_i, em_j = np.argmin(np.abs(l_em.value - ref_point[0])), np.argmin(np.abs(b_em.value - ref_point[1]))
        reference_point = emission.unmasked_data[:, em_j, em_i]
        corr_lags = correlation_lags(emission.shape[0], emission.shape[0])
        zpoint = corr_lags == 0

        correlation_image = np.zeros((emission.shape[1], emission.shape[2]))
        for i in range(emission.shape[1]):
            for j in range(emission.shape[2]):
                correlation_image[i, j] = correlate(emission.unmasked_data[:, i, j] / np.nansum(np.abs(emission.unmasked_data[:, i, j])), 
                                                    reference_point / np.nansum(np.abs(reference_point)))[zpoint]
        correlation_selection = np.where(correlation_image > threshold)
        correlation_l, correlation_b = l_em[correlation_selection[1]], b_em[correlation_selection[0]]

        dust_indices = np.array([dust.find_nearest_angular(correlation_l[i].value, correlation_b[i].value) for i in range(len(correlation_l))])
        # dust_coordinates = (dust.l_1d[dust_indices[:, 0]], dust.b_1d[dust_indices[:, 1]]) # for debuging 
        
        dust_profiles = dust.dustmap[dust_indices[:, 1], dust_indices[:, 0]] # remember that the dustmap is in b, l, d
        avg_dust_profile = np.nanmedian(dust_profiles, axis = 0)
        # avg_dust_profile[dust.distance < 400] = 0
        std_dust_profile = np.nanstd(dust_profiles, axis = 0, ddof = 1)

        distance = dust.distance
        n_bins = len(sightline.bins) - 1
        avg_dAVdd = np.zeros(n_bins)
        std_avg_dAVdd = np.zeros(n_bins)
        for i in range(len(avg_dAVdd)):
            bin_min, bin_max = sightline.bins[i], sightline.bins[i + 1]
            bin_profiles = dust_profiles[:, (distance > bin_min) & (distance <= bin_max)]
            avg_dAVdd[i] = np.nansum(avg_dust_profile[(distance > bin_min) & (distance <= bin_max)])
            # std_avg_dAVdd[i]  = np.sqrt(np.nansum(std_dust_profile[(distance > bin_min) & (distance <= bin_max)]**2)) / np.sum((distance > bin_min) & (distance <= bin_max))
            # avg_dAVdd[i] = np.nanmedian(np.nansum(bin_profiles, axis = 1))
            std_avg_dAVdd[i] = np.nanstd(np.nansum(bin_profiles, axis = 1), ddof = 1)

        self.avg_dAVdd = avg_dAVdd
        self.std_dAVdd = std_avg_dAVdd

        
    def log_prior_avg_av(self, theta, sightline=None, width_factor= 10):
        v, av = theta
        mask = sightline.dAVdd_mask
        av[mask] = np.nan
        avmed = np.nanmedian(av, axis = 0)
        avstd = sightline.voxel_dAVdd_std
        lp_val = np.nansum(- 0.5 * np.nansum((av - self.avg_dAVdd)**2 / ((width_factor * self.std_dAVdd)**2)))
        return lp_val


def log_prior(theta, sightline = None, log_priors = [],  **kwargs):
    log_prior_value = 0
    for item in log_priors:
        log_prior_fn, log_prior_kwargs = item
        log_prior_value += log_prior_fn(theta, sightline = sightline, **log_prior_kwargs)
    return log_prior_value
