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

# def logprob_2(p, sl, logprior = logprior_v, loglikely = loglikely_2, **kwargs): ## NEW AS OF 05.16LIke.
#     ndim = len(sl.voxel_dAVdd)
#     v = p[ :ndim]
#     av = p[ndim:].reshape(-1, ndim)
#     lp = logprior(v, **kwargs)
#     lp_davdd = logprior_davdd(av, AV_base = sl.dAVdd)
#     lp_davdd_reg = logprior_davdd_reg(av, sl, **kwargs)
#     lp_davdd_reg_group = logprior_davdd_reg_group(av, sl)
#     if (not np.isfinite(lp)) | (not np.isfinite(lp_davdd)) | (not np.isfinite(lp_davdd_reg)):
#         return -np.inf
#     return lp + lp_davdd  + lp_davdd_reg + loglikely_2(v, av, sl = sl, **kwargs) + lp_davdd_reg_group # group term added 10.13