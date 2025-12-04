import numpy as np 


def chi2_statistics(sl, chain):
    samples = chain.swapaxes(0,1)[-100:, :, :].reshape(-1, chain.shape[-1])
    v = np.nanmedian(samples[:, :sl.ndim], axis = 0)
    davdd = np.nanmedian(samples[:, sl.ndim:], axis = 0).reshape(-1, sl.ndim)

    signals = sl.signals
    signal_errs = sl.signal_errs
    modeled_signals = sl.model_signals(v, dAVdd = davdd)
    # this should be 1:1 with signals

    per_star_chi2 = np.nansum((signals - modeled_signals)**2 / signal_errs**2, axis = 1) / np.nansum(np.isnan(signals)==False, axis = 1)
    median_star_chi2 = np.median(per_star_chi2)
    std_star_chi2 = np.std(per_star_chi2, ddof = 1)

    sightline_chi2 = np.nansum((signals - modeled_signals)**2 / signal_errs**2) / np.nansum(np.isnan(signals) == False)

    return per_star_chi2, median_star_chi2, std_star_chi2, sightline_chi2

def bayesian_information_criterion():
    pass

def generate_sightlines_chain_statistics(sightline, chain, fname = "chain_stats", **kwargs):
    acceptance_fraction = np.diff(chain, axis = 0)
