import numpy as np
from astropy.io import fits
from specfns import resample_interp

def reprocess(hdul, rv):
    hdu_rf = hdul[3].data
    res = hdu_rf[0, :] / hdu_rf[2, :]
    res_corr = res / hdu_rf[4, :]

    bitmask = hdu_rf[3, :]
    mask_digits = [0, 1, 2, 9, 12, 13] # 0 BADPIX, 1 CRPIX, 2 SATPIX, 9 PERSIST_HIGH, 12 SIG_SKYLINE, 13 SIG_TELLURIC
    mask = np.zeros(bitmask.shape)
    for digit in mask_digits:
        mask = mask + np.bitwise_and(bitmask.astype(int), 2**digit) 
    error_mask = hdu_rf[1, :] > np.percentile(hdu_rf[1, :], 95)
    mask = (mask + error_mask).astype(int)
    medres_error_mask = hdu_rf[5, :] > np.percentile(hdu_rf[5, :], 95)
    mask = (mask + medres_error_mask).astype(int)
    sky_mask = np.bitwise_and(bitmask.astype(int), 2**12)
    for i in range(-2, 3):
        mask = (mask + np.roll(sky_mask, i)).astype(int)

    mask = mask.astype(bool)

    res_corr_m = np.ma.array(res_corr, mask = mask)
    res_corr_filled = res_corr_m.filled(np.nan)

    uncertainty = hdu_rf[1, :] 
    uncertainty_corr = np.sqrt(uncertainty**2 + hdu_rf[5, :]**2)

    res_corr_resamp = resample_interp(res_corr_filled, rv)
    uncertainty_corr_resamp = resample_interp(uncertainty_corr, rv)

    return res_corr_resamp, uncertainty_corr_resamp

def generateResidual(aspcap, medres, apstar, rv, return_masked = True):
    spectrum = aspcap[1].data
    model = aspcap[3].data
    err = aspcap[2].data
    bitmask = apstar[3].data[0, :]

    if medres[1].data is None:
        medres_model = np.ones(spectrum.shape)
        medres_err =np.zeros(spectrum.shape)
    else:
        medres_model = np.array(medres[1].data)
        medres_err = np.array(medres[3].data)

    mask_digits = [0, 1, 2, 9, 12, 13] # 0 BADPIX, 1 CRPIX, 2 SATPIX, 9 PERSIST_HIGH, 12 SIG_SKYLINE, 13 SIG_TELLURIC
    mask = np.zeros(bitmask.shape)
    for digit in mask_digits:
        mask = mask + np.bitwise_and(bitmask.astype(int), 2**digit) 
    err_mask = err > np.percentile(err, 95)
    mask = (mask + err_mask).astype(int)
    medres_err_mask = medres_err > np.percentile(medres_err, 95)
    mask = (mask + medres_err_mask).astype(int)
    sky_mask = np.bitwise_and(bitmask.astype(int), 2**12)
    for i in range(-2, 3):
        mask = (mask + np.roll(sky_mask, i)).astype(int)

    mask = mask.astype(bool)

    res_corr = spectrum / model / medres_model

    res_corr_m = np.ma.array(res_corr, mask = mask)
    res_corr_filled = res_corr_m.filled(np.nan)

    uncertainty_corr = np.sqrt(err**2 + medres_err**2)

    if not return_masked:
        res_corr_resamp = resample_interp(res_corr, rv)
        uncertainty_corr_resamp = resample_interp(uncertainty_corr, rv)
        return res_corr_resamp, uncertainty_corr_resamp


    res_corr_resamp = resample_interp(res_corr_filled, rv)
    uncertainty_corr_resamp = resample_interp(uncertainty_corr, rv)

    return res_corr_resamp, uncertainty_corr_resamp

def generateCleanedResidual(aspcap, medres, apstar, rv):
    spectrum = aspcap[1].data
    model = aspcap[3].data
    err = aspcap[2].data
    bitmask = apstar[3].data[0, :]

    if medres[1].data is None:
        medres_model = np.ones(spectrum.shape)
        medres_err =np.zeros(spectrum.shape)
    else:
        medres_model = np.array(medres[1].data)
        medres_err = np.array(medres[3].data)

    mask_digits = [0, 1, 2, 9, 12, 13] # 0 BADPIX, 1 CRPIX, 2 SATPIX, 9 PERSIST_HIGH, 12 SIG_SKYLINE, 13 SIG_TELLURIC
    mask = np.zeros(bitmask.shape)
    for digit in mask_digits:
        mask = mask + np.bitwise_and(bitmask.astype(int), 2**digit) 
    err_mask = err > np.percentile(err, 95)
    mask = (mask + err_mask).astype(int)
    medres_err_mask = medres_err > np.percentile(medres_err, 95)
    mask = (mask + medres_err_mask).astype(int)
    sky_mask = np.bitwise_and(bitmask.astype(int), 2**12)
    for i in range(-2, 3):
        mask = (mask + np.roll(sky_mask, i)).astype(int)

    mask = mask.astype(bool)

    res_corr = spectrum / model / medres_model

    # SKY PIXEL AT 15274.1 Angstrom

    res_corr_m = np.ma.array(res_corr, mask = mask)
    res_corr_filled = res_corr_m.filled(np.nan)

    uncertainty_corr = np.sqrt(err**2 + medres_err**2);

    res_corr_resamp = resample_interp(res_corr_filled, rv)
    uncertainty_corr_resamp = resample_interp(uncertainty_corr, rv)

    return res_corr_resamp, uncertainty_corr_resamp

