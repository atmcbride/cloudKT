import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
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

################################



def generateClippedResidual(self, star, rv = None, k = 3, **kwargs):
    if rv is None:
        rv = star["VHELIO_AVG"]
    aspcap = fits.open(self.getASPCAP(star))
    apstar = fits.open(self.getapStar(aspcap))
    medres = fits.open(
        self.get_medres(star["TEFF"], star["LOGG"], star["M_H"]))

    dibfn = lambda x, mu, sigma, a: 1-a * np.exp(-0.5 * (x - mu)**2 / sigma**2)
    def sigma_clip_mask(y, x = self.wavs, k = 2.5):
        y_over_gauss = None

        try:
            gaussfit = curve_fit(dibfn, x[self.window], y[self.window].filled(np.nan), p0 = (15272, 1.2, 0.05), bounds = ([15269, 0.5, 0], [15275, 2, 0.15]), check_finite = False, nan_policy = 'omit')

        except:
            # gaussfit = ((15272, 1.2, 0.05),())
            print('fail')
            return None, None
        #     y_over_gauss = None
        #     gaussfit = ((15272, 1.2, 0.05),())
        #     # fit = dibfn(x, 15272.42, 1.2, 0.05)
        #     # y_over_gauss = y / fit
        #     print('POOR GAUSS FIT IN SIGMA CLIP')
        y_over_gauss = y / dibfn(x, *gaussfit[0])

        med = np.nanmedian(y_over_gauss[self.window])
        stdev = np.std(y_over_gauss[self.window], ddof = 1)
        mask = np.abs(y_over_gauss - med) > k * stdev
        mask = mask + np.roll(mask, -1) + np.roll(mask, -1)
        mask = mask.astype(bool)
        return mask, stdev

    def sigmaClip(y, yerr,k=2.5):
        clip = True
        clip_iters = 0
        std = np.nanstd(y[self.window], ddof = 1)
        mask = np.zeros(y.shape).astype(bool)
        clip_success = True

        while clip:
            clip_mask, std_clipped = sigma_clip_mask(np.ma.array(y, mask = mask.copy()), k = k)
            if clip_mask is None:
                clip_success = False
                return mask, clip_success
            clip_mask = clip_mask.filled(False)

            if std - std_clipped  < 1e-4:
                clip = False


            else:
                clip_iters += 1
                std = std_clipped
                mask = mask + clip_mask

        return mask, clip_success

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

    mask = mask.astype(bool)
    res_corr = spectrum / model / medres_model
    # print('res corr shape', res_corr.shape)
    uncertainty_corr = np.sqrt(err**2) #+ medres_err**2)

    sky_residuals = [(15268.1, 1),(15274.1, 1),(15275.9, 1)]
    manual_masks = np.zeros(len(self.wavs))
    for sky in sky_residuals:
        wl = sky[0]
        manual_masks[np.argmin(np.abs(self.wavs - wl))] = True
    manual_masks = manual_masks + np.roll(manual_masks, -1) + np.roll(manual_masks, 1)

    # mask_sigmaclip = np.zeros(len(self.wavs))


    maskSigClip, clip_success = sigmaClip(res_corr, uncertainty_corr, k = k)

    mask = mask + maskSigClip
    mask = mask.astype(bool)

    res_corr_ma = np.ma.array(res_corr, mask = mask)
    res_corr_filled = res_corr_ma.filled(np.nan)

    res_corr_resamp = resample_interp(res_corr_filled, rv)
    uncertainty_corr_resamp = resample_interp(uncertainty_corr, rv)

    sky_residuals = [(15268.1, 1),(15274.1, 1),(15275.9, 1)]
    manual_masks = np.zeros(len(self.wavs))
    for sky in sky_residuals:
        wl = sky[0]
        manual_masks[np.argmin(np.abs(self.wavs - wl))] = True
    manual_masks = manual_masks + np.roll(manual_masks, -1) + np.roll(manual_masks, 1)


    res_corr_resamp = np.ma.array(res_corr_resamp, mask = manual_masks)
    res_corr_resamp = res_corr_resamp.filled(np.nan)


    return res_corr_resamp, uncertainty_corr_resamp#, clip_success