import numpy as np
from astropy.io import fits
from specfns import get_wavs, dopplershift, resample_interp
import globalvars 
import astropy.units as u
from residual_process import reprocess

from scipy.signal import correlate, correlation_lags
from scipy.optimize import curve_fit


lambda0 = 15272.42
sigma0 = 1.15

from spacefns_v2 import dAV_dd_array, differentialAmplitude
from filehandling import get_ca_res, get_madgics_res, get_medres, getASPCAP, getapStar

wavs = get_wavs()
window = (wavs > lambda0 -10) & (wavs < lambda0 + 10)
wavs_window = wavs[window]

class Sightline:
    ### Container object for stars, sightline dAV_dd, velocity, and spaxel assignment.
    def __init__(self, stars, bins = None, **kwargs):
        self.stars = stars
        dist = self.stars['DIST']

        if bins is not None:
            h = np.histogram(dist, bins)[0]
            self.bins = np.insert(bins[1:][h != 0], 0, bins[0])
            self.bin_inds = np.digitize(dist, bins)
        else:
            self.make_bins()
            self.bin_inds = np.digitize(dist, self.bins)  
        self.rvelo = np.zeros(len(self.bins) - 1)
        self.get_DIBs(**kwargs)
        self.init_signals = self.model_signals(self.rvelo, self.dAVdd)
        self.ndim = len(self.voxel_dAVdd)
        self.nsig = len(self.stars)
        
    def make_bins(self, binsep = 10, dmin = 0):
        ### Assigns stars to distance bins if bins are not already supplied.
        dist = self.stars['DIST']
        bins = np.sort(np.insert(np.delete(dist, np.where(dist <= dmin)[0]), 0, dmin))

        i = 0
        while i >= 0:
            try:
                next_bin = np.min(bins[bins > bins[i]])
            except:
                print('broke:')
                print(bins[bins > bins[i]])
                print(len(self.stars))

            bins[i+1] = np.max([next_bin, bins[i] + binsep]) + 0.01
            if bins[i+1] >= np.max(dist):
                bins = bins[:i+2]
                i = -np.inf
            i = i+1
        
        self.bins = bins
            
    def get_DIBs(self, MADGICS = False, alternative_data_processing = None, **kwargs):
        signals = np.zeros((len(self.stars), len(wavs_window)))
        signal_errs = np.zeros((len(self.stars), len(wavs_window)))
        dAVdd = np.zeros((len(self.stars), len(self.bins)-1))
        dAVdd_all = np.zeros((len(self.stars), len(self.bins)-1))
        dAVdd_mask =np.zeros((len(self.stars), len(self.bins)-1)).astype(bool)
        # dAVdd_v = np.zeros((len(self.stars), len(self.bins)-1))
        # print(len(dAVdd_all))
        if MADGICS:
            signals_aspcap = np.zeros((len(self.stars), len(wavs_window)))
            signal_errs_aspcap = np.zeros((len(self.stars), len(wavs_window)))


        # dAVdd = np.zeros(len(self.bins))

        for i in range(len(self.stars)):
            star = self.stars[i]
            star_rv = star['VHELIO_AVG']
            res_hdul = fits.open(get_ca_res(star['FILE']))
            signals[i, :] = res_hdul[1].data[window]
            signal_errs[i, :] = res_hdul[2].data[window]
            reprocess_uncertainty = True
            if reprocess_uncertainty:
                signal_errs[i, :] = self.reprocess_errs(res_hdul, star['VHELIO_AVG'])[window]
            reprocess_residual = True
            if reprocess_residual:
                res_repr, err_repr = reprocess(res_hdul, star['VHELIO_AVG'])
                signals[i, :] = res_repr[window]
                signal_errs[i, :] = err_repr[window]
            l, b = star['GLON'], star['GLAT']
            dAVdd[i], dAVdd_all[i], dAVdd_mask[i] = dAV_dd_array(l, b, self.bins, star['DIST'], **kwargs)
            # dAVdd_v[i], _, __ = dAV_dd_array_v(l, b, self.bins, star['DIST'])


            if MADGICS:
                signals_aspcap[i, :] = np.copy(signals[i, :])
                signal_errs_aspcap[i, :] = np.copy(signal_errs[i, :])
                res_hdul_m = fits.open(get_madgics_res(star['FILE']))
                signals[i, :] = res_hdul_m[1].data[0, 125:][window]
                # signal_errs[i, :] = res_hdul_m[5].data[0, 125:][window]
                # print(res_hdul[2].data.shape)
                # errs = resample_interp(res_hdul[2].data, rv = - star['VHELIO_AVG'])
                # # if 
                # errs = resample_interp(errs, rv = np.median(res_hdul_m[3].data['MADGICS_VBARY']))
                # signal_errs[i, :] = errs[window]
            
            # if alternative_data_processing is not None:
            #     signals[i, :], signal_errs[i, :] = alternative_data_processing()



        self.signals = signals
        self.signal_errs = signal_errs
        self.dAVdd = dAVdd
        self.voxel_dAVdd = np.nanmedian(dAVdd_all, axis = 0)
        self.voxel_dAVdd_std = np.nanstd(dAVdd_all, axis = 0, ddof = 1)
        self.dAVdd_mask = dAVdd_mask.astype(bool)
        # self.dAVdd_v = dAVdd_v
        if MADGICS:
            self.signals_aspcap = signals_aspcap
            self.signal_errs_aspcap = signal_errs_aspcap
        # print(self.voxel_dAVdd.shape)
        # self.dAVdd = dAV_dd_array(np.median(self.stars['GLON']), np.median(self.stars['GLAT']), 
        #                           self.bins, np.max(self.bins))

    def model_signals(self, rvelo, dAVdd = None, binsep = None):
        if dAVdd is None:
            dAVdd = self.dAVdd
        if binsep is None:
            binsep = self.bins[1:]-self.bins[:-1]
        # print('dAVdd shape: ', dAVdd.shape)
        # dAVdd[self.dAVdd_mask] = 0
        signals = np.zeros((len(self.stars), len(wavs_window)))
        peak_wavelength = dopplershift(rvelo)
        wavs_grid = np.tile(wavs_window, (len(self.bins) - 1, 1))
        voxel_DIB_unscaled = np.exp(-(wavs_grid - peak_wavelength[:, np.newaxis])**2 / (2 * sigma0**2))
        amp = differentialAmplitude(dAVdd, binsep)

        def single_signal(amp, bindex):
            amp[bindex :] = 0 # THIS MIGHT NEED TO BE -1

            voxel_DIB_scaled = -voxel_DIB_unscaled *  amp[:, np.newaxis] 
            summed_DIB = np.sum(voxel_DIB_scaled, axis = 0)
            # continuum = lambda x, m, b : m * (x - lambda0) + b
            # cont = continuum(wavs_window, 0, b)
            return summed_DIB  + 1


        for i in range(len(self.stars)):
            star = self.stars[i]
            dAVdd_star = dAVdd[i, :]
            # amp = Differential_Amplitude(dAVdd_star, self.bins[1:]-self.bins[:-1])
            amp = differentialAmplitude(dAVdd_star, 1)

            bin_index = self.bin_inds[i]
            # signals[i, :] = single_signal(bin_index)
            signals[i, :] = single_signal(amp, bin_index)
        return signals
    
    # def intake_full(self, sampler_full):
    #     self.sampler_full = sampler_full
  
    def intake(self, sampler):
        self.sampler = sampler

        samples = sampler.chain[:, int(sampler.chain.shape[1]/2):, :].reshape((-1, sampler.chain.shape[-1]))

        medians = np.nanmedian(samples[:, :], axis = 0)
        stdevs = np.nanstd(samples, axis = 0, ddof = 1)


        med_dAV_dd = medians[self.ndim:].reshape(-1, self.ndim)
        std_dAV_dd = stdevs[self.ndim:].reshape(-1, self.ndim)
        self.dAVddd_derived = med_dAV_dd
        self.dAVdd_derived_err = std_dAV_dd

    def intake_coords(self, l, b, AV = None):
        self.l = l
        self.b = b
        self.AV = AV


    def reprocess_errs(self, hdul, rv):
        flux_uncertainty_obs = hdul[2].data
        medres_uncertainty_rest = hdul[3].data[5, :]
        medres_uncertainty_obs = resample_interp(medres_uncertainty_rest,rv )
        return np.sqrt(flux_uncertainty_obs**2 + medres_uncertainty_obs**2)
    
######## 
    

    
class ForegroundModifiedSightline(Sightline):
    def __init__(self, stars, coords = None, dAVdd = None, dfore = 400, **kwargs):
        # self.all_stars = stars
        self.stars = stars[stars['DIST'] > dfore]
        dist = self.stars['DIST']

        self.make_fgbins()
        self.bin_inds = np.digitize(dist, self.bins)

        if coords is not None:
            self.l, self.b = coords
        else:
            self.l, self.b = (np.nanmean(self.stars['GLON']), np.nanmean(self.stars['GLAT']))
        
        self.rvelo = np.zeros(len(self.bins) - 1)
        self.get_DIBs(**kwargs)

        self.ndim = len(self.voxel_dAVdd)
        self.nsig = len(self.stars)

        self.test_init_signals = self.model_signals_fg(self.rvelo, self.dAVdd)
    
    def get_DIBs(self, MADGICS = False, alternative_data_processing = None, **kwargs):
        signals = np.zeros((len(self.stars), len(wavs_window)))
        signal_errs = np.zeros((len(self.stars), len(wavs_window)))
        dAVdd = np.zeros((len(self.stars), len(self.bins)-1))
        dAVdd_all = np.zeros((len(self.stars), len(self.bins)-1))
        dAVdd_mask =np.zeros((len(self.stars), len(self.bins)-1)).astype(bool)

        if alternative_data_processing is not None:
            # needs to take aspcap, medres, apstar, rv as arguments
            for i in range(len(self.stars)):
                star = self.stars[i]
                star_rv = star['VHELIO_AVG']
                aspcap = fits.open(getASPCAP(star))
                apstar = fits.open(getapStar(aspcap))
                medres = fits.open(get_medres(star['TEFF'], star['LOGG'], star['M_H']))
                sig, err = alternative_data_processing(aspcap, medres, apstar, star_rv)
                signals[i, :], signal_errs[i, :] = sig[window], err[window]

               
                l, b = star['GLON'], star['GLAT']
                dAVdd[i], dAVdd_all[i], dAVdd_mask[i] = dAV_dd_array(l, b, self.bins, star['DIST'], **kwargs)
        
        else:
            if MADGICS:
                signals_aspcap = np.zeros((len(self.stars), len(wavs_window)))
                signal_errs_aspcap = np.zeros((len(self.stars), len(wavs_window)))

            for i in range(len(self.stars)):
                star = self.stars[i]
                star_rv = star['VHELIO_AVG']
                res_hdul = fits.open(get_ca_res(star['FILE']))
                signals[i, :] = res_hdul[1].data[window]
                signal_errs[i, :] = res_hdul[2].data[window]
                reprocess_uncertainty = True
                if reprocess_uncertainty:
                    signal_errs[i, :] = self.reprocess_errs(res_hdul, star['VHELIO_AVG'])[window]
                reprocess_residual = True
                if reprocess_residual:
                    res_repr, err_repr = reprocess(res_hdul, star['VHELIO_AVG'])
                    signals[i, :] = res_repr[window]
                    signal_errs[i, :] = err_repr[window]
                l, b = star['GLON'], star['GLAT']
                dAVdd[i], dAVdd_all[i], dAVdd_mask[i] = dAV_dd_array(l, b, self.bins, star['DIST'], **kwargs)

                if MADGICS:
                    signals_aspcap[i, :] = np.copy(signals[i, :])
                    signal_errs_aspcap[i, :] = np.copy(signal_errs[i, :])
                    res_hdul_m = fits.open(get_madgics_res(star['FILE']))
                    signals[i, :] = res_hdul_m[1].data[0, 125:][window]



        self.signals = signals
        self.signal_errs = signal_errs
        self.dAVdd = dAVdd
        self.voxel_dAVdd = np.nanmedian(dAVdd_all, axis = 0)
        self.voxel_dAVdd_std = np.nanstd(dAVdd_all, axis = 0, ddof = 1)
        self.dAVdd_mask = dAVdd_mask.astype(bool)
        # self.dAVdd_v = dAVdd_v
        if MADGICS:
            self.signals_aspcap = signals_aspcap
            self.signal_errs_aspcap = signal_errs_aspcap

    def make_fgbins(self, binsep = 10, dfore = 400, **kwargs):
        dmin = 0 # start bins at 0pc
        dist = self.stars['DIST']
        bins = np.sort(np.insert(np.delete(dist, np.where(dist <= dmin)[0]), [0,1], [dmin, dfore]))
        # print('BINS BEFORE THING', bins)
        i = 0
        while i >= 0:
            try:
                next_bin = np.min(bins[bins > bins[i]])
            except:
                print('broke:')
                print(bins[bins > bins[i]])
                print(len(self.stars))

            bins[i+1] = np.max([next_bin, bins[i] + binsep]) + 0.01
            if bins[i+1] >= np.max(dist):
                bins = bins[:i+2]
                i = -np.inf
            i = i+1
        
        self.bins = bins

    def model_signals_fg(self, rvelo, dAVdd=None, binsep = None):
        if dAVdd is None:
            dAVdd = self.dAVdd
        if binsep is None:
            binsep = self.bins[1:]-self.bins[:-1]
        signals = np.zeros((len(self.stars), len(wavs_window)))
        peak_wavelength = dopplershift(rvelo)
        wavs_grid = np.tile(wavs_window, (len(self.bins)-1, 1))
        voxel_DIB_unscaled = np.exp(-(wavs_grid - peak_wavelength[:, np.newaxis])**2 / (2 * sigma0**2))
        amp = differentialAmplitude(dAVdd, binsep)

        def single_signal(amp, bindex):
            amp[bindex :] = 0 # THIS MIGHT NEED TO BE -1

            voxel_DIB_scaled = -voxel_DIB_unscaled *  amp[:, np.newaxis] 
            summed_DIB = np.sum(voxel_DIB_scaled, axis = 0)
            # continuum = lambda x, m, b : m * (x - lambda0) + b
            # cont = continuum(wavs_window, 0, b)
            return summed_DIB  + 1


        for i in range(len(self.stars)):
            star = self.stars[i]
            dAVdd_star = dAVdd[i, :]
            # amp = Differential_Amplitude(dAVdd_star, self.bins[1:]-self.bins[:-1])
            amp = differentialAmplitude(dAVdd_star, 1)

            bin_index = self.bin_inds[i]
            # signals[i, :] = single_signal(bin_index)
            signals[i, :] = single_signal(amp, bin_index)
        return signals
    

class ForegroundModifiedSightline_v2(ForegroundModifiedSightline):
    def __init__(self, input_table, cube_CO, distEW_vector = (0.05, 0.05), coords = None, dAVdd = None, dfore = 400, **kwargs):
        # self.all_stars = stars

        table_filament = input_table[self.selectOnFilamentCO(input_table, cube_CO)]
        stars = table_filament[self.selectOnEW(table_filament, distEW_vector)]
        self.stars = stars[(stars['DIST'] > dfore) & (stars['DIST'] < 650)]
        dist = self.stars['DIST']

        self.make_fgbins()
        self.bin_inds = np.digitize(dist, self.bins)

        if coords is not None:
            self.l, self.b = coords
        else:
            self.l, self.b = (np.nanmean(self.stars['GLON']), np.nanmean(self.stars['GLAT']))
        
        self.rvelo = np.zeros(len(self.bins) - 1)
        self.get_DIBs(**kwargs)

        self.ndim = len(self.voxel_dAVdd)
        self.nsig = len(self.stars)

        self.test_init_signals = self.model_signals_fg(self.rvelo, self.dAVdd)

    def get_DIBs(self, MADGICS = False, **kwargs):
        signals = np.zeros((len(self.stars), len(wavs_window)))
        signal_errs = np.zeros((len(self.stars), len(wavs_window)))
        dAVdd = np.zeros((len(self.stars), len(self.bins)-1))
        dAVdd_all = np.zeros((len(self.stars), len(self.bins)-1))
        dAVdd_mask =np.zeros((len(self.stars), len(self.bins)-1)).astype(bool)

        
        for i in range(len(self.stars)):
            star = self.stars[i]
            star_rv = star['VHELIO_AVG']
            aspcap = fits.open(getASPCAP(star))
            apstar = fits.open(getapStar(aspcap))
            medres = fits.open(get_medres(star['TEFF'], star['LOGG'], star['M_H']))
            sig, err = self.generateClippedResidual(aspcap, medres, apstar, star_rv, **kwargs)
            signals[i, :], signal_errs[i, :] = sig[window], err[window]

            
            l, b = star['GLON'], star['GLAT']
            dAVdd[i], dAVdd_all[i], dAVdd_mask[i] = dAV_dd_array(l, b, self.bins, star['DIST'], **kwargs)

    @staticmethod
    def selectOnFilamentCO(tab, cube_CO, threshold = 0.03):
        b_CO, l_CO = cube_CO.world[0, :, :][1:]
        b_CO, l_CO = b_CO[:, 0], l_CO[0, :]
        CO_star_indices = np.array([[ np.argmin((tab['GLAT'][i] - b_CO.value)**2), np.argmin((tab['GLON'][i] - l_CO.value)**2)] for i in range(len(tab))])
        co_i, co_j = np.argmin(np.abs(l_CO.value -167.4)), np.argmin(np.abs(b_CO.value + 8.3))
        reference_point = cube_CO.unmasked_data[:, co_j, co_i]
        corr_lags = correlation_lags(cube_CO.shape[0], cube_CO.shape[0])

        zpoint = corr_lags == 0
        correlation_image = np.zeros((cube_CO.shape[1], cube_CO.shape[2]))
        for i in range(cube_CO.shape[1]):
            for j in range(cube_CO.shape[2]):
                correlation_image[i, j] = correlate(cube_CO.unmasked_data[:, i, j] / np.nansum(np.abs(cube_CO.unmasked_data[:, i, j])), reference_point / np.nansum(np.abs(reference_point)))[zpoint]
        stars_CO_correlation = correlation_image[CO_star_indices[:, 0], CO_star_indices[:, 1]]
        return stars_CO_correlation > threshold


    @staticmethod
    def selectOnEW(tab, vector, select_first = None, norm_vals = None, plot = False):
        """Selects stars based on EW"""
        ew = tab['DIB_EQW']
        dist = tab['DIST']
        if norm_vals is not None:
            dist_norm = (dist - norm_vals[0]) / (norm_vals[1] - norm_vals[0])
            ew_norm = (ew - norm_vals[2]) / (norm_vals[3] - norm_vals[2])
        else:
            distperc5, distperc95 = np.percentile(dist, [5, 95])
            dist_norm = (dist - distperc5) / (distperc95 - distperc5)
            ewperc5, ewperc95 = np.percentile(ew, [5, 95])
            ew_norm = (ew - ewperc5) / (ewperc95 - ewperc5)

        if select_first is None:
            select_first = np.random.choice(len(tab), size = 1)[0]

        def next_star(star_idx, vector, direction = +1):
            dist_norm_star = dist_norm[star_idx]
            ew_norm_star = ew_norm[star_idx]
            diff_dist = direction * (dist_norm - dist_norm_star - direction * vector[0])
            diff_ew = direction * (ew_norm - ew_norm_star - direction * vector[1])
            diff_dist[diff_dist <= 0.0] = np.inf
            diff_ew[diff_ew <= 0.0] = np.inf
            next_star_idx = np.argmin(np.sqrt(diff_dist**2 + diff_ew**2))
            if diff_dist[next_star_idx] == np.inf:
                return np.array([])
            return np.array([next_star_idx])
        
        select_above, select_below = True, True
        selection = np.array([select_first])
        while select_above or select_below:
            if select_above:
                next_star_above = next_star(selection[-1], vector, direction = +1)
                selection = np.concatenate([selection, next_star_above]).astype(int)
                if len(next_star_above) == 0:
                    select_above = False
            
            if select_below:
                next_star_below = next_star(selection[0], vector, direction = -1)
                selection = np.concatenate([next_star_below, selection]).astype(int)
                if len(next_star_below) == 0:
                    select_below = False

        return selection 
    
    @staticmethod
    def generateClippedResidual(aspcap, medres, apstar, rv, k = 3):
        dibfn = lambda x, mu, sigma, a: 1-a * np.exp(-0.5 * (x - mu)**2 / sigma**2)
        def sigma_clip_mask(y, x = wavs, k = 2.5):
            y_over_gauss = None

            try:
                gaussfit = curve_fit(dibfn, x[window], y[window].filled(np.nan), p0 = (15272, 1.2, 0.05), bounds = ([15269, 0.5, 0], [15275, 2, 0.15]), check_finite = False, nan_policy = 'omit')

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

            med = np.nanmedian(y_over_gauss[window])
            stdev = np.std(y_over_gauss[window], ddof = 1)
            mask = np.abs(y_over_gauss - med) > k * stdev
            mask = mask + np.roll(mask, -1) + np.roll(mask, -1)
            mask = mask.astype(bool)
            return mask, stdev

        def sigmaClip(y, yerr,k=2.5):
            clip = True
            clip_iters = 0
            std = np.nanstd(y[window], ddof = 1)
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
        manual_masks = np.zeros(len(wavs))
        for sky in sky_residuals:
            wl = sky[0]
            manual_masks[np.argmin(np.abs(wavs - wl))] = True
        manual_masks = manual_masks + np.roll(manual_masks, -1) + np.roll(manual_masks, 1)

        # mask_sigmaclip = np.zeros(len(wavs))


        maskSigClip, clip_success = sigmaClip(res_corr, uncertainty_corr, k = k)

        mask = mask + maskSigClip
        mask = mask.astype(bool)

        res_corr_ma = np.ma.array(res_corr, mask = mask)
        res_corr_filled = res_corr_ma.filled(np.nan)

        res_corr_resamp = resample_interp(res_corr_filled, rv)
        uncertainty_corr_resamp = resample_interp(uncertainty_corr, rv)

        sky_residuals = [(15268.1, 1),(15274.1, 1),(15275.9, 1)]
        manual_masks = np.zeros(len(wavs))
        for sky in sky_residuals:
            wl = sky[0]
            manual_masks[np.argmin(np.abs(wavs - wl))] = True
        manual_masks = manual_masks + np.roll(manual_masks, -1) + np.roll(manual_masks, 1)


        res_corr_resamp = np.ma.array(res_corr_resamp, mask = manual_masks)
        res_corr_resamp = res_corr_resamp.filled(np.nan)


        return res_corr_resamp, uncertainty_corr_resamp#, clip_success