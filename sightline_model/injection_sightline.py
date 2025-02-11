### created 2025-`08 by a. mcbride

import numpy as np
from astropy.io import fits
from .sightline_model import BaseModel

from filehandling import get_medres, get_ca_res, get_madgics_res, getapStar, getASPCAP

lambda0 = 15272.42
sigma0 = 1.15

class InjectionSightline(BaseModel):
    def __init__(self, stars, rvelo, dAVdd = None, injectRealContinuum = True, bins = None, **kwargs):
        super().__init__()
        self.actual_DIBs = np.copy(self.signals)
        self.actual_DIB_errs = np.copy(self.signal_errs)
        self.makeSyntheticDIBs(rvelo, dAVdd = dAVdd, injectRealContinuum = injectRealContinuum)

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


    def makeSyntheticDIBs(self, rvelo, dust_data, dAVdd = None, injectRealContinuum = True):
        continua = np.zeros(self.signals.shape)
        continua_errs = np.zeros(self.signals.shape)
        dustcolumn = np.zeros((len(self.stars), dust_data.dustmap.shape[-1]))
        for i in range(len(self.stars)):
            star = self.stars[i]
            rv_star = star['VHELIO_AVG']
            if injectRealContinuum:
                continuum, continuum_uncertainty = self.getAnalogContinuum(star, rv_star)
                continua[i, :] = continuum[self.window]
                continua_errs[i, :] = continuum_uncertainty[self.window]
            else:
                continuum = 1 + np.random.normal(scale = 1/star['SNR'], size = np.sum(self.window))
                continuum_uncertainty = np.ones(np.sum(self.window)) * 1/star['SNR']
                continua[i, :] = continuum
                continua_errs[i, :] = continuum_uncertainty

            l_i, b_i = self.find_nearest_angular(star['GLON'], star['GLAT'])
            d_i = self.find_nearest_dist(star['DIST']).item()
            if dAVdd is not None:
                dcol = np.copy(dAVdd)
                dcol[d_i:] = 0
            else:
                dcol = dust_data.dustmap[b_i, l_i, :]
                dcol[d_i:] = 0
            dustcolumn[i, :] = dcol

                

        raw_DIB = self.integrateMockDIB(rvelo, dustcolumn)
        signals = raw_DIB - 1 + continua

        self.signals = signals
        self.signal_errs = continua_errs
        self.raw_DIB = raw_DIB
        self.continuum = continua
        self.dustcolumn = dustcolumn

    def getAnalogContinuum(self, star, rv_star, reference_stars = highLat):
            SNRdiff = np.abs(reference_stars['SNR'] - star['SNR'])
            TEFFdiff = np.abs(reference_stars['TEFF'] - star['TEFF'])
            LOGGdiff = np.abs(reference_stars['LOGG'] - star['LOGG'])
            M_Hdiff = np.abs(reference_stars['M_H'] - star['M_H'])
            starAnalogs = np.logical_and.reduce([(SNRdiff < 30), (TEFFdiff < 250), (LOGGdiff < 0.2), (M_Hdiff < 0.1)]) 
            analog_i = np.argmin(reference_stars[starAnalogs]['SFD_EBV'])
            analog = reference_stars[analog_i]
            medres = fits.open(get_medres(analog['TEFF'], analog['LOGG'], analog['M_H']))
            aspcap = fits.open(getASPCAP(analog))
            apstar = fits.open(getapStar(aspcap))
            res, res_err = generateClippedResidual(aspcap, medres, apstar, rv_star)
            return res, res_err
            
    def integrateMockDIB(self, rvelo, dAVdd):
        print(rvelo.shape)
        print(dAVdd.shape)
        signals = np.zeros((len(self.stars), len(self.wavs_window)))
        peak_wavelength = dopplershift(rvelo)
        wavs_grid = np.tile(self.wavs_window, (len(rvelo), 1))
        print(wavs_grid.shape)
        voxel_DIB_unscaled = np.exp(-(wavs_grid - peak_wavelength[:, np.newaxis])**2 / (2 * sigma0**2))
        amp = differentialAmplitude(dAVdd, 1)

        def single_signal(amp, bindex):
            # amp[bindex :] = 0 # THIS MIGHT NEED TO BE -1

            voxel_DIB_scaled = -voxel_DIB_unscaled *  amp[:, np.newaxis] 
            summed_DIB = np.sum(voxel_DIB_scaled, axis = 0)
            # continuum = lambda x, m, b : m * (x - lambda0) + b
            # cont = continuum(self.wavs_window, 0, b)
            return summed_DIB  + 1


        for i in range(len(self.stars)):
            star = self.stars[i]
            dAVdd_star = dAVdd[i, :]
            amp = differentialAmplitude(dAVdd_star, 1)

            bin_index = self.bin_inds[i]
            # signals[i, :] = single_signal(bin_index)
            signals[i, :] = single_signal(amp, 0)# bin_index)
        return signals

    def model_signals(self, rvelo, dAVdd = None, binsep = None):
        if dAVdd is None:
            dAVdd = self.dAVdd
        if binsep is None:
            binsep = self.bins[1:]-self.bins[:-1]
        signals = np.zeros((len(self.stars), len(self.wavs_window)))
        peak_wavelength = self.dopplershift(rvelo)
        wavs_grid = np.tile(self.wavs_window, (len(self.bins) - 1, 1))
        voxel_DIB_unscaled = np.exp(-(wavs_grid - peak_wavelength[:, np.newaxis])**2 / (2 * self.sigma0**2))
        amp = self.differentialAmplitude(dAVdd, binsep)

        def single_signal(amp, bindex):
            amp[bindex :] = 0 # THIS MIGHT NEED TO BE -1

            voxel_DIB_scaled = -voxel_DIB_unscaled *  amp[:, np.newaxis] 
            summed_DIB = np.sum(voxel_DIB_scaled, axis = 0)
            return summed_DIB  + 1


        for i in range(len(self.stars)):
            star = self.stars[i]
            dAVdd_star = dAVdd[i, :]
            amp = self.differentialAmplitude(dAVdd_star, 1)

            bin_index = self.bin_inds[i]
            signals[i, :] = single_signal(amp, bin_index)
        return signals

class Sightline(BaseModel): 
    def __init__(self, stars, coordinates,dust_data, bins = None, star_selection = None, **kwargs):
        super().__init__()
        # if star_selection is None:
        l, b = coordinates
        self.stars = stars[self.select_near_point(stars, l, b)]
        # else: 
        #     self.stars = star_selection(stars, l, b)

        dist = stars['DIST']
        if bins is not None:
            h = np.histogram(dist, bins)[0]
            self.bins = np.insert(bins[1:][h != 0], 0, bins[0])
            self.bin_inds = np.digitize(dist, bins)
        else:
            self.make_bins()
            self.bin_inds = np.digitize(dist, self.bins)  
        self.rvelo = np.zeros(len(self.bins) - 1)
        self.get_DIBs(dust_data, **kwargs)
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

    def get_DIBs(self, dust_data, MADGICS = False, alternative_data_processing = None, **kwargs):
        signals = np.zeros((len(self.stars), len(self.wavs_window)))
        signal_errs = np.zeros((len(self.stars), len(self.wavs_window)))
        dAVdd = np.zeros((len(self.stars), len(self.bins)-1))
        dAVdd_all = np.zeros((len(self.stars), len(self.bins)-1))
        dAVdd_mask =np.zeros((len(self.stars), len(self.bins)-1)).astype(bool)

        if alternative_data_processing is not None:
            # needs to take aspcap, medres, apstar, rv as arguments
            for i in range(len(self.stars)):
                star = self.stars[i]
                star_rv = star['VHELIO_AVG']
                aspcap = fits.open(self.getASPCAP(star))
                apstar = fits.open(self.getapStar(aspcap))
                medres = fits.open(self.get_medres(star['TEFF'], star['LOGG'], star['M_H']))
                sig, err = alternative_data_processing(self.aspcap, medres, apstar, star_rv)
                signals[i, :], signal_errs[i, :] = sig[self.self.window], err[self.self.window]

               
                l, b = star['GLON'], star['GLAT']
                dAVdd[i], dAVdd_all[i], dAVdd_mask[i] = self.generate_dAV_dd_array(l, b, self.bins, star['DIST'], **kwargs)
        
        else:
            if MADGICS:
                signals_aspcap = np.zeros((len(self.stars), len(self.wavs_window)))
                signal_errs_aspcap = np.zeros((len(self.stars), len(self.wavs_window)))

            for i in range(len(self.stars)):
                star = self.stars[i]
                star_rv = star['VHELIO_AVG']
                res_hdul = fits.open(get_ca_res(star['FILE']))
                signals[i, :] = res_hdul[1].data[self.self.window]
                signal_errs[i, :] = res_hdul[2].data[self.self.window]
                reprocess_uncertainty = True
                # if reprocess_uncertainty:
                #     signal_errs[i, :] = self.reprocess_errs(res_hdul, star['VHELIO_AVG'])[self.self.window]
                # reprocess_residual = True
                # if reprocess_residual:
                #     res_repr, err_repr = reprocess(res_hdul, star['VHELIO_AVG'])
                #     signals[i, :] = res_repr[self.self.window]
                #     signal_errs[i, :] = err_repr[self.self.window]
                l, b = star['GLON'], star['GLAT']
                dAVdd[i], dAVdd_all[i], dAVdd_mask[i] = self.generate_dAV_dd_array(l, b,  star['DIST'], self.bins, dust_data, **kwargs)

                if MADGICS:
                    signals_aspcap[i, :] = np.copy(signals[i, :])
                    signal_errs_aspcap[i, :] = np.copy(signal_errs[i, :])
                    res_hdul_m = fits.open(self.get_madgics_res(star['FILE']))
                    signals[i, :] = res_hdul_m[1].data[0, 125:][self.self.window]
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

    def model_signals(self, rvelo, dAVdd = None, binsep = None):
        if dAVdd is None:
            dAVdd = self.dAVdd
        if binsep is None:
            binsep = self.bins[1:]-self.bins[:-1]
        signals = np.zeros((len(self.stars), len(self.wavs_window)))
        peak_wavelength = self.dopplershift(rvelo)
        wavs_grid = np.tile(self.wavs_window, (len(self.bins) - 1, 1))
        voxel_DIB_unscaled = np.exp(-(wavs_grid - peak_wavelength[:, np.newaxis])**2 / (2 * self.sigma0**2))
        amp = self.differentialAmplitude(dAVdd, binsep)

        def single_signal(amp, bindex):
            amp[bindex :] = 0 # THIS MIGHT NEED TO BE -1

            voxel_DIB_scaled = -voxel_DIB_unscaled *  amp[:, np.newaxis] 
            summed_DIB = np.sum(voxel_DIB_scaled, axis = 0)
            return summed_DIB  + 1


        for i in range(len(self.stars)):
            star = self.stars[i]
            dAVdd_star = dAVdd[i, :]
            amp = self.differentialAmplitude(dAVdd_star, 1)

            bin_index = self.bin_inds[i]
            signals[i, :] = single_signal(amp, bin_index)
        return signals