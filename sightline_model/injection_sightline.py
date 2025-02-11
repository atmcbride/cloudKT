### created 2025-`08 by a. mcbride

import numpy as np
from astropy.io import fits
from cloudKT.sightline_model.oldsightline import Sightline

class InjectionSightline(Sightline):
    def __init__(self, stars, rvelo, dAVdd = None, injectRealContinuum = True, bins = None, **kwargs):
        super().__init__(stars, **kwargs)
        self.actual_DIBs = np.copy(self.signals)
        self.actual_DIB_errs = np.copy(self.signal_errs)
        self.makeSyntheticDIBs(rvelo, dAVdd = dAVdd, injectRealContinuum = injectRealContinuum)

    def makeSyntheticDIBs(self, rvelo, dustmap = dAVdd = None, injectRealContinuum = True):
        continua = np.zeros(self.signals.shape)
        continua_errs = np.zeros(self.signals.shape)
        dustcolumn = np.zeros((len(self.stars), dust_data.dustmap.shape[-1]))
        for i in range(len(self.stars)):
            star = self.stars[i]
            rv_star = star['VHELIO_AVG']
            if injectRealContinuum:
                continuum, continuum_uncertainty = self.getAnalogContinuum(star, rv_star)
                continua[i, :] = continuum[window]
                continua_errs[i, :] = continuum_uncertainty[window]
            else:
                continuum = 1 + np.random.normal(scale = 1/star['SNR'], size = np.sum(window))
                continuum_uncertainty = np.ones(np.sum(window)) * 1/star['SNR']
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
        signals = np.zeros((len(self.stars), len(wavs_window)))
        peak_wavelength = dopplershift(rvelo)
        wavs_grid = np.tile(wavs_window, (len(rvelo), 1))
        print(wavs_grid.shape)
        voxel_DIB_unscaled = np.exp(-(wavs_grid - peak_wavelength[:, np.newaxis])**2 / (2 * sigma0**2))
        amp = differentialAmplitude(dAVdd, 1)

        def single_signal(amp, bindex):
            # amp[bindex :] = 0 # THIS MIGHT NEED TO BE -1

            voxel_DIB_scaled = -voxel_DIB_unscaled *  amp[:, np.newaxis] 
            summed_DIB = np.sum(voxel_DIB_scaled, axis = 0)
            # continuum = lambda x, m, b : m * (x - lambda0) + b
            # cont = continuum(wavs_window, 0, b)
            return summed_DIB  + 1


        for i in range(len(self.stars)):
            star = self.stars[i]
            dAVdd_star = dAVdd[i, :]
            amp = differentialAmplitude(dAVdd_star, 1)

            bin_index = self.bin_inds[i]
            # signals[i, :] = single_signal(bin_index)
            signals[i, :] = single_signal(amp, 0)# bin_index)
        return signals

