### created 2025-08 by a. mcbride

import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.io import fits
import logging
from astropy.table import Table

from .base_model import BaseModel
from filehandling import get_medres, get_ca_res, get_madgics_res, getapStar, getASPCAP


logger = logging.getLogger(__name__)

lambda0 = 15272.42
sigma0 = 1.15


def fg_polynomial2d(x1, x2, theta = None, uncert = None):  
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

class ForegroundFixedSightline(BaseModel):
    def __init__(self, stars, dust_data, star_selection_kwargs = None, coordinates = None, dfore = 401, **kwargs):
        super().__init__(self, stars, **kwargs)
        
        if self.select_stars is None:
            l, b = coordinates
            self.stars = stars[self.select_near_point(stars, l, b, **kwargs)]
            self.stars = stars[stars['DIST'] > dfore]
        else:
            stars = self.select_stars(stars, **star_selection_kwargs)
            self.stars = stars[stars['DIST'] > dfore]
        dist = self.stars['DIST']

        self.make_fgbins()
        self.bin_inds = np.digitize(dist, self.bins)

        dist = stars["DIST"]

        if coordinates is not None:
            self.l, self.b = coordinates
        else:
            self.l, self.b = (np.nanmean(self.stars['GLON']), np.nanmean(self.stars['GLAT']))
        
        self.rvelo = np.zeros(len(self.bins) - 1)
        self.get_DIBs(dust_data, **kwargs)

        self.ndim = len(self.voxel_dAVdd)
        self.nsig = len(self.stars)
        self.fg_velo = self.foreground_velocities()

        self.test_init_signals = self.model_signals(self.rvelo, self.dAVdd)

    def get_DIBs(self, dust_data, use_MADGICS=False, **kwargs):
        signals = np.zeros((len(self.stars), len(self.wavs_window)))
        signal_errs = np.zeros((len(self.stars), len(self.wavs_window)))
        dAVdd = np.zeros((len(self.stars), len(self.bins) - 1))
        dAVdd_all = np.zeros((len(self.stars), len(self.bins) - 1))
        dAVdd_mask = np.zeros((len(self.stars), len(self.bins) - 1)).astype(bool)

        if self.alternative_data_processing is not None: 
            # alternative data processing must be assigned outside of class;
            # takes self, aspcap, medres, apstar, star_rv, and **kwargs
            for i in range(len(self.stars)):
                star = self.stars[i]

                sig, err = self.alternative_data_processing(
                    star, **kwargs
                )
                signals[i, :], signal_errs[i, :] = sig[self.window], err[self.window]

                l, b = star["GLON"], star["GLAT"]
                dAVdd[i], dAVdd_all[i], dAVdd_mask[i] = self.generate_dAV_dd_array(
                    l, b, star["DIST"], self.bins, dust_data)

        else:
            for i in range(len(self.stars)):
                sig, err = self.load_residual_from_file(star, star_rv = None, use_MADGICS = use_MADGICS)
                signals[i, :], signal_errs[i, :] = sig[self.window], err[self.window]
                l, b = star["GLON"], star["GLAT"]
                dAVdd[i], dAVdd_all[i], dAVdd_mask[i] = self.generate_dAV_dd_array(
                    l, b, star["DIST"], self.bins, dust_data)

        self.signals = signals
        self.signal_errs = signal_errs
        self.dAVdd = dAVdd
        self.voxel_dAVdd = np.nanmedian(dAVdd_all, axis=0)
        self.voxel_dAVdd_std = np.nanstd(dAVdd_all, axis=0, ddof=1)
        self.dAVdd_mask = dAVdd_mask.astype(bool)


    def get_DIBs_old(self, dust_data, MADGICS=False, alternative_data_processing=None, **kwargs):
        signals = np.zeros((len(self.stars), len(self.wavs_window)))
        signal_errs = np.zeros((len(self.stars), len(self.wavs_window)))
        dAVdd = np.zeros((len(self.stars), len(self.bins) - 1))
        dAVdd_all = np.zeros((len(self.stars), len(self.bins) - 1))
        dAVdd_mask = np.zeros((len(self.stars), len(self.bins) - 1)).astype(bool)

        if alternative_data_processing is not None:
            self.alternative_data_processing = alternative_data_processing
            # needs to take aspcap, medres, apstar, rv as arguments
            for i in range(len(self.stars)):
                star = self.stars[i]
                star_rv = star["VHELIO_AVG"]
                aspcap = fits.open(self.getASPCAP(star))
                apstar = fits.open(self.getapStar(aspcap))
                medres = fits.open(
                    self.get_medres(star["TEFF"], star["LOGG"], star["M_H"])
                )
                sig, err = self.alternative_data_processing(
                    self, aspcap, medres, apstar, star_rv, **kwargs
                )
                signals[i, :], signal_errs[i, :] = sig[self.window], err[self.window]

                l, b = star["GLON"], star["GLAT"]
                dAVdd[i], dAVdd_all[i], dAVdd_mask[i] = self.generate_dAV_dd_array(
                    l, b, star["DIST"], self.bins, dust_data, **kwargs
                )

        else:
            if MADGICS:
                signals_aspcap = np.zeros((len(self.stars), len(self.wavs_window)))
                signal_errs_aspcap = np.zeros((len(self.stars), len(self.wavs_window)))

            for i in range(len(self.stars)):
                star = self.stars[i]
                star_rv = star["VHELIO_AVG"]
                res_hdul = fits.open(get_ca_res(star["FILE"]))
                signals[i, :] = res_hdul[1].data[self.window]
                signal_errs[i, :] = res_hdul[2].data[self.window]
                reprocess_uncertainty = True
                # if reprocess_uncertainty:
                #     signal_errs[i, :] = self.reprocess_errs(res_hdul, star['VHELIO_AVG'])[self.window]
                # reprocess_residual = True
                # if reprocess_residual:
                #     res_repr, err_repr = reprocess(res_hdul, star['VHELIO_AVG'])
                #     signals[i, :] = res_repr[self.window]
                #     signal_errs[i, :] = err_repr[self.window]
                l, b = star["GLON"], star["GLAT"]
                dAVdd[i], dAVdd_all[i], dAVdd_mask[i] = self.generate_dAV_dd_array(
                    l, b, star["DIST"], self.bins, dust_data, **kwargs
                )

                if MADGICS:
                    signals_aspcap[i, :] = np.copy(signals[i, :])
                    signal_errs_aspcap[i, :] = np.copy(signal_errs[i, :])
                    res_hdul_m = fits.open(self.get_madgics_res(star["FILE"]))
                    signals[i, :] = res_hdul_m[1].data[0, 125:][self.window]
        self.signals = signals
        self.signal_errs = signal_errs
        self.dAVdd = dAVdd
        self.voxel_dAVdd = np.nanmedian(dAVdd_all, axis=0)
        self.voxel_dAVdd_std = np.nanstd(dAVdd_all, axis=0, ddof=1)
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

    def model_signals(self, rvelo, dAVdd=None, binsep = None): # CHANGED 
        if dAVdd is None:
            dAVdd = self.dAVdd
        if binsep is None:
            binsep = self.bins[1:]-self.bins[:-1]
        signals = np.zeros((len(self.stars), len(self.wavs_window)))
        peak_wavelength = self.dopplershift(rvelo)
        wavs_grid = np.tile(self.wavs_window, (len(self.bins)-1, 1))
        voxel_DIB_unscaled = np.exp(-(wavs_grid - peak_wavelength[:, np.newaxis])**2 / (2 * sigma0**2))
        amp = self.differentialAmplitude(dAVdd, binsep)

        def single_signal(amp, bindex, voxel_DIB_unscaled = voxel_DIB_unscaled):
            amp[bindex :] = 0 # THIS MIGHT NEED TO BE -1

            voxel_DIB_scaled = -voxel_DIB_unscaled *  amp[:, np.newaxis] 
            summed_DIB = np.sum(voxel_DIB_scaled, axis = 0)
            # continuum = lambda x, m, b : m * (x - lambda0) + b
            # cont = continuum(wavs_window, 0, b)
            return summed_DIB  + 1


        for i in range(len(self.stars)):
            star = self.stars[i]
            dAVdd_star = dAVdd[i, :]
            amp = self.differentialAmplitude(dAVdd_star, 1)

            # ADDED 2025.03.17:
            # signals = np.zeros((len(self.stars), len(self.wavs_window)))

            rvelo_per_star = np.copy(rvelo)

            rvelo_per_star[0] = self.fg_velo[i]

            peak_wavelength = self.dopplershift(rvelo_per_star)
            voxel_DIB_unscaled = np.exp(-(wavs_grid - peak_wavelength[:, np.newaxis])**2 / (2 * sigma0**2))
            ########


            bin_index = self.bin_inds[i]
            # signals[i, :] = single_signal(bin_index)
            signals[i, :] = single_signal(amp, bin_index, voxel_DIB_unscaled = voxel_DIB_unscaled)
        return signals
    

    def foreground_velocities(self):
        l, b = self.stars['GLON'], self.stars["GLAT"]
        fg_velo_star = np.zeros(len(self.stars))
        for i in range(len(self.stars)):
            fg_velo_star[i] = fg_polynomial2d(l[i], b[i])
        return fg_velo_star