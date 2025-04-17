### Created 2025-02-08 by a. mcbride
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.io import fits
import logging

from .base_model import BaseModel
from filehandling import get_medres, get_ca_res, get_madgics_res, getapStar, getASPCAP


logger = logging.getLogger(__name__)


class Sightline(BaseModel):
    def __init__(
        self, stars, dust_data,  *args, bins=None, coordinates = None, star_selection_kwargs = None,  **kwargs
    ):
        super().__init__(self, stars, **kwargs)
        # if star_selection is None:
        if self.select_stars is None:
            l, b = coordinates
            self.stars = stars[self.select_near_point(stars, l, b, **kwargs)]
        else:
            stars = self.select_stars(stars, **star_selection_kwargs)
            self.stars = stars
    

        dist = stars["DIST"]
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

    def make_bins(self, binsep=10, dmin=0):
        ### Assigns stars to distance bins if bins are not already supplied.
        dist = self.stars["DIST"]
        bins = np.sort(np.insert(np.delete(dist, np.where(dist <= dmin)[0]), 0, dmin))

        i = 0
        while i >= 0:
            try:
                next_bin = np.min(bins[bins > bins[i]])
            except:
                print("broke:")
                print(bins[bins > bins[i]])
                print(len(self.stars))

            bins[i + 1] = np.max([next_bin, bins[i] + binsep]) + 0.01
            if bins[i + 1] >= np.max(dist):
                bins = bins[: i + 2]
                i = -np.inf
            i = i + 1

        self.bins = bins

    def get_DIBs(
        self, dust_data, MADGICS=False, alternative_data_processing=None, **kwargs
    ):
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
                    l, b, star["DIST"], self.bins, dust_data, 
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

    def model_signals(self, rvelo, dAVdd=None, binsep=None):
        if dAVdd is None:
            dAVdd = self.dAVdd
        if binsep is None:
            binsep = self.bins[1:] - self.bins[:-1]
        signals = np.zeros((len(self.stars), len(self.wavs_window)))
        peak_wavelength = self.dopplershift(rvelo)
        wavs_grid = np.tile(self.wavs_window, (len(self.bins) - 1, 1))
        voxel_DIB_unscaled = np.exp(
            -((wavs_grid - peak_wavelength[:, np.newaxis]) ** 2)
            / (2 * self.sigma0**2)
        )
        amp = self.differentialAmplitude(dAVdd, binsep)

        def single_signal(amp, bindex):
            amp[bindex:] = 0  # THIS MIGHT NEED TO BE -1

            voxel_DIB_scaled = -voxel_DIB_unscaled * amp[:, np.newaxis]
            summed_DIB = np.sum(voxel_DIB_scaled, axis=0)
            return summed_DIB + 1

        for i in range(len(self.stars)):
            star = self.stars[i]
            dAVdd_star = dAVdd[i, :]
            amp = self.differentialAmplitude(dAVdd_star, 1)

            bin_index = self.bin_inds[i]
            signals[i, :] = single_signal(amp, bin_index)
        return signals

    