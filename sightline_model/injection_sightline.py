### created 2025-`08 by a. mcbride

import numpy as np
from astropy.io import fits
from .base_model import BaseModel
from astropy.table import Table

from filehandling import get_medres, get_ca_res, get_madgics_res, getapStar, getASPCAP
from scipy.signal import correlate, correlation_lags

lambda0 = 15272.42
sigma0 = 1.15

def get_highLat(tabfile = '/uufs/chpc.utah.edu/common/home/sdss/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits'):
    allStar = Table.read(tabfile, hdu = 1)
    starFlag = allStar['ASPCAPFLAG']
    starMask = np.invert((np.logical_and(starFlag, 2**23)==True))
    allStar = allStar[starMask]
    highLat = allStar[(np.abs(allStar['GLAT']) > 15) & (1000/allStar['GAIAEDR3_PARALLAX'] < 1.5e3) & (allStar['SFD_EBV'] < 0.2)]

    data_criteria_hl= (((highLat['SNR'] > 80) & (highLat['TEFF'] > 5000)) | (highLat['SNR'] > 150)) & (highLat['ASPCAP_CHI2'] > 1) & (highLat['ASPCAP_CHI2'] < 5)

    highLat = highLat[data_criteria_hl]
    return highLat

highLat = get_highLat()

class InjectionSightline(BaseModel):
    def __init__(
        self, stars, dust_data, rvelo_profile, dust_profile, *args, dust_profile_err = None, bins=None, emission= None, coordinates = None, star_selection_kwargs = None, injectRealContinuum = True, **kwargs
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
        self.get_DIBs_skeleton(dust_data, **kwargs)


        self.makeSyntheticDIBs(rvelo_profile, dust_data, dAVdd_profile = dust_profile, dAVdd_profile_err = dust_profile_err, injectRealContinuum = injectRealContinuum)

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

    def makeSyntheticDIBs(self, rvelo, dust_data, dAVdd_profile=None, dAVdd_profile_err = None, emission = None, injectRealContinuum=True):
        continua = np.zeros((len(self.stars), len(self.wavs_window)))
        continua_errs = np.zeros((len(self.stars), len(self.wavs_window)))
        dustcolumn = np.zeros((len(self.stars), dust_data.dustmap.shape[-1]))
        # signals = np.zeros((len(self.stars), len(self.wavs_window)))
        # signal_errs = np.zeros((len(self.stars), len(self.wavs_window)))

        dAVdd = np.zeros((len(self.stars), len(self.bins) - 1))
        dAVdd_all = np.zeros((len(self.stars), len(self.bins) - 1))
        dAVdd_mask = np.zeros((len(self.stars), len(self.bins) - 1)).astype(bool)

        for i in range(len(self.stars)):
            star = self.stars[i]

            dAVdd[i], dAVdd_all[i], dAVdd_mask[i] = self.generate_dAV_dd_array_synthetic(
            star["DIST"], self.bins, dust_data.distance, dAVdd_profile)
                
            rv_star = star["VHELIO_AVG"]
            if injectRealContinuum:
                continuum, continuum_uncertainty = self.getAnalogContinuum(
                    star, rv_star
                )
                continua[i, :] = continuum[self.window]
                continua_errs[i, :] = continuum_uncertainty[self.window]
            else:
                continuum = 1 + np.random.normal(
                    scale=1 / star["SNR"], size=np.sum(self.window)
                )
                continuum_uncertainty = np.ones(np.sum(self.window)) * 1 / star["SNR"]
                continua[i, :] = continuum
                continua_errs[i, :] = continuum_uncertainty

            l_i, b_i = dust_data.find_nearest_angular(star["GLON"], star["GLAT"])
            d_i = dust_data.find_nearest_distance(star["DIST"]).item()
            if dAVdd is not None:
                dcol = np.copy(dAVdd)
                dcol[d_i:] = 0
            else:
                dcol = dust_data.dustmap[b_i, l_i, :]
                dcol[d_i:] = 0
            # dustcolumn[i, :] = dcol

        raw_DIB = self.integrateMockDIB(rvelo, dAVdd_profile)
        signals = raw_DIB - 1 + continua

        self.signals = signals
        self.signal_errs = continua_errs
        self.raw_DIB = raw_DIB
        self.continuum = continua
        self.dustcolumn = dustcolumn

        self.dAVdd = dAVdd
        self.voxel_dAVdd = np.nanmedian(dAVdd_all, axis=0)
        # self.voxel_dAVdd_std = np.nanstd(dAVdd_all, axis=0, ddof=1)
        # self.voxel_dAVdd_std = np.sqrt(np.sum(dAVdd_profile_err**2)) * np.ones(self.voxel_dAVdd_std.shape)
        # self.voxel_dAVdd_std = np.nanmedian(self.voxel_dAVdd_std) * np.ones(self.voxel_dAVdd.shape)
        # self.dAVdd_mask = dAVdd_mask.astype(bool)

    def get_DIBs_skeleton(
        self, dust_data, dAVdd_profile = None, MADGICS=False, alternative_data_processing=None, **kwargs
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


        self.dAVdd = dAVdd
        self.voxel_dAVdd = np.nanmedian(dAVdd_all, axis=0)
        self.voxel_dAVdd_std = np.nanstd(dAVdd_all, axis=0, ddof=1)
        self.dAVdd_mask = dAVdd_mask.astype(bool)

        self.star_signals = signals
        self.star_signal_errs = signal_errs

    def getAnalogContinuum(self, star, rv_star, reference_stars=highLat):
        SNRdiff = np.abs(reference_stars["SNR"] - star["SNR"])
        TEFFdiff = np.abs(reference_stars["TEFF"] - star["TEFF"])
        LOGGdiff = np.abs(reference_stars["LOGG"] - star["LOGG"])
        M_Hdiff = np.abs(reference_stars["M_H"] - star["M_H"])
        starAnalogs = np.logical_and.reduce(
            [(SNRdiff < 30), (TEFFdiff < 250), (LOGGdiff < 0.2), (M_Hdiff < 0.1)]
        )
        analog_i = np.argmin(reference_stars[starAnalogs]["SFD_EBV"])
        analog = reference_stars[analog_i]
        medres = fits.open(get_medres(analog["TEFF"], analog["LOGG"], analog["M_H"]))
        aspcap = fits.open(getASPCAP(analog))
        apstar = fits.open(getapStar(aspcap))
        if hasattr(self, "alternative_data_processing"):
            res, res_err = self.alternative_data_processing(star)
        else: 
            print("Issue loading residuals from alternative_data_processing (fix this!)")
        return res, res_err


    def integrateMockDIB(self, rvelo, dAVdd):


        signals = np.zeros((len(self.stars), len(self.wavs_window)))
        peak_wavelength = self.dopplershift(rvelo)
        wavs_grid = np.tile(self.wavs_window, (len(rvelo), 1))

        voxel_DIB_unscaled = np.exp(
            -((wavs_grid - peak_wavelength[:, np.newaxis]) ** 2) / (2 * sigma0**2)
        )
        amp = self.differentialAmplitude(dAVdd, 1)

        def single_signal(amp, bindex):
            # amp[bindex :] = 0 # THIS MIGHT NEED TO BE -1

            voxel_DIB_scaled = -voxel_DIB_unscaled * amp[:, np.newaxis]
            summed_DIB = np.sum(voxel_DIB_scaled, axis=0)
            # continuum = lambda x, m, b : m * (x - lambda0) + b
            # cont = continuum(self.wavs_window, 0, b)
            return summed_DIB + 1

        for i in range(len(self.stars)):
            star = self.stars[i]
            # dAVdd_star = dAVdd[i, :]
            # amp = self.differentialAmplitude(dAVdd_star, 1)
            amp = self.differentialAmplitude(dAVdd, 1)


            bin_index = self.bin_inds[i]
            # signals[i, :] = single_signal(bin_index)
            signals[i, :] = single_signal(amp, 0)  # bin_index)
        return signals

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

            
    @staticmethod
    def get_dust_profile(dust, emission, threshold = 0.03, ref_point = (167.4, -8.3), dust_profile_type = None, av_offset = 0, **kwargs):
        if dust_profile_type == "average prior":
            # replicates dust extraction from the prior
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
            avg_dust_profile = avg_dust_profile + av_offset

            return avg_dust_profile, std_dust_profile

    @staticmethod
    def get_velo_profile(*args, velo_profile_type = "linear", v_noise_scale = 0.0, **kwargs):
        if velo_profile_type == "linear":
            dust, vnear, vfar, xnear, xfar = args
            dist = dust.distance
            velo_profile = np.zeros(dist.shape)
            slope = (vfar - vnear) / (xfar - xnear)
            intercept = - (xnear + xfar ) / 2  * slope 
            velo_profile[(dist > xnear)  & (dist <= xfar)] = slope * dist[(dist > xnear)  & (dist <= xfar)] + intercept
            velo_profile[(dist > xnear)  & (dist <= xfar)] += np.random.normal(scale = v_noise_scale, 
                        size = np.sum((dist > xnear)  & (dist <= xfar)))
        return velo_profile

    @staticmethod
    def generate_dAV_dd_array_synthetic(star_dist, distance_bins, distance, dust_profile):
        """
        Generates a dAV_dd array for a given stellar position, distance bin boundaries, and dust map.
        """
        dust_column = np.copy(dust_profile)
        n_bins = len(distance_bins) - 1
        dAVdd = np.zeros(n_bins)
        dAVdd_all = np.zeros(n_bins)
        dAVdd_mask = np.zeros(n_bins)
        for i in range(len(dAVdd)):
            bin_min, bin_max = distance_bins[i], distance_bins[i + 1]
            if bin_min < star_dist:
                dist_max = bin_max
                if bin_max >= star_dist:
                    dist_max = star_dist
            else:
                dist_max = -np.inf
            dAVdd[i] = np.sum(
                dust_column[(distance > bin_min) & (distance <= dist_max)]
            )
            dAVdd_all[i] = np.sum(
                dust_column[(distance > bin_min) & (distance <= bin_max)]
            )

        dAVdd_mask = (dAVdd == 0).astype(bool)
        return dAVdd, dAVdd_all, dAVdd_mask