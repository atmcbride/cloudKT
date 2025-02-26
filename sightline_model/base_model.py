### created 2025-02.08 by a. mcbride
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.table import Table

lambda0 = 15272.42
sigma0 = 1.15


class BaseModel:
    """
    Base class for models. Not for use on its own.
    """

    def __init__(self, *args, **kwargs):
        self.model_name = "BaseModel"
        self.lambda0 = 15272.42
        self.sigma0 = 1.15
        self.wavs = self.get_wavs()
        self.window = (self.wavs > lambda0 - 10) & (self.wavs < lambda0 + 10)
        self.wavs_window = self.wavs[self.window]
        self.medres_meta = None

    @staticmethod
    def select_near_point(tab, l, b, radius=1):
        """
        Selects stars within a specified radius of a position in l and b
        """
        cond = np.sqrt((tab["GLON"] - l) ** 2 + (tab["GLAT"] - b) ** 2) < radius
        return np.where(cond)[0]

    @staticmethod
    def differentialAmplitude(dAV_dd, dd=1.0):
        """
        Converts a dAV_dd value to a dAMP(DIB)_dd value.
        """
        if np.isscalar(dd):
            return 0.024 * dAV_dd * dd
        elif dd.shape == dAV_dd.shape:
            return 0.024 * dAV_dd * dd
        else:
            return 0.024 * dAV_dd * dd[np.newaxis, :]

    @staticmethod
    def generate_dAV_dd_array(l, b, star_dist, distance_bins, dustdata):
        """
        Generates a dAV_dd array for a given stellar position, distance bin boundaries, and dust map.
        """
        distance = dustdata.distance
        dustmap = dustdata.dustmap
        l_ind, b_ind = dustdata.find_nearest_angular(l, b)
        dust_column = np.copy(dustmap[b_ind, l_ind, :])
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

    @staticmethod
    def get_wavs(hdulist=None, rv=0):
        """
        Utility function for setting a wavelength array from a header or default values.
        """
        if hdulist is None:
            CRVAL1 = 4.179
            CDELT1 = 6e-06
            LEN = 8575
        else:
            header = hdulist[1].header
            CRVAL1 = header["CRVAL1"]
            CDELT1 = header["CDELT1"]
            LEN = header["NAXIS1"]

        wavs = np.power(10, CRVAL1 + CDELT1 * np.arange(LEN))
        wavs = wavs * (
            1 + rv / 3e5
        )  # allows for shifting to observed frame from rest frame
        return wavs

    @staticmethod  # ed8062ec98407386ebfbc66a7c615eb56ffcd1ea  729d96ac711661aad43b005c138efa4c096d20b1
    def resample_interp(self, data, rv, hdu_sel=None):
        """
        Utility function for re-sampling spectra to a new RV frame"""
        wavs_rv = self.get_wavs(rv=rv, hdulist=hdu_sel)
        interp = interp1d(wavs_rv, data, kind="slinear", bounds_error=False)
        data_interp = interp(self.get_wavs(rv=0, hdulist=hdu_sel))
        return data_interp

    @staticmethod
    def dopplershift(v, lambda0=15272.42):
        """
        Takes a velocity in km/s, and a wavelength (default lambda0 of the 15272A DIB)
        returns the doppler-shifted wavelength in Angstroms
        """
        return lambda0 * (1 + v / 3e5)

    @staticmethod 
    def populate_from_file(*args, fname = None):
        tab = Table(fits.open(fname)[1].data)
        return tab

    @staticmethod
    def getASPCAP(row):
        specdir = '/uufs/chpc.utah.edu/common/home/sdss/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/{TELESCOPE}/{FIELD}/'
        specname = 'aspcapStar-dr17-{SOURCEID}.fits'
        telescope = np.array(row['TELESCOPE'], dtype = str)
        field = np.array(row['FIELD'], dtype = str)
        sourceid = np.array(row['APOGEE_ID'], dtype = str)
        path = (specdir + specname).format(TELESCOPE = telescope, FIELD = field, SOURCEID = sourceid)
        return path

    @staticmethod
    def getapStar(hdulist):
        specdir = '/uufs/chpc.utah.edu/common/home/sdss/dr17/apogee/spectro/redux/dr17/stars/{TELESCOPE}/{FIELD}/'
        telescope = str(hdulist[4].data['TELESCOPE'][0])
        field = str(hdulist[4].data['FIELD'][0])
        fname = str(hdulist[4].data['FILE'][0])
        path = ((specdir + fname).format(TELESCOPE = telescope, FIELD = field))
        return path
    
    @staticmethod
    def get_medres(teff, logg, m_h, medres_dir = '/uufs/astro.utah.edu/common/home/u1371365/StellarResidualsSpring2022/Residuals/'):
        meta = Table(fits.open('/uufs/astro.utah.edu/common/home/u1371365/StellarResidualsSpring2022/Residuals/meta.fits')[1].data)
        rowselect = np.where(np.logical_and.reduce(
                        [teff >= meta['TEFF_MIN'], teff < meta['TEFF_MAX'], 
                        logg >= meta['LOGG_MIN'], logg < meta['LOGG_MAX'],
                    m_h >= meta['M_H_MIN'], m_h < meta['M_H_MAX']]))[0]
        
        row = meta[rowselect]
        filename = row['FNAME'].item()
        return medres_dir + filename