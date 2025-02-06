import numpy as np
import h5py
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.table import Table
from astropy.io import fits
        
class DustData: 
    def __init__(self, **kwargs):
        self.dustmap_grid(**kwargs)
        # self.load_map()

    def dustmap_grid(self, **kwargs):
        self.distance = np.linspace(0, 800, 800)
        self.l0, self.b0 = (163., -8.0)
        self.l_ = np.linspace(self.l0 - 9., self.l0 + 9., 800)
        self.b_ = np.linspace(self.b0 - 9., self.b0 + 9., 800)
        self.l, self.b, self.d = np.meshgrid(self.l_, self.b_, self.distance) 
    def load_map(self, map_fname = '/uufs/astro.utah.edu/common/home/u1371365/DIB_KT_CACloud/edenhofer_out.h5', **kwargs):
        with h5py.File(map_fname, 'r') as f:
            edenhofer = np.array(f['data'])
        self.dustmap = edenhofer
    def intake_map(self, map_array):
        self.dustmap = map_array

def clean_star_data(parameters):
    star_directory = parameters['RESIDUALS_PATH']
    meta_fname = star_directory + parameters['META_FILE']
    meta_tab = Table(fits.open(meta_fname)[1].data)
    print('N input stars', len(meta_tab))
    # CA_meta  = Table(fits.open('../240723_EmissionMaps/CA_meta_LSR_uncerts.fits')[1].data)

    # added 02.01
    data_criteria = (((meta_tab['SNR'] > 80) & (meta_tab['TEFF'] > 5000)) | (meta_tab['SNR'] > 150)) & (meta_tab['ASPCAP_CHI2_1'] > 1) & (meta_tab['ASPCAP_CHI2_1'] < 5)
    meta_tab = meta_tab[data_criteria]

    print('N stars after quality cuts', len(meta_tab))
    return meta_tab

def assign_stars(parameters):
    return