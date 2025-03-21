### Created 2025-02-06 by atmcb
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u
from spectral_cube import SpectralCube
import h5py

import logging

from utilities import load_module 

logger = logging.getLogger(__name__)

def load_data(parameters):
    """
    Load data from the specified modules and methods in the configuration.
    """
    
    logger.info('Loading star metadata...')
    # star_loader_module = load_module(parameters['LOAD_STARS_MODULE'])
    # star_loader = getattr(star_loader_module, parameters['LOAD_STARS_METHOD'])
    stars_data = load_stars(parameters['LOAD_STARS_PARAMETERS'])

    logger.info('Loading dust data...')
    # dust_loader_module = load_module(parameters['DUST_LOAD_MODULE'])
    # dust_loader = getattr(dust_loader_module, parameters['DUST_LOAD_METHOD'])
    dust_data = load_dust(parameters['LOAD_DUST_PARAMETERS'])
    dust_extent = (dust_data.l.min(), dust_data.l.max(), dust_data.b.min(), dust_data.b.max())

    logger.info('Loading CO emission data...')
    emission_CO_data = load_emission(parameters['LOAD_CO_PARAMETERS'], dust_extent = dust_extent, carrier = 'CO')
    logger.info('Loading HI emission data...')
    emission_HI_data = load_emission(parameters['LOAD_HI_PARAMETERS'], dust_extent = dust_extent, carrier = 'HI')

    logger.info('Data loaded!')
    return stars_data, dust_data, emission_CO_data, emission_HI_data



def load_stars(parameters):
    """
    Load stellar metadata from meta table specified in parameters 
    """ 

    meta_tab = Table(fits.open('dummyMeta.fits')[1].data)
    return meta_tab



def load_dust(parameters):
    """
    load an instance of the dust data object
    """
    dust_data = DustData()
    dust_data.load_map()
    return dust_data



def load_emission(parameters, dust_extent, carrier = 'CO'):
    """
    Load in emission data from the specified modules and methods in the configuration."""
    if carrier == 'HI':
        fname_hi4pi = '/uufs/chpc.utah.edu/common/home/astro/zasowski/catalogs/HI4PI_CAR.fits'
        hdul_hi4pi = fits.open(fname_hi4pi)
        hdul_hi4pi[0].header['CDELT3'] = hdul_hi4pi[0].header['CDELT3'] / 1e3
        hdul_hi4pi[0].header['CRVAL3'] = hdul_hi4pi[0].header['CRVAL3'] / 1e3
        hdul_hi4pi[0].header['CUNIT3'] = 'km/s'
        hdul_hi4pi[0].header['COMMENT'] = 'Converted to km/s 08.23.24 in 240723_EmissionMaps/240922_HIMapsExplore.ipynb'
        cube_hi4pi = SpectralCube(data = hdul_hi4pi[0].data, wcs = WCS(hdul_hi4pi[0].header), header = hdul_hi4pi[0].header)
        cube_CA = cube_hi4pi.subcube(xlo = dust_extent[0]*u.deg, xhi = dust_extent[1]*u.deg, ylo = dust_extent[2]*u.deg, yhi = dust_extent[3]*u.deg).spectral_slab(-15 * u.km/u.s, 10 * u.km/u.s)


    elif carrier == 'CO':
        fname_CO = '/uufs/chpc.utah.edu/common/home/astro/zasowski/catalogs/DHT21_Taurus_interp.fits'
        hdul_CO = fits.open(fname_CO)
        hdul_CO[0].header['CTYPE1'] = 'VRAD'
        hdul_CO[0].header['CUNIT1'] = 'km/s'
        hdul_CO[0].header['COMMENT'] = 'Edited 08.23.24 in 240723_EmissionMaps/240922_HIMapsExplore.ipynb'

        cube_CO = SpectralCube(data = hdul_CO[0].data, wcs = WCS(hdul_CO[0].header), header = hdul_CO[0].header)
        cube_CA = cube_CO.subcube(xlo = dust_extent[0]*u.deg, xhi = dust_extent[1]*u.deg, ylo = dust_extent[2]*u.deg, yhi = dust_extent[3]*u.deg).spectral_slab(-15 * u.km/u.s, 10 * u.km/u.s)
    return cube_CA



class DustData: 
    def __init__(self, **kwargs):
        self.dustmap_grid(**kwargs)
        # self.load_map()
    def dustmap_grid(self, **kwargs):
        self.distance = np.linspace(0, 8, 8)
        self.l0, self.b0 = (163., -8.0)
        self.l_ = np.linspace(self.l0 - 9., self.l0 + 9., 8)
        self.b_ = np.linspace(self.b0 - 9., self.b0 + 9., 8)
        self.l, self.b, self.d = np.meshgrid(self.l_, self.b_, self.distance) 
    def load_map(self, map_fname = '/uufs/astro.utah.edu/common/home/u1371365/DIB_KT_CACloud/edenhofer_out.h5', **kwargs):
        self.dustmap = np.zeros((len(self.l_), len(self.b_), len(self.distance)))
    def intake_map(self, map_array):
        self.dustmap = map_array



def apply_restrictions(tab, restrictions): # THIS WILL NOT PRESENTLY WORK DUE TO THE PRESENCE OF BOTH &s and |s in data_criteria. Think about this later.
    select = np.ones(len(tab), dtype = bool)
    for key in restrictions.keys():
        entry = restrictions[key]
        if entry[-1] == '>':        
            select &= (tab[key] > entry[0])
        elif entry[-1] == '<':
            select &= (tab[key] < entry[0])
        elif entry[-1] == '=':
            select &= (tab[key] == entry[0])
        elif entry[-1] == '!=':
            select &= (tab[key] != entry[0])
    return select
