import numpy as np
import h5py

class DustData: 
    """
    Class for handling dustmap data for the CA cloud
    Added 2025-02-08: this class now contains functions for finding the nearest dustmap coordinate 
    index for a specified (l, b) or distance
    """
    def __init__(self, **kwargs):
        self.dustmap_grid(**kwargs)

    def dustmap_grid(self, **kwargs):
        self.distance = np.linspace(0, 800, 800)
        self.l0, self.b0 = (163., -8.0)
        self.l_1d = np.linspace(self.l0 - 9., self.l0 + 9., 800)
        self.b_1d = np.linspace(self.b0 - 9., self.b0 + 9., 800)
        self.l, self.b, self.d = np.meshgrid(self.l_1d, self.b_1d, self.distance) 
    def load_map(self, map_fname = '/uufs/astro.utah.edu/common/home/u1371365/DIB_KT_CACloud/edenhofer_out.h5', **kwargs):
        with h5py.File(map_fname, 'r') as f:
            edenhofer = np.array(f['data'])
        self.dustmap = edenhofer
    def intake_map(self, map_array): # probably not needed
        self.dustmap = map_array
    def find_nearest_angular(self, ll, bb):
        l_ind, b_ind = (np.argmin(np.abs(self.l_1d - ll)), np.argmin(np.abs(self.b_1d - bb)))
        return l_ind, b_ind
    def find_nearest_distance(self, d):
        return np.argmin(np.abs(self.distance[:, np.newaxis] - d), axis = 0)