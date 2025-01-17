import numpy as np
import h5py
from astropy.coordinates import SkyCoord
import astropy.units as u

l0, b0 = (163., -8.0)
lambda0 = 15272.42
sigma0 = 1.15

# global global_var
# global_var = 'hi'

# global distance 
# distance = np.linspace(0, 800, 800)

# global l_
# l_ = np.linspace(l0 - 9., l0 + 9., 800)
# global b_
# b_ = np.linspace(b0 - 9., b0 + 9., 800)

# global l, b, d
# l, b, d = np.meshgrid(l_, b_, distance) 

# global coords
# coords = SkyCoord(l*u.deg, b*u.deg,
#                   distance=distance*u.pc, frame='galactic')

# global dustmap 
# import h5py
# with h5py.File('/uufs/astro.utah.edu/common/home/u1371365/DIB_KT_CACloud/edenhofer_out.h5', 'r') as f:
#     edenhofer = np.array(f['data'])
# dustmap = edenhofer

# def update_global(var_name, new_val):
#     globals()[var_name] = new_val

# class DustData: 
#     def __init__(self, **kwargs):
#         self.dustmap_grid(**kwargs)
#         # self.load_map()

#     def dustmap_grid(self, ):
#         self.distance = np.linspace(0, 800, 800)
#         self.l0, self.b0 = (163., -8.0)
#         self.l_ = np.linspace(self.l0 - 9., self.l0 + 9., 800)
#         self.b_ = np.linspace(self.b0 - 9., self.b0 + 9., 800)
#         self.l, self.b, self.d = np.meshgrid(self.l_, self.b_, self.distance) 
#         self.coords = SkyCoord(self.l*u.deg, self.b*u.deg,
#                   distance=self.distance*u.pc, frame='galactic')
#     def load_map(self, map_fname = '/uufs/astro.utah.edu/common/home/u1371365/DIB_KT_CACloud/edenhofer_out.h5', **kwargs):
#         with h5py.File(map_fname, 'r') as f:
#             edenhofer = np.array(f['data'])
#         self.dustmap = edenhofer
#     def intake_map(self, map_array):
#         self.dustmap = map_array
        
class DustData: 
    def __init__(self, **kwargs):
        self.dustmap_grid(**kwargs)
        # self.load_map()

    def dustmap_grid(self, ):
        self.distance = np.linspace(0, 800, 800)
        self.l0, self.b0 = (163., -8.0)
        self.l_ = np.linspace(self.l0 - 9., self.l0 + 9., 800)
        self.b_ = np.linspace(self.b0 - 9., self.b0 + 9., 800)
        self.l, self.b, self.d = np.meshgrid(self.l_, self.b_, self.distance) 
#         self.coords = SkyCoord(self.l*u.deg, self.b*u.deg,
#                   distance=self.distance*u.pc, frame='galactic')
    def load_map(self, map_fname = '/uufs/astro.utah.edu/common/home/u1371365/DIB_KT_CACloud/edenhofer_out.h5', **kwargs):
        with h5py.File(map_fname, 'r') as f:
            edenhofer = np.array(f['data'])
        self.dustmap = edenhofer
    def intake_map(self, map_array):
        self.dustmap = map_array