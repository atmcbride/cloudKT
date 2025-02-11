from dust_data import DustData
import h5py
import time

directory = '/uufs/chpc.utah.edu/common/home/astro/zasowski/mcbride/data/'

def main():
    t0 = time.time()
    dust = DustData()
    dust.load_map()
    t1 = time.time()

    with h5py.File(directory + 'edenhofer_coords.h5', 'w') as f:
        f.create_dataset('dustmap', data = dust.dustmap)
        f.create_dataset('l', data = dust.l)
        f.create_dataset('b', data = dust.b)
        f.create_dataset('d', data = dust.d)
    
    t2 = time.time()
    with h5py.File(directory + 'edenhofer_coords.h5', 'r') as f:
        dustmap = f['dustmap'][:]
        l = f['l'][:]
        b = f['b'][:]
        d = f['d'][:]
    t3 = time.time()

    print(f"Time to load map and generate coordinates: {t1 - t0:.2f} seconds")
    print(f"Time to load map and coordinates: {t2 - t1:.2f} seconds")

if __name__ == "__main__":
    main()