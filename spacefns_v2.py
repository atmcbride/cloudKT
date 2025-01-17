import numpy as np 
import globalvars 


dustdata_class = globalvars.DustData
dustdata = dustdata_class()


def select_stars(tab, ll, bb, dustdata=dustdata, radius = 1):
    cond = np.sqrt((tab['GLON'] - ll)**2 + (tab['GLAT'] - bb)**2) < radius
    return np.where(cond)[0]

def find_nearest(ll, bb, dustdata=dustdata):#l_sel = \\\, b_sel = b_):
    l_sel, b_sel = (dustdata.l_, dustdata.b_)
    return np.argmin(np.abs(l_sel - ll)), np.argmin(np.abs(b_sel - bb))

def find_nearest_dist(d, dustdata=dustdata):
    distance = dustdata.distance
    return np.argmin(np.abs(distance[:, np.newaxis] - d), axis = 0)

### Added 10.03
def find_radius(ll, bb, count, stars): # = CA_meta
    angdist = np.sqrt((ll - stars['GLON'])**2 + (bb - stars['GLAT'])**2)
    angdist_sort = np.sort(angdist)
    return angdist_sort[count]


def dAV_dd(l0, b0, bin_edges, dustdata=dustdata):
    dustmap = dustdata.dustmap

    l_ind, b_ind = find_nearest(l0, b0, dustdata)
    sightline = np.copy(dustmap[b_ind, l_ind, :]) #needs to be b then l then :

    d_min, d_max = bin_edges

    extinction = sightline[(distance > d_min) & (distance < d_max)]
    return np.sum(extinction )

def dAV_dd_star(l0, b0, bin_edges, distances, dustdata=dustdata):
    distance = dustdata.distance
    dustmap = dustdata.dustmap

    l_ind, b_ind = find_nearest(l0, b0, dustdata)
    d_min, d_max = bin_edges
    sightline = np.copy(dustmap[b_ind, l_ind, :])
    sightline[(distance < d_min) | (distance > d_max)] = 0
    sightline_av = (np.cumsum(sightline)) 
    d_ind = find_nearest_dist(distances)

    return np.nanmedian(sightline_av[d_ind])

def differentialAmplitude(dAv_dd, dd = 1):
    if type(dd) == int:
        return  0.024 * dAv_dd * dd  # 1/(np.sqrt(2 * np.pi) * sigma0) * 102e-3 * dAv_dd * dd
    elif dd.shape == dAv_dd.shape:
        return 0.024 * dAv_dd * dd 
    else:
        return 0.024 * dAv_dd * dd[np.newaxis, :]

        #  return 0.024 * dAv_dd * dd[:, np.newaxis]

def dAV_dd_array(l, b, bins, star_dist, dustdata=dustdata):#= dmap_distance):
    distance = dustdata.distance
    dustmap = dustdata.dustmap
    l_ind, b_ind = find_nearest(l, b, dustdata)
    dustmap_sightline = np.copy(dustmap[b_ind, l_ind, :]) 
    dAVdd = np.zeros(len(bins)-1)
    dAVdd_all = np.zeros(len(bins)-1)
    dAVdd_mask = np.zeros(len(bins-1))
    for i in range(len(dAVdd)):
        bin_min, bin_max = bins[i], bins[i+1]
        if bin_min < star_dist:
            dist_max = bin_max
            if bin_max >= star_dist:
                dist_max = star_dist
        else:
            dist_max = -np.inf
            
        dAVdd[i] = np.sum(dustmap_sightline[(distance > bin_min) & (distance <= dist_max)])

        dAVdd_all[i] = np.sum(dustmap_sightline[(distance > bin_min) & (distance <= bin_max)])

    dAVdd_mask = (dAVdd == 0).astype(bool)
    return dAVdd, dAVdd_all, dAVdd_mask