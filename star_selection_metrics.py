import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import correlate, correlation_lags

def select_on_EW(tab, vector = (0.05, 0.05), select_first = None, norm_vals = None, plot = False, **kwargs):
    """Selects stars based on EW"""
    ew = tab['DIB_EQW']
    dist = tab['DIST']
    if norm_vals is not None:
        dist_norm = (dist - norm_vals[0]) / (norm_vals[1] - norm_vals[0])
        ew_norm = (ew - norm_vals[2]) / (norm_vals[3] - norm_vals[2])
    else:
        distperc5, distperc95 = np.percentile(dist, [5, 95])
        dist_norm = (dist - distperc5) / (distperc95 - distperc5)
        ewperc5, ewperc95 = np.percentile(ew, [5, 95])
        ew_norm = (ew - ewperc5) / (ewperc95 - ewperc5)

    if plot:
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 5))
        ax = axs[0]
        ax.scatter(dist, ew)
        ax.set_xlabel('Distance (kpc)')
        ax.set_ylabel('EW ($\AA$)')

        ax = axs[1]
        ax.scatter(dist_norm, ew_norm)
        ax.set_xlabel('Normalized Distance')
        ax.set_ylabel('Normalized EW')
        plt.show()

    if select_first is None:
        select_first = np.random.choice(len(tab), size = 1)[0]

    def next_star(star_idx, vector, direction = +1):
        dist_norm_star = dist_norm[star_idx]
        ew_norm_star = ew_norm[star_idx]
        diff_dist = direction * (dist_norm - dist_norm_star - direction * vector[0])
        diff_ew = direction * (ew_norm - ew_norm_star - direction * vector[1])
        diff_dist[diff_dist <= 0.0] = np.inf
        diff_ew[diff_ew <= 0.0] = np.inf
        next_star_idx = np.argmin(np.sqrt(diff_dist**2 + diff_ew**2))
        if diff_dist[next_star_idx] == np.inf:
            return np.array([])
        return np.array([next_star_idx])
    
    select_above, select_below = True, True
    selection = np.array([select_first])
    while select_above or select_below:
        if select_above:
            next_star_above = next_star(selection[-1], vector, direction = +1)
            selection = np.concatenate([selection, next_star_above]).astype(int)
            if len(next_star_above) == 0:
                select_above = False
        
        if select_below:
            next_star_below = next_star(selection[0], vector, direction = -1)
            selection = np.concatenate([next_star_below, selection]).astype(int)
            if len(next_star_below) == 0:
                select_below = False

    return selection 


def select_on_emission(tab, emission = None, threshold = 0.03, ref_point = (167.4, -8.3), **kwargs):
    b_em, l_em = emission.world[0, :, :][1:]
    b_em, l_em = b_em[:, 0], l_em[0, :]
    em_i, em_j = np.argmin(np.abs(l_em.value - ref_point[0])), np.argmin(np.abs(b_em.value - ref_point[1]))
    em_star_indices = np.array([[ np.argmin((tab['GLAT'][i] - b_em.value)**2), np.argmin((tab['GLON'][i] - l_em.value)**2)] for i in range(len(tab))])

    reference_point = emission.unmasked_data[:, em_j, em_i]
    corr_lags = correlation_lags(emission.shape[0], emission.shape[0])
    zpoint = corr_lags == 0
    correlation_image = np.zeros((emission.shape[1], emission.shape[2]))

    for i in range(emission.shape[1]):
        for j in range(emission.shape[2]):
            correlation_image[i, j] = correlate(emission.unmasked_data[:, i, j] / np.nansum(np.abs(emission.unmasked_data[:, i, j])), 
                                                reference_point / np.nansum(np.abs(reference_point)))[zpoint]
    
    stars_CO_correlation = correlation_image[em_star_indices[:, 0], em_star_indices[:, 1]]

    star_selection = stars_CO_correlation > threshold
    return star_selection

def select_stars(tab,reset_position = True, **kwargs):
    tab = tab[select_on_emission(tab,  **kwargs)]
    tab = tab[select_on_EW(tab, **kwargs)]

    if reset_position:
        tab['GLON_TRUE'] = np.copy(tab['GLON'])
        tab['GLAT_TRUE'] = np.copy(tab['GLAT'])
        tab['GLON'] = 167.4 + np.random.normal(scale = 0.4, size = len(tab))
        tab['GLAT'] = -8.3 + np.random.normal(scale = 0.4, size = (len(tab)))

    return tab

