import numpy as np
import matplotlib.pyplot as plt

def selectOnEW(tab, vector, select_first = None, norm_vals = None, plot = False):
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