# created 2025.03.27 by amcbride
import numpy as np 
import matplotlib.pyplot as plt

def plot_velo(chain, sl , min_walker = None, plot_objs = None, color = None, plot_lines = False, bestprob = False, lnprob = None):
    samples = chain.swapaxes(0,1)[-100:, :, :].reshape(-1, chain.shape[-1])
    if plot_objs == None:
        fig, ax = plt.subplots(figsize = (8,6))
    else:
        fig, ax = plot_objs
    ndim = len(sl.voxel_dAVdd)

    walker_max = chain.shape[0]

    if min_walker is None:
        min_walker = -100
    # else:
    min_walker_val = walker_max - min_walker

    # samples = chain[min_walker_val:, :, :].reshape((-1, chain.shape[-1]))

    vel_samples = samples[:, :sl.ndim]
    avg_av = np.nansum(np.median(sl.dAVdd, axis = 0))

    medians = np.nanmedian(samples[:, :], axis = 0)
    if bestprob:
        lp = lnprob
        lp[:, :-100] = -np.infty
        w_ind, stp_ind = np.unravel_index(np.argmax(lp), lp.shape)

        medians = chain[w_ind, stp_ind, :]

    stdevs = np.nanstd(samples[min_walker_val:, :], ddof = 1, axis = 0)

    med_velo = medians[:ndim]
    std_velo = stdevs[:ndim]


    med_dAV_dd = medians[ndim:]
    med_dAV_dd = stdevs[ndim:]

    perc16, perc50,  perc84 = (np.percentile(samples[min_walker_val:, :], 16, axis = 0), 
                               np.percentile(samples[min_walker_val:, :], 50, axis = 0),
                               np.percentile(samples[min_walker_val:, :], 84, axis = 0) )
    velo16, velo50, velo84 = (perc16[:ndim], perc50[:ndim], perc84[:ndim])

   
    if color == None:
        color_choice = 'k'
    else:
        color_choice = color
    

    colorlist = ['grey', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    bin_pos = np.arange(len(sl.bins[:]))

    bin_pos[0] 
    pos = (bin_pos[1:] + bin_pos[:-1])/2
    
    w = (bin_pos[1:] - bin_pos[:-1])
    vparts=ax.violinplot(vel_samples, pos, widths = w, showmeans=False, showextrema=False, showmedians=True,)
    for vindx, part in enumerate(vparts['bodies']):
        part.set_facecolor( 'C'+str(vindx))

        
    plot_guides = True
    if plot_guides:
        for pos in bin_pos:
            ax.plot((pos, pos), (-10, 20), color = 'k', linestyle = 'dotted')

    # ax.set_xlim(axmin, 600)s
    ax.set_xlabel('Distance (pc)')
    ax.set_ylabel('Radial Velocity (km/s)')

    dist_xx = (sl.bins[1:] + sl.bins[:-1] ) /2
    # med_velo

    return fig, ax, dist_xx, med_velo, std_velo