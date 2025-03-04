import numpy as np 
import matplotlib.pyplot as plt

def plot_velo_dist(chain, sl , min_walker = None, plot_objs = None, color = None, plot_lines = False, plot_box = False, plot_violin = True, bestprob = False, lnprob = None):
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
    

    if plot_box:
        # ax.hlines(med_velo, sl.bins[:-1], sl.bins[1:], color = color_choice, linestyle = 'dashed', linewidth = 0.5)
        # ax.hlines(velo50, sl.bins[:-1], sl.bins[1:], color = color_choice)
        ax.hlines(velo50, sl.bins[:-1], sl.bins[1:], color = 'k')


        for j in range(len(sl.bins)-1):
            # ax.fill_between([sl.bins[i], sl.bins[i +1]], med_velo[i]+std_velo[i], med_velo[i]-std_velo[i], 
            #                 alpha = 0.3, color = color_choice, hatch = '/')
            ax.fill_between([sl.bins[j], sl.bins[j +1]], velo84[j], velo16[j], 
                    alpha = 0.3, color = 'C{}'.format(j))

    axmin = 350
    if plot_violin:
        colorlist = ['grey', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
        bin_pos = sl.bins[:]
        bin_pos[0] = axmin
        pos = (bin_pos[1:] + bin_pos[:-1])/2
        
        w = (bin_pos[1:] - bin_pos[:-1])
        vparts=ax.violinplot(vel_samples, pos, widths = w, showmeans=False, showextrema=False, showmedians=True, side = 'both')
        for vindx, part in enumerate(vparts['bodies']):
            part.set_facecolor( 'C'+str(vindx))

            
        plot_guides = True
        if plot_guides:
            for pos in bin_pos:
                ax.plot((pos, pos), (-10, 20), color = 'k', linestyle = 'dotted')
                # axs[1].plot((pos, pos), (0, 1), color = 'k', linestyle = 'dotted')
    
    plot_violin_half = False
    if plot_violin_half:
        colorlist = ['grey', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
        bin_pos = sl.bins[:]
        bin_pos[0] = axmin
        pos =  bin_pos[1:]
        
        w = np.min(2* (bin_pos[1:] - bin_pos[:-1]))
        vparts=ax.violinplot(vel_samples, pos, widths = w, showmeans=False, showextrema=False, showmedians=True, side = 'low')
        for vindx, part in enumerate(vparts['bodies']):
            part.set_facecolor(colorlist[vindx])


    # else:  
    #     # ax.errorbar((sl.bins[1:] + sl.bins[:-1] ) /2 , med_velo, yerr = std_velo, fmt = '.', color = color_choice, capsize = 5)
    #     # ax.scatter((sl.bins[1:] + sl.bins[:-1] ) /2 , med_velo, c = color_choice)
    #     ax.errorbar((sl.bins[1:] ) , med_velo, yerr = std_velo, fmt = '.', color = color_choice, capsize = 5)
    #     ax.scatter((sl.bins[1:] ) , med_velo, c = color_choice)
    #     if plot_lines:
    #         ax.hlines(med_velo, sl.bins[:-1], sl.bins[1:], color = color_choice, linestyle = 'solid', linewidth = .5)



    # ax.errorbar((sl.bins[1:]),med_velo, xerr = (sl.bins[1:] - sl.bins[:-1], np.zeros(med_velo.shape)), yerr = std_velo, fmt = '.' )
    ax.set_xlim(axmin, 600)
    ax.set_xlabel('Distance (pc)')
    ax.set_ylabel('Radial Velocity (km/s)')

    dist_xx = (sl.bins[1:] + sl.bins[:-1] ) /2
    # med_velo

    return fig, ax, dist_xx, med_velo, std_velo