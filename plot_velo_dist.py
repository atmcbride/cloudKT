import numpy as np 
import matplotlib.pyplot as plt
import astropy.coordinates as coords
import astropy.units as u
from scipy.signal import correlate, correlation_lags
from mcmc_functions import Logprior_Average_Extinction
import json

def sample_from_chain(chain, burnin = 200, thin = 20):
    return chain[burnin::thin, :, :].reshape(-1, chain.shape[-1])

def plot_velo_dist(chain, sl , min_walker = None, plot_objs = None, color = None, plot_lines = False, plot_box = False, plot_violin = True, bestprob = False, lnprob = None):
    # samples = chain.swapaxes(0,1)[-100:, :, :].reshape(-1, chain.shape[-1])
    if plot_objs == None:
        fig, ax = plt.subplots(figsize = (8,6))
    else:
        fig, ax = plot_objs
    ndim = sl.ndim 

    samples = sample_from_chain(chain)

    vel_samples = samples[:, :sl.ndim]
    avg_av = np.nansum(np.median(sl.dAVdd, axis = 0))

    medians = np.nanmedian(samples[:, :], axis = 0)
    if bestprob:
        lp = lnprob.T
        lp[:, :-100] = -np.infty
        w_ind, stp_ind = np.unravel_index(np.argmax(lp), lp.shape)

        best_vals = chain[w_ind, stp_ind, :]

    stdevs = np.nanstd(samples[:, :], ddof = 1, axis = 0)

    med_velo = medians[:ndim]
    std_velo = stdevs[:ndim]


    med_dAV_dd = medians[ndim:]
    std_dAV_dd = stdevs[ndim:] 

    perc16, perc50,  perc84 = (np.percentile(samples[:, :], 16, axis = 0), 
                               np.percentile(samples[:, :], 50, axis = 0),
                               np.percentile(samples[:, :], 84, axis = 0) )
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
        vparts=ax.violinplot(vel_samples, pos, widths = w, showmeans=False, showextrema=False, showmedians=True,)
        for vindx, part in enumerate(vparts['bodies']):
            part.set_facecolor( 'C'+str(vindx))

            
        plot_guides = True
        if plot_guides:
            for pos in bin_pos:
                ax.plot((pos, pos), (-10, 20), color = 'k', linestyle = 'dotted')
                # axs[1].plot((pos, pos), (0, 1), color = 'k', linestyle = 'dotted')

        if bestprob:
            for bin_idx in range(len(bin_pos)-1):
                bin_min, bin_max = bin_pos[bin_idx], bin_pos[bin_idx + 1] #########
                ax.plot((bin_min, bin_max), (best_vals[bin_idx], best_vals[bin_idx]), color = "green")
    
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

# def transform_spectral_axis1(cube):
#     # https://www.ipac.caltech.edu/iso/lws/vhelio.html
#     vlsr = cube.spectral_axis.value
#     lmean = 0.5 * np.sum(cube.world_extrema[0])
#     bmean = 0.5 * np.sum(cube.world_extrema[1])
#     ut = -np.cos(bmean) * np.cos(lmean)
#     vt = np.cos(bmean) * np.sin(lmean)
#     wt = np.sin(bmean)
#     vhelio = vlsr -  ( -10.27 * ut ) + ( 15.32 * vt ) + ( 7.74 * wt )
#     return vhelio

def transform_spectral_axis2(cube):
    vlsr = cube.spectral_axis.value
    lmean = 0.5 * np.sum(cube.world_extrema[0])
    bmean = 0.5 * np.sum(cube.world_extrema[1])
    coord0 = coords.SkyCoord(l = lmean, b = bmean, distance = 500 * u.pc, radial_velocity = 0*u.km/u.s, pm_l_cosb = 0*u.arcsec/u.year, pm_b = 0*u.arcsec/u.year, frame = "galacticlsr")
    coord0_ICRS = coord0.transform_to(coords.ICRS)
    vhelio = vlsr + coord0_ICRS.radial_velocity.to(u.km/u.s).value
    return vhelio

def get_typical_emission_profile(emission):
    threshold = 0.03
    ref_point = (167.4, -8.3)
    b_em, l_em = emission.world[0, :, :][1:]
    b_em, l_em = b_em[:, 0], l_em[0, :]
    em_i, em_j = np.argmin(np.abs(l_em.value - ref_point[0])), np.argmin(np.abs(b_em.value - ref_point[1]))
    reference_point = emission.unmasked_data[:, em_j, em_i]
    corr_lags = correlation_lags(emission.shape[0], emission.shape[0])
    zpoint = corr_lags == 0

    correlation_image = np.zeros((emission.shape[1], emission.shape[2]))
    for i in range(emission.shape[1]):
        for j in range(emission.shape[2]):
            correlation_image[i, j] = correlate(emission.unmasked_data[:, i, j] / np.nansum(np.abs(emission.unmasked_data[:, i, j])), 
                                                reference_point / np.nansum(np.abs(reference_point)))[zpoint]
    emission_filament = np.copy(emission.unmasked_data[:])
    emission_filament[:, correlation_image < 0.03] = np.nan
    return np.nanmedian(emission_filament, axis = (1,2))


def plot_velo_dist_busy(reader, sl, emission = None, dust = None, avprior = None, metrics = None):
    chain = reader.get_chain()
    logprob = reader.get_log_prob()


    fig, ax = plt.subplots(figsize = (10, 10))
    fig, ax, dist_xx, med_velo, std_velo  = plot_velo_dist(chain, sl, plot_objs = (fig, ax), lnprob = logprob, bestprob= True)

    aux1 = ax.inset_axes([0, -0.1, 1, 0.1])
    aux2 = ax.inset_axes([0, 1, 1, 0.4])
    aux3 = ax.inset_axes([1, 0, 0.1, 1])

    samples = chain.swapaxes(0,1)[-100:, :, :].reshape(-1, chain.shape[-1])
    ndim = len(sl.voxel_dAVdd)
    walker_max = chain.shape[0]
    min_walker = -100
    vel_samples = samples[:, :sl.ndim]
    avg_av = np.nansum(np.median(sl.dAVdd, axis = 0))

    medians = np.nanmedian(samples[:, :], axis = 0)
    stdevs = np.nanstd(samples[:, :], ddof = 1, axis = 0)

    med_velo = medians[:ndim]
    std_velo = stdevs[:ndim]


    med_dAV_dd = medians[ndim:].reshape(-1, sl.ndim)
    std_dAV_dd = stdevs[ndim:].reshape(-1, sl.ndim) #CAUGHT 04.01 THIS WAS ASSIGNED TO med_dAV_dd
    mask = sl.dAVdd_mask
    med_dAVdd_masked = np.copy(med_dAV_dd)
    med_dAVdd_masked[mask] = np.nan

    if avprior == None:
        avprior = Logprior_Average_Extinction(sl, dust, emission, )
        prior_av = avprior.avg_dAVdd

    if metrics is not None:
        inds = np.digitize(sl.stars['DIST'], sl.bins) 
        aux1.hlines(metrics, sl.bins[inds -1], sl.bins[inds])
        aux1.set_xlabel('Distance (pc)')
        aux1.set_ylabel(r"$\chi^2$")

    if True: 
        best_step, best_walker = np.unravel_index(np.argmax(logprob), logprob.shape)
        vbest = chain[best_step, best_walker, :sl.ndim]
        # davdd = chain[best_step, best_walker, sl.ndim:2*sl.ndim]


    aux2.hlines(prior_av, sl.bins[:-1], sl.bins[1:], color = 'k', linestyles = "solid", label = "Avg prior")
    aux2.hlines(sl.voxel_dAVdd, sl.bins[:-1], sl.bins[1:], color = 'k', linestyles = 'dotted', label = "Mean")
    # aux2.hlines(med_dAV_dd, sl.bins[:-1], sl.bins[1:], color = 'k', linestyles = "dotted", label = "Mean")

    for i in range(len(sl.stars)):
        aux2.hlines(med_dAVdd_masked[i, :], sl.bins[:-1], sl.bins[1:], color = "C{}".format(i))
    aux2.set_ylabel(r'$\delta A_V$')
    
    vhelio_em = transform_spectral_axis2(emission)
    typical_emission = get_typical_emission_profile(emission)
    aux3.plot(typical_emission, vhelio_em, linestyle = "dashed")
    aux3.set_xlabel('Intensity \n (K km/s)')
    
    aux1.set_xlim(ax.get_xlim())
    aux2.set_xlim(ax.get_xlim())
    aux3.set_ylim(ax.get_ylim())
    

    
    return fig, ax