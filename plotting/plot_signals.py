import numpy as np
import matplotlib.pyplot as plt


def plot_signals_sample(chain, sl, logprob = None):
    samples = chain[:, int(0.8 * chain.shape[1]), :]
    v = np.nanmedian(samples[:, :sl.ndim], axis = 0)
    davdd_all = np.nanmedian(samples[:, sl.ndim:], axis = 0).reshape((sl.nsig, sl.ndim))
    wavs_window = sl.wavs_window

    if False: 
        best_step, best_walker = np.unravel_index(np.argmax(logprob), logprob.shape)
        davdd = chain[best_step, best_walker, sl.ndim:2*sl.ndim]
        print(davdd.shape)

    def sample_signal():
        idx = (np.random.choice(samples.shape[0]))
        v = samples[idx, :sl.ndim]
        av = samples[idx, sl.ndim:]
        # av_scatter = samples[idx, 2*sl.ndim:]
        davdd_all = av.reshape((sl.nsig, sl.ndim))
        model_signals = sl.model_signals(v, davdd_all)
        return model_signals

    model_signals = sl.model_signals(v, davdd_all)

    fig, ax = plt.subplots(figsize = (10, 20))
    order_inds = np.array(np.argsort(sl.stars['DIST']))
    for i in range(len(order_inds)):
        ii = order_inds[i]
        # bindex = sl.bin_inds[ii]
        # ax.plot(wavs_window, sl.signals[ii, :] + 0.05 * i, color = 'C{}'.format(i))
        # ax.fill_between(wavs_window, sl.signals[ii, :] + sl.signal_errs[ii, :] + 0.05 * i,
        #                  sl.signals[ii, :] - sl.signal_errs[ii, :] + 0.05 * i, color = 'C{}'.format(bindex), alpha = 0.1)

        ax.plot(wavs_window, model_signals[ii, :] + 0.1 * i, color = 'C{}'.format(i))        
        ax.plot(wavs_window, sl.signals[ii, :]+ 0.1 * i, color = 'C{}'.format(i))
        for k in range(50):
            ax.plot(wavs_window, sample_signal()[ii, :] + 0.1 * i, color = 'C{}'.format(i), alpha = 0.05)
        ax.set_xlabel('Wavelength ($\AA$)')
        ax.set_ylabel('Flux + Offset')
    return fig, ax


###########################################################################################################

sigma0 = 1.15
lambda0 = 15272.42
def model_signals_fg(rvelo, sl, dAVdd):
    # dAVdd = sl.dAVdd
    wavs_window = sl.wavs_window
    voxeldiffDIB = np.zeros((len(sl.stars), len(wavs_window)))
    unsummed_signals = np.zeros((len(sl.stars), len(sl.bins)-1,len(wavs_window)))
    # unsummed_signals = np.zeros((len(sl.bins)-1, len(sl.bins)-1,len(wavs_window)))

#     print(voxeldiffDIB.shape, unsummed_signals.shape, sl.dAVdd.shape)
    peak_wavelength = sl.dopplershift(rvelo)
    wavs_grid = np.tile(wavs_window, (len(sl.bins) - 1, 1))
    voxel_DIB_unscaled = np.exp(-(wavs_grid - peak_wavelength[:, np.newaxis])**2 / (2 * sigma0**2))
    amp = sl.differentialAmplitude(dAVdd, sl.bins[1:]-sl.bins[:-1])

    def single_signal(amp, bindex):

        amp[bindex:] = 0 # THIS MIGHT NEED TO BE -1
        # print(amp)

        voxel_DIB_scaled = -voxel_DIB_unscaled *  amp[:, np.newaxis] 
        summed_DIB = np.sum(voxel_DIB_scaled, axis = 0)
        # continuum = lambda x, m, b : m * (x - lambda0) + b
        # cont = continuum(wavs_window, 0, b)
        return summed_DIB  + 1, voxel_DIB_scaled 

    fgdiffDIB = np.zeros((len(sl.stars), len(wavs_window)))
    

    for i in range(len(sl.stars)): # Iterate over each star in dAVdd array
#         print(dAVdd)
        dAVdd_bin = dAVdd[i, :] 

        amp = sl.differentialAmplitude(dAVdd_bin, 1)

        bin_index = np.concatenate([sl.bin_inds]).astype(int)[i] # this only goes to 
        # signals[i, :] = single_signal(bin_index)
        voxeldiffDIB[i, :], unsummed_signals[i, :, :] = single_signal(amp, bin_index)
        fgdiffDIB[i, :], _ = single_signal(amp, 1)


    return voxeldiffDIB, unsummed_signals, fgdiffDIB


def plot_signals_sample_fg(chain, sl, logprob = None):
    samples = chain.swapaxes(0,1)[-100:, :, :].reshape(-1, chain.shape[-1])

    wavs_window = sl.wavs_window
    v = np.nanmedian(samples[:, :sl.ndim], axis = 0)
    davdd = np.nanmedian(samples[:, sl.ndim:], axis = 0).reshape(-1, sl.ndim)
    # av_offset = np.nanmedian(samples[:, 2*sl.ndim:], axis = 0)
    # davdd_all = np.ones((sl.nsig, sl.ndim)) * davdd + av_offset.reshape((sl.nsig, sl.ndim))

    if False: 
        best_step, best_walker = np.unravel_index(np.argmax(logprob), logprob.shape)
        davdd = chain[best_step, best_walker, sl.ndim:2*sl.ndim]
#         print(davdd.shape)
        davdd_all = davdd * np.ones((sl.nsig, sl.ndim)) + av_offset.reshape((sl.nsig, sl.ndim))
    # model_signals_fg = sl.model_signals

    def sample_signal():
        idx = (np.random.choice(samples.shape[0]))
        v = samples[idx, :sl.ndim]
        av = samples[idx, sl.ndim:].reshape(-1, sl.ndim)

        voxeldiffDIB, unsummed_signals, fgdiffDIB = model_signals_fg(v, sl, dAVdd = av)

        return  voxeldiffDIB, unsummed_signals, fgdiffDIB

    # model_signals, signal_recreated_unsummed, fg_unsummed = model_signals_fg(v, sl, davdd)\
    model_signals, signal_recreated_unsummed, fg_unsummed = model_signals_fg(v, sl, dAVdd = davdd)

    fig, ax = plt.subplots(figsize = (10, 20))

    samp_signal = np.zeros((50, len(sl.stars), len(wavs_window)))
    samp_unsummed = np.zeros((50, len(sl.stars), len(sl.bins)-1, len(wavs_window)))
    samp_fgdiffDIB = np.zeros((50, len(sl.stars), len(wavs_window)))

    for idx in range(50):
        voxeldiffDIB, unsummed_signals, fgdiffDIB = sample_signal()
        samp_signal[idx, :, :] = voxeldiffDIB
        samp_unsummed[idx, :, :] = unsummed_signals
        samp_fgdiffDIB[idx, : :] = fgdiffDIB

    order_inds = np.array(np.argsort(sl.stars['DIST']))

    sep = 0.1
    offset = 0.1

    # signal_recreated, signal_recreated_unsummed = model_signals_thing(med_velo, sl, med_dAV_dd) 
    for i in range(len(order_inds)):
        ii = order_inds[i]
        # bindex = sl.bin_inds[ii]
        # ax.plot(wavs_window, sl.signals[ii, :] + 0.05 * i, color = 'C{}'.format(i))
        # ax.fill_between(wavs_window, sl.signals[ii, :] + sl.signal_errs[ii, :] + 0.05 * i,
        #                  sl.signals[ii, :] - sl.signal_errs[ii, :] + 0.05 * i, color = 'C{}'.format(bindex), alpha = 0.1)


        ax.plot(wavs_window, model_signals[ii, :] + sep * i + offset, color = 'C{}'.format(i))        
        ax.plot(wavs_window, sl.signals[ii, :]+ sep * i + offset, linestyle = 'dotted', color = 'C{}'.format(i))
        for k in range(50):
            # samp, _, _ = sample_signal()
            ax.plot(wavs_window, samp_signal[k, ii, :] + sep * i + offset, alpha = 0.05, color ='C{}'.format(i))
        # for j in range(len(sl.bins)-1):
        #     if j==0:
        #         col = 'grey'
        #     else:
        #         col = 'C{}'.format(j-1)
        #     if (j > i+1) | (j >= 1):
        #         continue

        #     ax.plot(wavs_window, unsummed_signals[ii, j, : ] + 1  + sep * i + offset, color=col, linestyle = 'dashed', alpha = 1)
        


    ax.set_xlabel('Wavelength ($\AA$)')
    ax.set_ylabel('Flux + Offset')
    ax.plot(wavs_window, fg_unsummed[0, :]  -sep + offset, linestyle = 'dashed', color ='grey',)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(15272-10, 15272 + 10)
    ax.fill_between([15272-10, 15272+10], [1 + 0.1 * sep, 1 + 0.1 * sep], [ymin, ymin], color = 'grey', alpha = 0.1)
        
    ax.set_xticks(np.arange(15272-10, 15272+14, 4))
    return fig, ax        
