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