import numpy as np
import emcee
import logging
from functools import partial
import os


from utilities import load_module


logger = logging.getLogger(__name__)


def build_moves(mcmc_config):
    if "MOVES" not in mcmc_config.keys():
        logger.info("No moves specified; defaulting to StretchMove")
        return [(emcee.moves.StretchMove(), 1.0)]
    move_names_list = mcmc_config["MOVES"]
    move_args_list = mcmc_config["MOVES_ARGS"]
    move_kwargs_list = mcmc_config["MOVES_KWARGS"]
    move_probs_list = mcmc_config["MOVES_PROBS"]

    moves = []
    for i in range(len(move_names_list)):
        move_name = move_names_list[i]
        move_args = tuple(move_args_list[i])
        move_kwargs = move_kwargs_list[i]
        move_prob = move_probs_list[i]

        move = getattr(emcee.moves, move_name)
        moves.append((move(*move_args, **move_kwargs), move_prob))

    logger.info("Custom moves built: " + str(move_names_list))
    return moves


def run_mcmc(
    sightline,
    mcmc_config,
    *args,
    steps=1000,
    nwalkers=100,
    pool=None,
    filename=None,
    init_vabs=15,
    resume_mcmc=False,
    skip_existing=False,
    **kwargs,
):
    """ "
    Run the MCMC
    """
    ndim = len(sightline.voxel_dAVdd)
    nstar = len(sightline.stars)
    ndim_amp = int(ndim + ndim * nstar)

    if nwalkers < 2 * ndim_amp:
        nwalkers = 2 * ndim_amp + 5
        logger.info("N walkers updated to " + str(nwalkers))

    skip_running = False
    if filename is not None:
        if os.path.exists(filename) & skip_existing:
            logger.info("A file currently exists for this run. Skip existing: " + str(skip_existing))
            skip_running = True
        backend = emcee.backends.HDFBackend(filename)

        if skip_running:
            if backend.iteration != steps:
                logger.info("Previous model run was not to the correct number of steps. Has {existing}, needs {needed}.".format(existing=backend.iteration, needed=steps))
                logger.info("Will re-run this model.")
                skip_running = False # in the future add code for resuming with x number steps

        # backend_shape = backend.shape
        # print(backend_shape)

        
    else:
        backend = None
        logger.warning("NO BACKEND")

    ll_config = mcmc_config["LOG_LIKELIHOOD"]
    ll_module = load_module(ll_config["MODULE"])
    ll_fn = getattr(ll_module, ll_config["FUNCTION"])
    ll_params = ll_config["PARAMETERS"]
    log_likelihood = (ll_fn, ll_params)  # Pass as tuple (fn, fn_kwargs)

    # if "proposal_size" in mcmc_config.keys():
    #     proposal_size = mcmc_config['proposal_size']
    #     logger.info('Proposal size specified' + str(proposal_size))
    # else:
    #     proposal_size = 2

    # add a function for specifying moves in the config files! but for now this is fine
    # moves = [(emcee.moves.StretchMove(a=proposal_size), 1.0)]

    moves = build_moves(mcmc_config)

    log_priors = []
    lp_config = mcmc_config["LOG_PRIOR"]
    for lp_entry in lp_config:
        lp_module = load_module(lp_entry["MODULE"])
        if "OBJECT" in lp_entry.keys():
            lp_object = getattr(lp_module, lp_entry["OBJECT"])(
                sightline, *args, **lp_entry["INIT_KWARGS"]
            )
            lp_fn = getattr(lp_object, lp_entry["FUNCTION"])
            lp_params = lp_entry["PARAMETERS"]
            log_prior = (lp_fn, lp_params)
            log_priors.append(log_prior)

        else:
            lp_fn = getattr(lp_module, lp_entry["FUNCTION"])
            lp_params = lp_entry["PARAMETERS"]
            log_prior = (lp_fn, lp_params)
            log_priors.append(log_prior)  # Pass as list of tuples (fn, fn_kwargs)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim_amp,
        log_probability,
        pool=pool,
        backend=backend,
        kwargs={
            "sightline": sightline,
            "log_likelihood": log_likelihood,
            "log_priors": log_priors,
        },
        moves=moves,
    )

    # init = 10 *  (np.random.random((nwalkers, ndim_amp)) - 0.5)

    init = 10 * (np.random.random((nwalkers, ndim_amp)) - 0.5)
    init = 90 * (np.random.random((nwalkers, ndim_amp)) - 0.5)
    init = 2 * init_vabs * (np.random.random((nwalkers, ndim_amp)) - 0.5)

    init[:, ndim:] = np.abs(
        sightline.dAVdd.ravel()[np.newaxis, :]
        + 0.1 * (np.random.random(init[:, ndim:].shape) - 0.5)
    )
    # init[:, ndim:][(init[:, ndim:] <= 0.1)] = 0.11 + 0.05 * np.random.random(np.sum(init[:, ndim:]<= 0.1))
    # init[:, ndim:][(init[:, ndim:] <= 0.0)] = 0.11 + 0.05 * np.random.random(np.sum(init[:, ndim:]<= 0.0))

    print("NDIM:", ndim, "NSTAR:", nstar, "INITSHAPE:", init.shape)

    # sampler.run_mcmc(init, steps, progress = False, store = True);

    if "BACKEND_THIN" not in mcmc_config.keys():
        backend_thin = 1
    else:
        backend_thin = mcmc_config["BACKEND_THIN"]

    if skip_running:
        return sampler
    
    if not resume_mcmc:
        backend.reset(nwalkers, ndim_amp) #oops 

    sampler.run_mcmc(init, steps, progress=False, thin_by=backend_thin, store=True)

    # for sample in sampler.sample(init, iterations=steps, thin_by = backend_thin, progress=True, store= True):
    #     pass
    # if sampler.iteration % backend_thin == 0:
    #     backend.save_step(sampler)  # this is handled automatically in backend

    return sampler


def run_mcmc_smaller(
    sightline,
    mcmc_config,
    *args,
    steps=1000,
    nwalkers=100,
    pool=None,
    filename=None,
    **kwargs,
):
    """
    Run the MCMC (this version, not implemented, cuts down on some of the variables)
    """
    ndim = len(sightline.voxel_dAVdd)
    nstar = len(sightline.stars)
    ndim_amp = int(ndim + ndim * nstar)

    if nwalkers < 2 * ndim_amp:
        nwalkers = 2 * ndim_amp + 5
        logger.info("N walkers updated to " + str(nwalkers))

    if filename is not None:
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim_amp)
    else:
        backend = None
        logger.warning("NO BACKEND")

    ll_config = mcmc_config["LOG_LIKELIHOOD"]
    ll_module = load_module(ll_config["MODULE"])
    ll_fn = getattr(ll_module, ll_config["FUNCTION"])
    ll_params = ll_config["PARAMETERS"]
    log_likelihood = (ll_fn, ll_params)  # Pass as tuple (fn, fn_kwargs)

    log_priors = []
    lp_config = mcmc_config["LOG_PRIOR"]
    for lp_entry in lp_config:
        lp_module = load_module(lp_entry["MODULE"])
        if "OBJECT" in lp_entry.keys():
            lp_object = getattr(lp_module, lp_entry["OBJECT"])(
                sightline, *args, **lp_entry["INIT_KWARGS"]
            )
            lp_fn = getattr(lp_object, lp_entry["FUNCTION"])
            lp_params = lp_entry["PARAMETERS"]
            log_prior = (lp_fn, lp_params)
            log_priors.append(log_prior)

        else:
            lp_fn = getattr(lp_module, lp_entry["FUNCTION"])
            lp_params = lp_entry["PARAMETERS"]
            log_prior = (lp_fn, lp_params)
            log_priors.append(log_prior)  # Pass as list of tuples (fn, fn_kwargs)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim_amp,
        log_probability,
        pool=pool,
        backend=backend,
        kwargs={
            "sightline": sightline,
            "log_likelihood": log_likelihood,
            "log_priors": log_priors,
        },
    )

    init = 10 * (np.random.random((nwalkers, ndim_amp)) - 0.5)  # KEEP
    init = 30 * (np.random.random((nwalkers, ndim_amp)) - 0.5)

    # init = 60 * (np.random.random((nwalkers, ndim_amp)) - 0.5)

    init[:, ndim:] = np.abs(
        sightline.dAVdd.ravel()[np.newaxis, :]
        + 0.1 * (np.random.random(init[:, ndim:].shape) - 0.5)
    )
    init[:, ndim:][(init[:, ndim:] <= 0.1)] = np.abs(
        0.11 + 0.05 * np.random.random(np.sum(init[:, ndim:] <= 0.1))
    )  # KEEP
    # init[:, ndim:][init[:, ndim:] < 0] = init[:, ndim:][init[:, ndim:] < 0]
    # init[:, ndim:][(init[:, ndim:] <= 0.0)] = 0.11 + 0.05 * np.random.random(np.sum(init[:, ndim:]<= 0.0))

    print("NDIM:", ndim, "NSTAR:", nstar, "INITSHAPE:", init.shape)

    sampler.run_mcmc(init, steps, progress=False, store=True)

    return sampler


def log_probability(
    theta, sightline=None, log_likelihood=None, log_priors=None, **kwargs
):
    """
    For a given input vector theta, populated Sightline object (for data and modeling functions), log-likelihood function with inputs,
    and a list of log-prior functions with inputs, calculate and return the log-probability of theta
    """
    ll_fn, ll_kwargs = log_likelihood
    ll = ll_fn(theta, sightline=sightline, **ll_kwargs)
    lp, lp_list = evaluate_log_prior(
        theta, log_priors=log_priors, sightline=sightline, **kwargs
    )

    return ll + lp, lp, *tuple(lp_list)


def expand_theta(theta, sightline):
    mask = sightline.dAVdd_mask


def evaluate_log_prior(theta, log_priors=None, sightline=None, **kwargs):
    lp = 0
    lp_list = []
    for lp_entry in log_priors:
        lp_fn, lp_kwargs = lp_entry
        lp_fn_val = lp_fn(theta, sightline=sightline, **lp_kwargs)
        lp += lp_fn_val
        lp_list.append(lp_fn_val)
    return lp, lp_list


def load_from_hdf5(h5_fname):
    reader = emcee.backends.HDFBackend(h5_fname)
    return reader
