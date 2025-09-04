import argparse
import json
import os

import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"
import sys
import logging

from multiprocessing import Pool

from utilities import load_module, load_json, merge_configs

logger = logging.getLogger(__name__)

program_directory = None
args = None

### UPDATE THIS ###
from star_selection_metrics import select_stars
from residual_process import generateClippedResidual

# zas


from plotting.plot_walkers import plot_walkers
from plotting.plot_signals import plot_signals_sample, plot_signals_sample_fg
from plot_velo_dist import plot_velo_dist, plot_velo_dist_busy
from plotting.plot_velo import plot_velo
from mcmc_framework import load_from_hdf5

def main():
    # Parse the command line arguments
    global args
    args = parse_args()
    global program_directory
    program_directory = args.directory
    config_file = args.config
    default_file = args.default

    logger_setup(program_directory + "/cloudKT.log")

    # Load the configuration file to get projects
    if os.path.exists(program_directory + "/" + config_file + ".json"):
        user_config = load_json(program_directory + "/" + config_file + ".json")
    else:
        user_config = {}
    default_config = load_json(default_file + ".json")
    config = merge_configs(user_config, default_config)

    if args.run_pipeline == "true":
        pipeline(config)


def pipeline(config):
    logger.info("Comment: " + args.init_comment)
    logger.info("Running pipeline...")
    logger.info("--- Loading Data ---")
    load_data_module = load_module(config["LOAD_DATA"]["MODULE"])
    load_data = getattr(load_data_module, config["LOAD_DATA"]["METHOD"])
    stars, dust, emission_CO, emission_HI = load_data(config["LOAD_DATA"]["PARAMETERS"])

    # load the sightline module
    logger.info("--- Loading sightline module ---")
    sightline_setup_module = load_module(config['SIGHTLINE_SETUP']['MODULE'])
    sightline_setup = getattr(sightline_setup_module, config['SIGHTLINE_SETUP']['FUNCTION'])
    uses_foreground = False
    if "foreground" in config['SIGHTLINE_SETUP']['MODULE']:
        uses_foreground = True
    sightline_setup_config = config['SIGHTLINE_SETUP']['PARAMETERS']


    sightline_setup_config["POPULATE_FROM_FILES"] = args.populate_from_files == "true"
    if args.populate_from_files=="true":
        sightline_setup_config["STARS_TO_FILES"] = False
    else:
        sightline_setup_config["STARS_TO_FILES"] = args.stars_to_files == "true"

    sightlines = sightline_setup(stars, dust, emission_CO, emission_HI, sightline_setup_config, program_directory = program_directory)
        
    for i in range(len(sightlines)):
        if args.stars_to_files == "true":
            if not os.path.exists(program_directory + "/sightline_outputs/"):
                os.mkdir(program_directory + "/sightline_outputs/")
            sightlines[i].stars.write(program_directory + "/sightline_outputs/stars_{}.fits".format(i), overwrite = True)


    logger.info("--- Running Model ---")
    logger.info("Starting MCMC Setup...")
    mcmc_module = load_module(config["RUN_MCMC"]["MODULE"])
    run_mcmc = getattr(mcmc_module, config["RUN_MCMC"]["METHOD"])
    mcmc_config = config["RUN_MCMC"]["PARAMETERS"]

    if args.run_MCMC == "true":
        for i in range(len(sightlines)):
            logger.info("Running MCMC for sightline {}...".format(i))
            mcmc_file = program_directory + "/sightline_outputs/mcmc_output_{}.h5".format(i)
            pool = Pool(12)
            logger.info("Running MCMC...")
            sampler = run_mcmc(
                sightlines[i], mcmc_config, dust, emission_CO, pool=pool, filename=mcmc_file
            , **mcmc_config)

    for i in range(len(sightlines)):

        mcmc_file = program_directory + "/sightline_outputs/mcmc_output_{}.h5".format(i)
        reader = load_from_hdf5(mcmc_file)
        chain = reader.get_chain()
        # for j in range(0, sightlines[i].ndim, 1):
        #     fig, ax = plot_walkers(chain, j, sightline = sightlines[i])
        #     fig.savefig(program_directory+ '/figures/chain_sl{i}_var{j}.jpg'.format(i=i, j=j))
        #     plt.close()
        # for j in range(sightlines[i].ndim, 2 * sightlines[i].ndim, 2):
        #     fig, ax = plot_walkers(chain, j, sightline = sightlines[i])
        #     fig.savefig(program_directory+ '/figures/chain_sl{i}_var{j}.jpg'.format(i=i, j=j))
        #     plt.close()

        plot_signals = plot_signals_sample_fg if uses_foreground else plot_signals_sample
        fig, ax = plot_signals(reader, sightlines[i])

        fig.savefig(program_directory + '/figures/signals_sl_{i}.jpg'.format(i=i))
        plt.close()


        # fig, ax, dist_xx, med_velo, std_velo = plot_velo_dist(chain, sightlines[i])
        # fig.savefig(program_directory + "/figures/v_d_sl_{i}.jpg".format(i=i))
        # plt.close()

    metrics_out = {}
    for i in range(len(sightlines)):


        mcmc_file = program_directory + "/sightline_outputs/mcmc_output_{}.h5".format(i)
        reader = load_from_hdf5(mcmc_file)
        chain = reader.get_chain()
        postprocessing_module = load_module("postprocessing")
        chi2_statistics = getattr(postprocessing_module, "chi2_statistics")
        per_star_chi2, median_star_chi2, std_star_chi2, sightline_chi2 = chi2_statistics(sightlines[i], chain)
        logger.info("Sightline {} chi2 ".format(i), str(sightline_chi2))
        metrics_out["sl_{}".format(i)] = {"sightline_chi2": sightline_chi2, "median_chi2": median_star_chi2,
                                            "std_chi2": std_star_chi2, "perstar_chi2": list(per_star_chi2)}
    with open(program_directory + "/sightline_outputs/sightline_metrics.json", mode = "a") as f:
        json.dump(metrics_out, f, indent = 2)

    if not os.path.exists(program_directory + "/figures/"):
        os.mkdir(program_directory + "/figures/")
    
    for i in range(len(sightlines)):

        sl = sightlines[i]
        # sl = sightlines[i]

        mcmc_file = program_directory + "/sightline_outputs/mcmc_output_{}.h5".format(i)
        reader = load_from_hdf5(mcmc_file)
        chain = reader.get_chain()
        per_star_chi2, median_star_chi2, std_star_chi2, sightline_chi2 = chi2_statistics(sl, chain)

        fig, ax, dist_xx, med_velo, std_velo = plot_velo_dist(chain, sl)
        fig.savefig(program_directory + "/figures/velodist_sl_{i}.jpg".format(i=i))
        plt.close()

        # fig, ax = plot_velo_dist_busy(reader, sightlines[i], dust = dust, emission = emission_CO, avprior = None, metrics = per_star_chi2)
        # fig.savefig(program_directory + "/figures/velodist_busy_sl_{i}.jpg".format(i=i))


        # fig, ax, dist_xx, med_velo, std_velo = plot_velo(chain, sightlines[i])
        # fig.savefig(program_directory + "/figures/veloposterior_sl_{i}.jpg".format(i=i))
        # plt.close()      

    logger.info("Successfully ran through cloudKT.pipeline!")

def parse_args():
    parser = argparse.ArgumentParser(description="nanoKT_v2")
    parser.add_argument("directory", type=str, help="Input directory")
    parser.add_argument(
        "--config", type=str, help="Configuration file", default="CONFIG"
    )
    # parser.add_argument('--log', type=str, help='Log file name', default='LOG')
    parser.add_argument(
        "--default", type=str, help="Default configuration file", default="DEFAULTS"
    )
    parser.add_argument("--run_pipeline", type=str, help= "Run pipeline", default="true")
    parser.add_argument("--stars_to_files", type = str, help = 'Save sightline input stars to .fits files', default = 'true')
    parser.add_argument("--populate_from_files", type = str, help = 'Populate sightlines from previously-saved .fits files', default = 'false')
    parser.add_argument("--run_MCMC", type=str, help = "Run MCMC? Otherwise, load chains from a previous run", default = "true")
    parser.add_argument("--make_plots", type=str, help = "Make plots using specified plotting functions", default = "true")
    parser.add_argument("--init_comment", type = str, help = "Initial message to describe run", default = "")
    return parser.parse_args()


def logger_setup(log_file, level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fhandler = logging.FileHandler(log_file, mode="w")
    fhandler.setLevel(logging.INFO)
    fformatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    fhandler.setFormatter(fformatter)

    shandler = logging.StreamHandler()
    shandler.setLevel(logging.INFO)
    sformatter = logging.Formatter("%(asctime)s - %(message)s")
    shandler.setFormatter(sformatter)

    logger.addHandler(shandler)
    logger.addHandler(fhandler)


if __name__ == "__main__":
    main()
