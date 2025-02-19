import argparse
import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
import sys
import logging

from multiprocessing import Pool

from utilities import load_module, load_json, merge_configs

logger = logging.getLogger(__name__)

program_directory = None




from plotting.plot_walkers import plot_walkers
from plotting.plot_signals import plot_signals_sample
from mcmc_framework import load_from_hdf5

def main():
    # Parse the command line arguments
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
    logger.info("Running pipeline...")
    logger.info("--- Loading Data ---")
    load_data_module = load_module(config["LOAD_DATA"]["MODULE"])
    load_data = getattr(load_data_module, config["LOAD_DATA"]["METHOD"])
    stars, dust, emission_CO, emission_HI = load_data(config["LOAD_DATA"]["PARAMETERS"])

    # load the sightline module
    logger.info("--- Loading sightline module ---")
    sightline_module = load_module(config["SIGHTLINE"]["MODULE"])
    Sightline = getattr(sightline_module, config["SIGHTLINE"]["CLASS"])
    logger.info("Populating sightlines...")

    sightlines = []
    for i in range(1):
        sightlines.append(Sightline(stars, (163 + i, -8.5), dust))

    logger.info("--- Running Model ---")
    logger.info("Starting MCMC Setup...")
    mcmc_module = load_module(config["RUN_MCMC"]["MODULE"])
    run_mcmc = getattr(mcmc_module, config["RUN_MCMC"]["METHOD"])
    mcmc_config = config["RUN_MCMC"]["PARAMETERS"]

    mcmc_file = program_directory + "/mcmc_output.h5"
    if True:
        pool = None
        logger.info("Running MCMC...")

        mcmc_file = program_directory + "/mcmc_output.h5"
        sampler = run_mcmc(
            sightlines[0], mcmc_config, pool=pool, filename=mcmc_file
        )

    reader = load_from_hdf5(mcmc_file)
    for i in range(1):
        chain = reader.get_chain()
        for j in range(0, sightlines[0].ndim + sightlines[0].ndim * sightlines[0].nsig, 1):
            fig, ax = plot_walkers(chain, j)
            fig.savefig(program_directory+ '/chain_sl{i}_var{j}.jpg'.format(i=i, j=j))
            fig.close()
        fig, ax = plot_signals_sample(chain, sightlines[i])
        fig.savefig(program_directory + '/signals_sl{i}.jpg'.format(i=i, j=j))
        fig.close()




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
    parser.add_argument("--load_chains", type=str, help = "Load chains from a previous run?", default = "false")
    parser.add_argument("--make_plots", type=str, help = "Make plots using specified plotting functions", default = "false")
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
