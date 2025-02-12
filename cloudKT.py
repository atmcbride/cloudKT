import argparse
import json
import os
import sys
import logging

from multiprocessing import Pool

from utilities import load_module, load_json, merge_configs

logger = logging.getLogger(__name__)

program_directory = None

def main():
    # Parse the command line arguments
    args = parse_args()
    global program_directory
    program_directory = args.directory
    config_file = args.config
    default_file = args.default

    logger_setup(program_directory + '/cloudKT.log')

    # Load the configuration file to get projects
    if os.path.exists(program_directory + '/' + config_file + '.json'):
        user_config = load_json(program_directory + '/' + config_file + '.json')
    else: 
        user_config = {}
    default_config = load_json(default_file + '.json')
    config = merge_configs(user_config, default_config)
    
    if args.run_pipeline == 'true':
        pipeline(config)

def pipeline(config):
    logger.info('Running pipeline...')
    logger.info('--- Loading Data ---')
    load_data_module = load_module(config['LOAD_DATA_MODULE'])
    load_data = getattr(load_data_module, config['LOAD_DATA_METHOD'])
    stars, dust, emission_CO, emission_HI = load_data(config['LOAD_DATA_PARAMETERS'])


    # load the sightline module
    logger.info('--- Loading sightline module ---')
    sightline_module = load_module(config['SIGHTLINE_MODULE'])
    Sightline = getattr(sightline_module, config['SIGHTLINE_OBJECT'])
    logger.info('Populating sightlines...')

    sightlines = []
    log_probabilities = []

    mcmc_module = load_module(config['MCMC_FRAMEWORK']['MODULE'])
    # log_likelihood = getattr(mcmc_module, config['MCMC_FUNCTIONS']['LOG_LIKELIHOOD'])
    # build_log_probability = getattr(mcmc_module, config['MCMC_FRAMEWORK']['BUILD_LOGPROB_FUNCTION'])
    run_mcmc = getattr(mcmc_module, config['MCMC_FRAMEWORK']['RUN_MCMC_FUNCTION'])
    mcmc_fns_module = load_module(config['MCMC_FUNCTIONS']['LOG_LIKELIHOOD']['MODULE'])
    log_likelihood = getattr(mcmc_fns_module, config['MCMC_FUNCTIONS']['LOG_LIKELIHOOD']['FUNCTION'])

    log_priors = []
    for item in config['MCMC_FUNCTIONS']['LOG_PRIOR']:
        log_prior_module = load_module(item['MODULE'])
        log_prior = getattr(log_prior_module, item['FUNCTION'])
        log_priors.append((log_prior, item['PARAMETERS']))

    for i in range(1): 
        sightlines.append(Sightline(stars, (160+i, -8.5), dust))

    logger.info('--- Running Model ---')
    pool = Pool(8)
    logger.info('Running MCMC...')

    mcmc_file = program_directory + '/mcmc_output.h5'
    sampler = run_mcmc(sightlines[0], log_likelihood, log_priors, pool = pool, filename = mcmc_file)


    # Load in dust data
    # FILE: Load in stellar data, apply selections to assign to sightlines
    # FILE: Pass in stars, assign sightlines
    # FILE: Run the model, save outputs into h5 format...

def parse_args():
    parser = argparse.ArgumentParser(description='nanoKT_v2')
    parser.add_argument('directory', type=str, help='Input directory')
    parser.add_argument('--config', type=str, help='Configuration file', default='CONFIG')
    # parser.add_argument('--log', type=str, help='Log file name', default='LOG')
    parser.add_argument('--default', type=str, help='Default configuration file', default='DEFAULTS')
    parser.add_argument('--run_pipeline', type =str, help ='Run pipeline', default='true')
    return parser.parse_args()

def logger_setup(log_file, level = logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fhandler = logging.FileHandler(log_file, mode='w')    
    fhandler.setLevel(logging.INFO)
    fformatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fhandler.setFormatter(fformatter)

    shandler = logging.StreamHandler()
    shandler.setLevel(logging.INFO)
    sformatter = logging.Formatter('%(asctime)s - %(message)s')
    shandler.setFormatter(sformatter)
    
    logger.addHandler(shandler)
    logger.addHandler(fhandler)


if __name__ == '__main__':
    main()
