import argparse
import json
import os
import sys
import logging

from utilities import load_module, load_json, merge_configs#, logger_setup

logger = logging.getLogger(__name__)

def main():
    # Parse the command line arguments
    args = parse_args()
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

    sightlines = []
    for i in range(10): 
        sightlines.append(Sightline(stars, (158+i, -8.5)))
    
    logger.info('Populating sightlines...')

    logger.info('--- Loading Model ---')
    logger.info('Loading MCMC module...')

    logger.info('--- Running Model ---')
    logger.info('Running MCMC...')

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
