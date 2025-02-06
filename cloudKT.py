import argparse
import importlib
import json
import os
import sys
import logging

def main():
    # Parse the command line arguments
    args = parse_args()
    program_directory = args.directory
    config_file = args.config
    default_file = args.default

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

    print("--- Loading Data ---")
    load_data = load_module(config['INPUT_DATA_MODULE'])
    print('Loading star data...')
    load_stars = getattr(load_data, config['STAR_LOADER'])
    star_data = load_stars(config['STAR_LOADER_PARAMETERS'])

    print('Loading dust data...')
    dust_loader = getattr(load_data, config['DUSTMAP_LOADER'])
    dust_data = dust_loader(map_fname = config['DUSTMAP_FILE'])



    dust_data.load_map()
    print(dust_data)

    # load the sightline module
    print('--- Loading sightline module ---')
    sightline_module = load_module(config['SIGHTLINE_MODULE'])
    Sightline = getattr(sightline_module, config['SIGHTLINE_OBJECT'])
    print(Sightline)
    print('Populating sightlines...')

    print('--- Loading Model ---')
    print('Loading MCMC module...')

    print('--- Running Model ---')
    print('Running MCMC...')

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

def load_json(json_path, **kwargs):
    # Load the configuration file.
    with open(json_path, 'r') as f:
        return json.load(f)
    
def merge_configs(user, default):
    # Merge user configuration with defaults.
    merged = default.copy()
    merged.update(user)
    return merged

def load_module(module_name):
    # Load the module
    return importlib.import_module(module_name)

if __name__ == '__main__':
    main()
