import numpy as np
import argparse
import sys
import os
import json
from utilities import load_module, load_json, merge_configs
import logging 

logger = logging.getLogger(__name__)

def sigma_clip_save_stars(input_dir, output_dir, make_plots = False, overwrite = False):

    if os.path.exists(output_dir) & (not overwrite):
        print('New folder already exists! If you mean to overwrite existing location, specify --overwrite.')
        return
    else:
        os.makedirs(output_dir + '/figures', exist_ok = True)
        os.makedirs(output_dir + '/sightline_outputs', exist_ok = True)

    # Get config setup
    if os.path.exists(input_dir + "/config.json"):
        user_config = load_json(input_dir + "/config.json")
    else:
        user_config = {}
    default_config = load_json('DEFAULTS.json')
    config = merge_configs(user_config, default_config)

    # Load data
    load_data_module = load_module(config["LOAD_DATA"]["MODULE"])
    load_data = getattr(load_data_module, config["LOAD_DATA"]["METHOD"])
    stars, dust, emission_CO, emission_HI = load_data(config["LOAD_DATA"]["PARAMETERS"])

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process star data:")
    parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--make_plots", action="store_true", help="Generate plots")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    sigma_clip_save_stars(args.input_dir, args.output_dir, args.make_plots, args.overwrite)