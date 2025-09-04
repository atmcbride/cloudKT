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
    sightline_setup_config["POPULATE_FROM_FILES"] = True
    sightlines = sightline_setup(stars, dust, emission_CO, emission_HI, sightline_setup_config, program_directory = input_dir)

    with open(input_dir + "/sightline_outputs/sightline_metrics.json", mode = "r") as f:
        initial_chi2_json = json.load(f)

    for i in range(len(sightlines)):
        stars = initial_sightlines[i].stars
        chi2_entry = np.array(initial_per_star_chi2[i])
        select_on_chi2 = chi2_entry < 2
        stars = stars[select_on_chi2]
        print(np.sum(select_on_chi2), len(chi2_entry))
        stars.write(output_dir + "/sightline_outputs/stars_{}.fits".format(i), overwrite = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process star data:")
    parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--make_plots", action="store_true", help="Generate plots")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    sigma_clip_save_stars(args.input_dir, args.output_dir, args.make_plots, args.overwrite)