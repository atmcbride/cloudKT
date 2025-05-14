import numpy as np
import logging

from utilities import load_module

logger = logging.getLogger(__name__)


def initialize_synthetic_sightlines(stars, dust, emission_CO, emission_HI, sightline_config, program_directory = None, **kwargs):
    sightlines = []
    


    sightline_module = load_module(sightline_config["SIGHTLINE"]["MODULE"])
    Sightline = getattr(sightline_module, sightline_config["SIGHTLINE"]["CLASS"])

    if sightline_config["DATA_REPROCESS"] != "none":
        reprocess_module = load_module(sightline_config["DATA_REPROCESS"]["MODULE"])
        reprocess_function = getattr(reprocess_module, sightline_config["DATA_REPROCESS"]["FUNCTION"])
        Sightline.alternative_data_processing = reprocess_function
    data_processing_kwargs = sightline_config["DATA_REPROCESS"]["PARAMETERS"]

    if sightline_config["STAR_SELECTION"] != "none":
        if sightline_config["POPULATE_FROM_FILES"] != True:
            star_selection_module = load_module(sightline_config["STAR_SELECTION"]["MODULE"])
            star_selection_function = getattr(star_selection_module, sightline_config["STAR_SELECTION"]["FUNCTION"])
            Sightline.select_stars = star_selection_function
        else:
            Sightline.select_stars = Sightline.populate_from_file

    star_selection_kwargs = sightline_config["STAR_SELECTION"]["PARAMETERS"]
    star_selection_kwargs["emission"] = emission_CO


    for i in range(sightline_config["N_SIGHTLINES"]):
        initialization_params = sightline_config['SIGHTLINE_INIT_SETUPS'][i]
        if sightline_config["POPULATE_FROM_FILES"] == True:
            if i == 0:
                print('populating from files!')
            star_selection_kwargs = {"fname": program_directory + "/sightline_outputs/stars_{}.fits".format(i)}

        coords = None

        dust_profile, dust_profile_err = Sightline.get_dust_profile(dust, emission_CO, 400, 600, 4, **initialization_params)
        velo_profile = Sightline.get_velo_profile(dust, -10, 10, 400, 600)


        sightline = Sightline(stars, dust, velo_profile, dust_profile, dust_profile_err = dust_profile_err, data_processing_kwargs = data_processing_kwargs, star_selection_kwargs = star_selection_kwargs)
        sightlines.append(sightline)

    return sightlines
