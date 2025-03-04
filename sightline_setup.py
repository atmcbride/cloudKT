import numpy as np
import logging

from utilities import load_module

logger = logging.getLogger(__name__)

def initialize_sightlines(stars, dust, emission_CO, emission_HI, sightline_config, program_directory = None, **kwargs):
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
        if sightline_config["POPULATE_FROM_FILES"] == True:
            if i == 0:
                print('populating from files!')
            star_selection_kwargs = {"fname": program_directory + "/sightline_outputs/stars_{}.fits".format(i)}


        sightline = Sightline(stars, dust, data_processing_kwargs = data_processing_kwargs, star_selection_kwargs = star_selection_kwargs)
        sightlines.append(sightline)

    return sightlines



# def select_from_seed_stars():
#     return

# def select_from_spatial_grid():
#     return


# def previous_version():
#     sightlines = []
#     for i in range(1):
#         if args.populate_from_files == "false":
#             logger.info("Populating sightlines from the California Cloud dataset...")
#             selection_kwargs = {"emission": emission_CO, "vector": (0.02, 0.02)}
#             sightlines.append(Sightline(stars, (164 + i, -8.5), dust, select_stars = (select_stars, selection_kwargs), alternative_data_processing = generateClippedResidual))
        
#             if args.stars_to_files == "true":
#                 if not os.path.exists(program_directory + "/sightline_outputs/"):
#                     os.makedir(program_directory + "/sightline_outputs/")
#                 sightlines[i].stars.write(program_directory + "/sightline_outputs/stars_{}.fits".format(i), overwrite = True)
#         else:
#             logger.info("Populating sighltines from previously-saved .fits files...")
#             selection_kwargs = {"fname": program_directory + "/sightline_outputs/stars_{}.fits".format(i)}
#             sightlines.append(Sightline(stars, (163 + i, -8.5), dust, select_stars = (Sightline.populate_from_file, selection_kwargs)))

#         logstring = "\t Sightline {i}: Nstar = {nstar}, Ndim = {nbin}, Nvar = {nvar}".format(
#                             i = i, nstar = sightlines[i].nsig, nbin = sightlines[i].ndim, nvar = sightlines[i].ndim + sightlines[i].ndim * sightlines[i].nsig)
#         logstring += "; method = {}".format(select_stars)