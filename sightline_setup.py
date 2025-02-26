import numpy as np
import logging

from utilities import load_module

logger = logging.GetLogger(__name__)

def initialize_sightlines(stars, dust, CO, HI, sightline_config):
    sightlines = []
    
    sightline_module = load_module(sightline_config["MODULE"])
    Sightline = getattr(sightline_module, sightline_config["CLASS"])



    return sightlines


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