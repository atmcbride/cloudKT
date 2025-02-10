# TO DO

## MISC
[] update DEFAULTS.json

## cloudKT


## load_data
### load_stars
    [x] functionalize data cuts in apply_restrictions
[] generalize emission inputs

sightline_model/ 



# DONE 


# Needed
cloudKT.py 
+ main fn 
+ arge parse and configs
+ pipeline

load_data.py
+ load in star data
+ load in dust data 
+ load in emission data

sightline_model/ whole folder
+ gives me a basis to build new models
+ so far, using base_model.py, sightline.py, injection_sightline.py, foreground_sightline.py

For the MCMC functions, I will probably create a class for BayesianModel, which will
contain the models' log-likelihood, log-prior, and log-probability functions. I may instead
be able to add these as expected inputs to the sightline mode objects instead? I need 
to think about this, seems like there'd be advantages and disadvantages to each. 

