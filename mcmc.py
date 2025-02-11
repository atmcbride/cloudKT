import numpy as np
import emcee
import logging

logger = logging.getLogger(__name__)

class MCMC_Framework:
    def __init__(self, sightline):
        self.log_likelihood = None
        self.log_priors = []

    # should be assigned functions 

    def intake_log_prior(self):
        pass

    def intake_log_likelihood(self):
        pass

    def log_probability(self, theta):
        pass    