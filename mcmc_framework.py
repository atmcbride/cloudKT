import numpy as np
import emcee
import logging

logger = logging.getLogger(__name__)

class MCMC_Framework:
    def __init__(self):
        self.log_likelihood = None
        self.log_priors = []

    # should be assigned functions 

    def intake_log_prior(self, log_prior_function, **log_prior_kwargs):
        self.log_priors.append((log_prior_function, log_prior_kwargs))
        pass

    def intake_log_likelihood(self, log_likelihood_function, **kwargs):
        self.log_likelihood = log_likelihood_function
        self.log_likelihood_kwargs = kwargs
    
    def log_prior(theta):
        log_prior_value = 0
        for item in self.log_priors:
            log_prior_fn, log_prior_kwargs = item
            log_prior_value += log_prior_fn(theta, **log_prior_kwargs)
        return log_prior_value

    def log_probability(self, theta):

        pass    

    def check_function_exists():
        pass


def main():
    mcmc_object = MCMC_Framework()
    mcmc_object.intake_log_prior(lp_fn, a=1, b=2)

    print(mcmc_object.log_priors)

def lp_fn(theta, **kwargs):
    return theta


if __name__ == '__main__':
    main()
