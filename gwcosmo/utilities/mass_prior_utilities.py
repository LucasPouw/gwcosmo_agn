"""
Custom functions for handling implementation of range of mass 
prior distributions.

Alexander Papadopoulos
"""
import os
import gwcosmo
import sys
import importlib.util
import bilby
import numpy as np
import gwcosmo.prior.priors as priors



def extract_parameters_from_instance(instance):
    # Get the __init__ method of the class
    init_method = instance.__init__

    # Get the names of parameters from the __init__ method
    parameter_names = list(init_method.__code__.co_varnames[1:])

    # Create a dictionary of parameters and their values
    parameters = {name: getattr(instance, name) for name in parameter_names}

    return parameters

def multipeak_constraint(params):
    # Shallow copy of param dictionary
    converted_params = params.copy()

    # Define dummy constraint prior
    converted_params['peak_constraint'] = params['mu_g_high'] - params['mu_g_low']
    
    return converted_params


def mass_model_selector(model, parser):

    # Assign mass model from paser output or return error
    if model == 'BBH-powerlaw':
        mass_priors = priors.BBH_powerlaw()
    elif model == 'NSBH-powerlaw':
        mass_priors = priors.NSBH_powerlaw()
    elif model == 'BBH-powerlaw-gaussian':
        mass_priors = priors.BBH_powerlaw_gaussian()
    elif model == 'NSBH-powerlaw-gaussian':
        mass_priors = priors.NSBH_powerlaw_gaussian()
    elif model == 'BBH-broken-powerlaw':
        mass_priors = priors.BBH_broken_powerlaw()
    elif model == 'NSBH-broken-powerlaw':
        mass_priors = priors.NSBH_broken_powerlaw()
    elif model == 'BBH-multi-peak-gaussian':
        mass_priors = priors.BBH_multi_peak_gaussian()
    elif model == 'NSBH-multi-peak-gaussian':
        mass_priors = priors.NSBH_multi_peak_gaussian()
    elif model == 'BNS':
        mass_priors = priors.BNS()
    else:
        parser.error('Unrecognized mass model')

    return mass_priors
