"""
Checking the valid interpolation boundary of parameters given the injections.

Anson Chen
"""

import numpy as np
import copy
import logging


def check_boundary(cosmo, parameter_dict, injections, mass_priors, gravity_model, mass_model):
    """
    WARNING, CURRENTLY UNREVIEWED
    """
    
    logging.warning("This part of gwcosmo is unreviewed: proceed with caution.")

    injections = copy.deepcopy(injections)

    if mass_model == 'BNS':
        mass_priors.mmax = parameter_dict['mmaxns']['value']
    else:
        mass_priors.mmax = parameter_dict['mmaxbh']['value']

    if np.size(parameter_dict['H0']['value'])>1:
        H0_max = parameter_dict['H0']['value'][1]
    else:
        H0_max = parameter_dict['H0']['value']

    if gravity_model == 'GR':

        # highest redshift case
        cosmo.H0 = H0_max

        if max(injections.dldet*H0_max/cosmo.c) > max(cosmo.dlH0overc_z_arr):
            H0_bound = cosmo.c*max(cosmo.dlH0overc_z_arr)/max(injections.dldet)
            raise ValueError('Injection GW distances exceed the LOS redshift prior bound. The maximum H0 higher bound is %.1f.' % H0_bound)

        z_inj_max = cosmo.z_dgw(max(injections.dldet))
        Msmax = max(injections.m1det)/(1+z_inj_max)

        if Msmax < mass_priors.mmax:
            raise ValueError('For the extreme value of H0=%.1f, the source-frame prior becomes inconsistent with the injections at a dL of %.1f Mpc, which is within the detection horizon for the current set up. Please consider a lower M_max for mass prior (e.g. %.1f Msol), or a lower H0_max.'%(H0_max,max(injections.dldet),Msmax+0.1))
        else:
            print('Source-frame prior M_max=%.1f Msol is consistent with the injections at a dL of %.1f Mpc at H0_max=%.1f'%(mass_priors.mmax,max(injections.dldet),H0_max))

    elif gravity_model == 'Xi0_n':
        if np.size(parameter_dict['Xi0']['value'])>1:
            Xi0_min = parameter_dict['Xi0']['value'][0]
        else:
            Xi0_min = parameter_dict['Xi0']['value']
        if np.size(parameter_dict['n']['value'])>1:
            n_max = parameter_dict['n']['value'][1]
        else:
            n_max = parameter_dict['n']['value']

        # highest redshift case
        cosmo_param_dict = {'H0': H0_max, 'Xi0': Xi0_min, 'n': n_max}
        cosmo.update_parameters(cosmo_param_dict)

        if max(injections.dldet*H0_max/cosmo.c) > cosmo.dgw_dL_ratio(cosmo.zmax)*max(cosmo.dlH0overc_z_arr):
            z_ratio = 1/(1+cosmo.zmax)**n_max
            Xi0_bound = (max(injections.dldet*H0_max/cosmo.c)/max(cosmo.dlH0overc_z_arr)-z_ratio) / (1-z_ratio)
            raise ValueError('Injection GW distances exceed the redshift prior bound. For the H0 upper bound of %d and the n upper bound of %.2f, the minimum Xi0 lower bound is %.2f.' %(H0_max, n_max, Xi0_bound+0.01))

        z_inj_max = cosmo.z_dgw(max(injections.dldet))
        Msmax = max(injections.m1det)/(1+z_inj_max)

        if Msmax < mass_priors.mmax:
            raise ValueError('For the extreme values of H0=%.1f, Xi0=%.2f, n=%.2f, the source-frame prior becomes inconsistent with the injections at a dL of %.1f Mpc, which is within the detection horizon for the current set up. Please consider a lower M_max for mass prior (e.g. %.1f Msol), a lower H0_max or a higher Xi0_min.'%(H0_max,Xi0_min,n_max,max(injections.dldet),Msmax+0.1))
        else:
            print('Source-frame prior M_max=%.1f Msol is consistent with the injections at a dL of %.1f Mpc at H0_max=%.1f, Xi0_min=%.2f, n_max=%.2f'%(mass_priors.mmax,max(injections.dldet),H0_max,Xi0_min,n_max))

