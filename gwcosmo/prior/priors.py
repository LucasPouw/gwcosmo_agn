"""
Priors
Ignacio Magana, Rachel Gray
"""
from __future__ import absolute_import

import numpy as np

from scipy.interpolate import interp1d

import numpy as _np
import copy as _copy
import sys as _sys
from . import custom_math_priors as _cmp

def pH0(H0, prior='log'):
    """
    Returns p(H0)
    The prior probability of H0

    Parameters
    ----------
    H0 : float or array_like
        Hubble constant value(s) in kms-1Mpc-1
    prior : str, optional
        The choice of prior (default='log')
        if 'log' uses uniform in log prior
        if 'uniform' uses uniform prior

    Returns
    -------
    float or array_like
        p(H0)
    """
    if prior == 'uniform':
        return np.ones(len(H0))
    if prior == 'log':
        return 1./H0

class distance_distribution(object):
    def __init__(self, name):
        self.name = name

        if self.name == 'BBH-powerlaw':
            dist = PriorDict(conversion_function=constrain_m1m2)
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=15000)

        if self.name == 'BNS':
            dist = PriorDict(conversion_function=constrain_m1m2)
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=1000)

        if self.name == 'NSBH':
            dist = PriorDict(conversion_function=constrain_m1m2)
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=1000)

        if self.name == 'BBH-constant':
            dist = PriorDict()
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=15000)

        self.dist = dist

    def sample(self, N_samples):
        samples = self.dist.sample(N_samples)
        return samples['luminosity_distance']

    def prob(self, samples):
        return self.dist['luminosity_distance'].prob(samples)
    

class mass_prior(object):
    """
    Wrapper for for managing the priors on source frame masses.
    The prior is factorized as :math:`p(m_1,m_2) \\propto p(m_1)p(m_2|m_1)`

    Parameters
    ----------
    name: str
        Name of the model that you want. Available 'BBH-powerlaw', 'BBH-powerlaw-gaussian'
        'BBH-broken-powerlaw'.
    hyper_params_dict: dict
        Dictionary of hyperparameters for the prior model you want to use. See code lines for more details
    """

    def __init__(self, name, hyper_params_dict):

        self.name = name
        self.hyper_params_dict=_copy.deepcopy(hyper_params_dict)
        dist = {}

        if self.name == 'BBH-powerlaw' or self.name == 'NSBH-powerlaw':
            alpha = hyper_params_dict['alpha']
            beta = hyper_params_dict['beta']
            mmin = hyper_params_dict['mmin']
            mmax = hyper_params_dict['mmax']
            if self.name == 'BBH-powerlaw':
                dist={'mass_1':_cmp.PowerLaw_math(alpha=-alpha,min_pl=mmin,max_pl=mmax),
                     'mass_2':_cmp.PowerLaw_math(alpha=beta,min_pl=mmin,max_pl=mmax)}
            else:
                dist={'mass_1':_cmp.PowerLaw_math(alpha=-alpha,min_pl=mmin,max_pl=mmax),
                     'mass_2':_cmp.PowerLaw_math(alpha=0.0,min_pl=1.0,max_pl=3.0)}

            self.mmin = mmin
            self.mmax = mmax

        elif self.name == 'BBH-powerlaw-gaussian' or self.name == 'NSBH-powerlaw-gaussian':
            alpha = hyper_params_dict['alpha']
            beta = hyper_params_dict['beta']
            mmin = hyper_params_dict['mmin']
            mmax = hyper_params_dict['mmax']

            mu_g = hyper_params_dict['mu_g']
            sigma_g = hyper_params_dict['sigma_g']
            lambda_peak = hyper_params_dict['lambda_peak']

            delta_m = hyper_params_dict['delta_m']

            if self.name == 'BBH-powerlaw-gaussian':
                m1pr = _cmp.PowerLawGaussian_math(alpha=-alpha,min_pl=mmin,max_pl=mmax,lambda_g=lambda_peak
                    ,mean_g=mu_g,sigma_g=sigma_g,min_g=mmin,max_g=mu_g+5*sigma_g)
                m2pr = _cmp.PowerLaw_math(alpha=beta,min_pl=mmin,max_pl=_np.max([mu_g+5*sigma_g,mmax]))
                dist={'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
                      'mass_2':_cmp.SmoothedProb(origin_prob=m2pr,bottom=mmin,bottom_smooth=delta_m)}

            else:
                m1pr = _cmp.PowerLawGaussian_math(alpha=-alpha,min_pl=mmin,max_pl=mmax,lambda_g=lambda_peak
                    ,mean_g=mu_g,sigma_g=sigma_g,min_g=mmin,max_g=mu_g+5*sigma_g)
                m2pr = _cmp.PowerLaw_math(alpha=0.0,min_pl=1.0,max_pl=3.0)
                dist={'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
                      'mass_2':m2pr}


            # TODO Assume that the gaussian peak does not overlap too much with the mmin
            
            self.mmin = mmin
            self.mmax = dist['mass_1'].maximum

        elif self.name == 'BBH-broken-powerlaw' or self.name == 'NSBH-broken-powerlaw':
            alpha_1 = hyper_params_dict['alpha']
            alpha_2 = hyper_params_dict['alpha_2']
            beta = hyper_params_dict['beta']
            mmin = hyper_params_dict['mmin']
            mmax = hyper_params_dict['mmax']
            b =  hyper_params_dict['b']

            delta_m = hyper_params_dict['delta_m']
            
            if self.name == 'BBH-broken-powerlaw':
                m1pr = _cmp.BrokenPowerLaw_math(alpha_1=-alpha_1,alpha_2=-alpha_2,min_pl=mmin,max_pl=mmax,b=b)
                m2pr = _cmp.PowerLaw_math(alpha=beta,min_pl=mmin,max_pl=mmax)

                dist={'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
                      'mass_2':_cmp.SmoothedProb(origin_prob=m2pr,bottom=mmin,bottom_smooth=delta_m)}
            else:
                m1pr = _cmp.BrokenPowerLaw_math(alpha_1=-alpha_1,alpha_2=-alpha_2,min_pl=mmin,max_pl=mmax,b=b)
                m2pr = _cmp.PowerLaw_math(alpha=0.0,min_pl=1.0,max_pl=3.0)

                dist={'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
                      'mass_2':m2pr}

            self.mmin = mmin
            self.mmax = mmax
            
        elif self.name == 'BNS':
            
            dist={'mass_1':_cmp.PowerLaw_math(alpha=0.0,min_pl=1.0,max_pl=3.0),
                  'mass_2':_cmp.PowerLaw_math(alpha=0.0,min_pl=1.0,max_pl=3.0)}

            self.mmin = 1.0
            self.mmax = 3.0
            
        else:
            print('Name not known, aborting')
            _sys.exit()

        self.dist = dist
        
        if self.name.startswith('NSBH'):
            self.mmin=1.0

    def joint_prob(self, ms1, ms2):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        ms1: np.array(matrix)
            mass one in solar masses
        ms2: dict
            mass two in solar masses
        """
        
        if self.name == 'BNS':
            to_ret =self.dist['mass_1'].prob(ms1)*self.dist['mass_2'].prob(ms2)
        else:
            to_ret =self.dist['mass_1'].prob(ms1)*self.dist['mass_2'].conditioned_prob(ms2,self.mmin*_np.ones_like(ms1),ms1)

        return to_ret

    def sample(self, Nsample):
        """
        This method samples from the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        Nsample: int
            Number of samples you want
        """

        vals_m1 = np.random.rand(Nsample)
        vals_m2 = np.random.rand(Nsample)
        
        m1_trials = np.linspace(self.dist['mass_1'].minimum,self.dist['mass_1'].maximum,10000)
        m2_trials = np.linspace(self.dist['mass_2'].minimum,self.dist['mass_2'].maximum,10000)
        
        cdf_m1_trials = self.dist['mass_1'].cdf(m1_trials)
        cdf_m2_trials = self.dist['mass_2'].cdf(m2_trials)
        
        interpo_icdf_m1 = interp1d(cdf_m1_trials,m1_trials)
        interpo_icdf_m2 = interp1d(cdf_m2_trials,m2_trials)
        
        mass_1_samples = interpo_icdf_m1(vals_m1)
        
        if self.name == 'BNS':
            mass_2_samples = interpo_icdf_m2(vals_m2)
            indx = np.where(mass_2_samples>mass_1_samples)[0]
            
            for indx_sw in indx:
                support = mass_1_samples[indx_sw]
                mass_1_samples[indx_sw] = mass_2_samples[indx_sw]
                mass_2_samples[indx_sw] = support
        else:
            mass_2_samples = interpo_icdf_m2(vals_m2*self.dist['mass_2'].cdf(mass_1_samples))
         
        return mass_1_samples, mass_2_samples
