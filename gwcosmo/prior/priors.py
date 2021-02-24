"""
Priors
Ignacio Magana, Rachel Gray
"""
from __future__ import absolute_import

import numpy as np
import sys
import copy

from scipy.integrate import quad, dblquad
from scipy.stats import ncx2, norm
from scipy.interpolate import splev, splrep,interp1d
from astropy import constants as const
from astropy import units as u
from bilby.core.prior import Uniform, PowerLaw, PriorDict, Constraint, DeltaFunction
from bilby import gw

import numpy as _np
import copy as _copy
import sys as _sys
from . import custom_math_priors as _cmp

import gwcosmo

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
'''
class mass_sampling(object):
     def __init__(self, name, alpha=1.6, mmin=5, mmax=50, m1=50, m2=50):
        self.name = name
        self.alpha = alpha
        self.mmin = mmin
        self.mmax = mmax
        self.m1 = m1
        self.m2 = m2

     def sample(self, N_samples):
        if self.name == 'BBH-powerlaw':
           m1, m2 =  self.Binary_mass_distribution(name='BBH-powerlaw', N=N_samples, mmin=self.mmin, mmax=self.mmax, alpha=self.alpha)
        elif self.name == 'BNS':
           m1, m2 =  self.Binary_mass_distribution(name='BNS', N=N_samples, mmin=1.0, mmax=3.0, alpha=0.0)
        elif self.name == 'NSBH':
           m1, m2 =  self.Binary_mass_distribution(name='NSBH', N=N_samples, mmin=self.mmin, mmax=self.mmax, alpha=self.alpha)
        return m1, m2

     def Binary_mass_distribution(self, name, N, mmin=5., mmax=40., alpha=1.6):
        """
        Returns p(m1,m2)
        The prior on the mass distribution that follows a power
        law for BBHs.

        Parameters
        ----------
        N : integer
            Number of masses sampled
        mmin : float
            minimum mass
        mmax : float
            maximum mass
        alpha : float
            slope of the power law p(m) = m^-\alpha where alpha > 0

        Returns
        -------
        float or array_like
            m1, m2
        """
        alpha_ = -1*alpha
        u = np.random.rand(N)
        if alpha_ != -1:
           m1 = (u*(mmax**(alpha_+1)-mmin**(alpha_+1)) +
              mmin**(alpha_+1))**(1.0/(alpha_+1))
           print('Powerlaw mass distribution with alpha = ' + str(alpha))
        else:
           m1 = np.exp(u*(np.log(mmax)-np.log(mmin))+np.log(mmin))
           print('Flat in log mass distribution')
        if name== 'NSBH':
           m2 = np.random.uniform(low=1.0, high=3.0, size=N)
        else:
           m2 = np.random.uniform(low=mmin, high=m1)
        return m1, m2



class mass_distribution(object):
    def __init__(self, name, alpha=1.6, mmin=5, mmax=50, m1=50, m2=50):
        self.name = name
        self.alpha = alpha
        self.mmin = mmin
        self.mmax = mmax
        self.m1 = m1
        self.m2 = m2

        dist = {}

        if self.name == 'BBH-powerlaw':

            if self.alpha != 1:
                dist['mass_1'] = lambda m1s: (np.power(m1s,-self.alpha)*(1-self.alpha))/(np.power(self.mmax,1-self.alpha)-np.power(self.mmin,1-self.alpha))
            else:
                dist['mass_1'] = lambda m1s: np.power(m1s,-self.alpha)/(np.log(self.mmax)-np.log(self.mmin))

            dist['mass_2'] = lambda m1s: 1/(m1s-self.mmin)

        if self.name == 'BNS':
            # We assume p(m1,m2)=p(m1)p(m2)
            dist['mass_1'] = lambda m1s: np.ones_like(m1s)/(3-1)
            dist['mass_2'] = lambda m2s: np.ones_like(m2s)/(3-1)

        if self.name == 'NSBH':
            if self.alpha != 1:
                dist['mass_1'] = lambda m1s: (np.power(m1s,-self.alpha)*(1-self.alpha))/(np.power(self.mmax,1-self.alpha)-np.power(self.mmin,1-self.alpha))
            else:
                dist['mass_1'] = lambda m1s: np.power(m1s,-self.alpha)/(np.log(self.mmax)-np.log(self.mmin))

            dist['mass_2'] = lambda m2s: np.ones_like(m2s)/(3-1)

        if self.name == 'BBH-constant':
            dist['mass_1'] = DeltaFunction(self.m1)
            dist['mass_2'] = DeltaFunction(self.m2)

        self.dist = dist

    def joint_prob(self, ms1, ms2):

        if self.name == 'BBH-powerlaw':
            # ms1 is not a bug in mass_2. That depends only on that var

            arr_result = self.dist['mass_1'](ms1)*self.dist['mass_2'](ms1)
            arr_result[(ms1>self.mmax) | (ms2<self.mmin) | (ms1<ms2)]=0

        if self.name == 'BNS':
            # We assume p(m1,m2)=p(m1)p(m2)
            arr_result = self.dist['mass_1'](ms1)*self.dist['mass_2'](ms2)
            arr_result[(ms1>3) | (ms2<1) | (ms1<ms2)]=0

        if self.name == 'NSBH':
            arr_result = self.dist['mass_1'](ms1)*self.dist['mass_2'](ms2)
            arr_result[(ms1>self.mmax) | (ms1<self.mmin) | (ms2<1) | (ms2>3) | (ms1<ms2)]=0

        if self.name == 'BBH-constant':
            arr_result = self.dist['mass_1'].prob(ms1)*self.dist['mass_2'].prob(ms2)

        return arr_result
'''

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
