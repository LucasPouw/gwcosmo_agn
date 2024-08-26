"""
Priors
Ignacio Magana, Rachel Gray, Sergio Vallejo-Peña, Antonio Enea Romano 
"""
from __future__ import absolute_import

import numpy as np

from scipy.interpolate import interp1d
import bilby 
from gwcosmo.utilities.mass_prior_utilities import multipeak_constraint

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

class m_priors(object):
    """
    Parent class with common methods for managing the priors on source frame masses.
    The prior is factorized as :math:`p(m_1,m_2) \\propto p(m_1)p(m_2|m_1)`
    """

    def __init__(self):
        pass

    def update_parameters(self,param_dict):
        """
        Method to dynamically determine attributes in a mass prior class and use these.
        """
        for key, value in param_dict.items():
            setattr(self, key, value)
        self.update_mass_priors()

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

        to_ret = self.mdis['mass_1'].prob(ms1)*self.mdis['mass_2'].conditioned_prob(ms2,self.mmin*np.ones_like(ms1),np.minimum(ms1,self.mmax2))
        #print('ms1',ms1)
        #print('ms2',ms2)
        #print('mmin',self.mmin)
        #print('mmax2',self.mmax2)
        #print('to_ret',to_ret)

        return to_ret
    
    def log_joint_prob(self,ms1, ms2):
        
        to_ret = np.log(self.joint_prob(ms1, ms2))
        to_ret[np.isnan(to_ret)] = -np.inf

        return to_ret

    def sample(self, Nsample):
        """
        *Not used in O4, due to the use of injections instead of Pdet*
        This method samples from the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        Nsample: int
            Number of samples you want
        """

        vals_m1 = np.random.rand(Nsample)
        vals_m2 = np.random.rand(Nsample)

        m1_trials = np.logspace(np.log10(self.mdis['mass_1'].minimum),np.log10(self.mdis['mass_1'].maximum),10000)
        m2_trials = np.logspace(np.log10(self.mdis['mass_2'].minimum),np.log10(self.mdis['mass_2'].maximum),10000)

        cdf_m1_trials = self.mdis['mass_1'].cdf(m1_trials)
        cdf_m2_trials = self.mdis['mass_2'].cdf(m2_trials)

        m1_trials = np.log10(m1_trials)
        m2_trials = np.log10(m2_trials)

        _,indxm1 = np.unique(cdf_m1_trials,return_index=True)
        _,indxm2 = np.unique(cdf_m2_trials,return_index=True)

        interpo_icdf_m1 = interp1d(cdf_m1_trials[indxm1],m1_trials[indxm1],bounds_error=False,fill_value=(m1_trials[0],m1_trials[-1]))
        interpo_icdf_m2 = interp1d(cdf_m2_trials[indxm2],m2_trials[indxm2],bounds_error=False,fill_value=(m2_trials[0],m2_trials[-1]))

        mass_1_samples = 10**interpo_icdf_m1(vals_m1)
        mass_2_samples = 10**interpo_icdf_m2(vals_m2*self.mdis['mass_2'].cdf(mass_1_samples))

        return mass_1_samples, mass_2_samples
    
    @staticmethod
    def grid_constraint(*args):
        pass 

    @staticmethod
    def sampling_constraint(prior_dict):
        return prior_dict

class BBH_powerlaw(m_priors):
    """
    Child class for BBH power law distribution.
    
    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha: Spectral index for the PL of the primary mass distribution
    beta: Spectral index for the PL of the mass ratio distribution

    The default values of the parameters are set to the corresponding median values in the uniform priors reported in 2111.03604

    ************
    NOTE: The spectral indices passed to PowerLaw_math are alpha=-self.alpha, and alpha=self.beta, according to eqs. A8,A10 in 2111.03604
    ************
    
    The method m_priors.update_parameters is used  in the constructor to initialize the objects
    """
    def __init__(self,mminbh=6.0,mmaxbh=125.0,alpha=6.75,beta=4.0):
        super().__init__()

        self.update_parameters(param_dict={'alpha':alpha, 'beta':beta, 'mminbh':mminbh, 'mmaxbh':mmaxbh})               
              
    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects. 
        It sets the maximum value of the primary mass distribution mmax to mmaxbh, 
        the minimum value of the secondary mass distribution mmin to mminbh, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxbh.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''

        self.mmax = self.mmaxbh #Maximum value of m1, used in injections.Injections.update_VT (m_prior.mmax)
        self.mmin = self.mminbh #Minimum value of m2, used in self.joint_prob and in injections.Injections.update_VT (m_prior.mmin) 
        self.mmax2 = self.mmaxbh #Maximum value of m2, used in self.joint_prob

        self.mdis={'mass_1':_cmp.PowerLaw_math(alpha=-self.alpha,min_pl=self.mminbh,max_pl=self.mmaxbh),
                     'mass_2':_cmp.PowerLaw_math(alpha=self.beta,min_pl=self.mminbh,max_pl=self.mmaxbh)}
        

class NSBH_powerlaw(m_priors):
    """
    Child class for NS-BH power law distribution.
    
    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha: Spectral index for the PL of the primary mass distribution
    mminns: Minimum mass of the neutron star distribution
    mmaxns: Maximum mass of the neutron star distribution
    alphans: Spectral index for the PL of the neutron star mass distribution

    The default values of the black hole mass distribution parameters are set to the corresponding median values in the uniform priors reported in 2111.03604
    The default values of the neutron star mass distribution parameters are set to the corresponding values reported in section 4.2 (page 23) in 2111.03604

    ************
    NOTE: The spectral indices passed to PowerLaw_math are alpha=-self.alpha, and alpha=-self.alphans, according to eq. A10 in 2111.03604
    *************   

    The method m_priors.update_parameters is used in the constructor to initialize the objects
    """
    def __init__(self,mminbh=6.0,mmaxbh=125.0,alpha=6.75,mminns=1.0,mmaxns=3.0,alphans=0.0):
        super().__init__()

        self.update_parameters(param_dict={'alpha':alpha, 'mminbh':mminbh, 'mmaxbh':mmaxbh, 'alphans':alphans, 'mminns':mminns, 'mmaxns':mmaxns})
        
    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to mmaxbh, 
        the minimum value of the secondary mass distribution mmin to mminns, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxns.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''

        self.mmax=self.mmaxbh
        self.mmin=self.mminns        
        self.mmax2=self.mmaxns

        self.mdis={'mass_1':_cmp.PowerLaw_math(alpha=-self.alpha,min_pl=self.mminbh,max_pl=self.mmaxbh),
                     'mass_2':_cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns)}
   
class BBH_powerlaw_gaussian(m_priors):
    """
    Child class for BBH power law gaussian distribution.
    
    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha: Spectral index for the PL of the primary mass distribution    
    mu_g: Mean of the Gaussian component in the primary mass distribution
    sigma_g: Width of the Gaussian component in the primary mass distribution
    lambda_g: Fraction of the model in the Gaussian component
    delta_m: Range of mass tapering on the lower end of the mass distribution
    beta: Spectral index for the PL of the mass ratio distribution

    The default values of the parameters are set to the corresponding values reported in section 4.2 (page 23) in 2111.03604

    ************
    NOTE: The spectral indices passed to PowerLawGaussian_math, and PowerLaw_math, are alpha=-self.alpha, and alpha=self.beta, according to eqs. A8,A11 in 2111.03604
    *************   

    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self,mminbh=4.98,mmaxbh=112.5,alpha=3.78,mu_g=32.27,sigma_g=3.88,lambda_g=0.03,delta_m=4.8,beta=0.81):
        super().__init__()
        
        self.update_parameters(param_dict={'alpha':alpha, 'beta':beta, 'mminbh':mminbh, 'mmaxbh':mmaxbh, 'mu_g':mu_g, 'sigma_g':sigma_g, 'lambda_g':lambda_g, 'delta_m':delta_m})

    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum, 
        the minimum value of the secondary mass distribution mmin to mminbh, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxbh.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''
                       
        self.m1pr = _cmp.PowerLawGaussian_math(alpha=-self.alpha,min_pl=self.mminbh,max_pl=self.mmaxbh,lambda_g=self.lambda_g
                    ,mu_g=self.mu_g,sigma_g=self.sigma_g,min_g=self.mminbh,max_g=self.mu_g+5*self.sigma_g)

        # The max of the secondary mass is adapted to the primary mass maximum which is desided byt the Gaussian and PL
        self.m2pr = _cmp.PowerLaw_math(alpha=self.beta,min_pl=self.mminbh,max_pl=np.max([self.mu_g+5*self.sigma_g,self.mmaxbh]))

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,bottom=self.mminbh,bottom_smooth=self.delta_m),
                      'mass_2':_cmp.SmoothedProb(origin_prob=self.m2pr,bottom=self.mminbh,bottom_smooth=self.delta_m)}
       
        # TO DO Add a check on the mu_g - 5 sigma of the gaussian to not overlap with mmin, print a warning
        #if (mu_g - 5*sigma_g)<=mmin:
        #print('Warning, your mean (minuse 5 sigma) of the gaussian component is too close to the minimum mass')

        self.mmax = self.mdis['mass_1'].maximum 
        self.mmin = self.mminbh  
        self.mmax2 = self.mmaxbh

class NSBH_powerlaw_gaussian(m_priors):
    """
    Child class for NS-BH power law gaussian distribution.
    
    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha: Spectral index for the PL of the primary mass distribution    
    mu_g: Mean of the Gaussian component in the primary mass distribution
    sigma_g: Width of the Gaussian component in the primary mass distribution
    lambda_g: Fraction of the model in the Gaussian component    
    delta_m: Range of mass tapering on the lower end of the mass distribution
    mminns: Minimum mass of the neutron star distribution
    mmaxns: Maximum mass of the neutron star distribution
    alphans: Spectral index for the PL of the neutron star mass distribution

    The default values of the parameters are set to the corresponding values reported in section 4.2 (page 23) in 2111.03604

    ************
    NOTE: The spectral indices passed to PowerLawGaussian_math, and PowerLaw_math, are alpha=-self.alpha, and alpha=-self.alphans, according to eqs. A10,A11 in 2111.03604
    *************
        
    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self,mminbh=4.98,mmaxbh=112.5,alpha=3.78,mu_g=32.27,sigma_g=3.88,lambda_g=0.03,delta_m=4.8,mminns=1.0,mmaxns=3.0,alphans=0.0):
        super().__init__()

        self.update_parameters(param_dict={'alpha':alpha, 'mminbh':mminbh, 'mmaxbh':mmaxbh, 'mu_g':mu_g, 'sigma_g':sigma_g, 'lambda_g':lambda_g, 'delta_m':delta_m, 'alphans':alphans, 'mminns':mminns, 'mmaxns':mmaxns})

    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum, 
        the minimum value of the secondary mass distribution mmin to mminns, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxns.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''
                
        self.m1pr = _cmp.PowerLawGaussian_math(alpha=-self.alpha,min_pl=self.mminbh,max_pl=self.mmaxbh,lambda_g=self.lambda_g
                    ,mu_g=self.mu_g,sigma_g=self.sigma_g,min_g=self.mminbh,max_g=self.mu_g+5*self.sigma_g)

        # The max of the secondary mass is adapted to the primary mass maximum which is desided byt the Gaussian and PL
        self.m2pr = _cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,bottom=self.mminbh,bottom_smooth=self.delta_m),
                      'mass_2':self.m2pr}

        self.mmax = self.mdis['mass_1'].maximum 
        self.mmin = self.mminns  
        self.mmax2 = self.mmaxns

class BBH_broken_powerlaw(m_priors):
    """
    Child class for BBH broken power law distribution.

    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha_1: PL slope of the primary mass distribution for masses below mbreak 
    alpha_2: PL slope for the primary mass distribution for masses above mbreak 
    b: The fraction of the way between mminbh and mmaxbh at which the primary mass distribution breaks
    delta_m: Range of mass tapering on the lower end of the mass distribution
    beta: Spectral index for the PL of the mass ratio distribution

    The default values of the parameters are set to the corresponding median values in the uniform priors reported in 2111.03604
    
    ************
    NOTE: The spectral indices passed to BrokenPowerLaw_math, and PowerLaw_math, are alpha_1=-self.alpha_1, alpha_2=-self.alpha_2, and alpha=self.beta, according to eqs. A8,A12 in 2111.03604
    ************

    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self,mminbh=26,mmaxbh=125,alpha_1=6.75,alpha_2=6.75,b=0.5,delta_m=5,beta=4):
        super().__init__()

        self.update_parameters(param_dict={'alpha_1':alpha_1, 'alpha_2':alpha_2, 'beta':beta, 'mminbh':mminbh, 'mmaxbh':mmaxbh, 'b':b, 'delta_m':delta_m})

    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects. 
        It sets the maximum value of the primary mass distribution mmax to mmaxbh, 
        the minimum value of the secondary mass distribution mmin to mminbh, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxbh.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''

        self.mmax = self.mmaxbh 
        self.mmin = self.mminbh  
        self.mmax2 = self.mmaxbh
                
        self.m1pr = _cmp.BrokenPowerLaw_math(alpha_1=-self.alpha_1,alpha_2=-self.alpha_2,min_pl=self.mminbh,max_pl=self.mmaxbh,b=self.b)
        self.m2pr = _cmp.PowerLaw_math(alpha=self.beta,min_pl=self.mminbh,max_pl=self.mmaxbh)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,bottom=self.mminbh,bottom_smooth=self.delta_m),
                      'mass_2':_cmp.SmoothedProb(origin_prob=self.m2pr,bottom=self.mminbh,bottom_smooth=self.delta_m)}

class NSBH_broken_powerlaw(m_priors):
    """
    Child class for NS-BH broken power law distribution.
    
    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the black hole mass distribution
    alpha_1: PL slope of the primary mass distribution for masses below mbreak 
    alpha_2: PL slope for the primary mass distribution for masses above mbreak 
    b: The fraction of the way between mminbh and mmaxbh at which the primary mass distribution breaks
    delta_m: Range of mass tapering on the lower end of the mass distribution
    mminns: Minimum mass of the neutron star distribution
    mmaxns: Maximum mass of the neutron star distribution
    alphans: Spectral index for the PL of the neutron star mass distribution

    The default values of the black hole mass distribution parameters are set to the corresponding median values in the uniform priors reported in 2111.03604
    The default values of the neutron star mass distribution parameters are set to the corresponding values reported in section 4.2 (page 23) in 2111.03604

    ************
    NOTE: The spectral indices passed to BrokenPowerLaw_math, and PowerLaw_math, are alpha_1=-self.alpha_1, alpha_2=-self.alpha_2, and alpha=-self.alphans, according to eqs. A10,A12 in 2111.03604
    ************
    
    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self,mminbh=26,mmaxbh=125,alpha_1=6.75,alpha_2=6.75,b=0.5,delta_m=5,mminns=1.0,mmaxns=3.0,alphans=0.0):
        super().__init__()

        self.update_parameters(param_dict={'alpha_1':alpha_1, 'alpha_2':alpha_2, 'mminbh':mminbh, 'mmaxbh':mmaxbh, 'b':b, 'delta_m':delta_m, 'alphans':alphans, 'mminns':mminns, 'mmaxns':mmaxns})

    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to mmaxbh, 
        the minimum value of the secondary mass distribution mmin to mminns, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxns.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''
        
        self.mmax=self.mmaxbh
        self.mmin=self.mminns        
        self.mmax2=self.mmaxns
                
        self.m1pr = _cmp.BrokenPowerLaw_math(alpha_1=-self.alpha_1,alpha_2=-self.alpha_2,min_pl=self.mminbh,max_pl=self.mmaxbh,b=self.b)
        self.m2pr = _cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,bottom=self.mminbh,bottom_smooth=self.delta_m),
                      'mass_2':self.m2pr}
             

class BBH_multi_peak_gaussian(m_priors):
    """
    Child class for BBH with powerlaw component and two gaussian peaks.

    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha: Spectral index for the PL of the primary mass distribution    
    mu_g_low: Mean of the lower mass Gaussian component in the primary mass distribution
    sigma_g_low: Width of the lower mass Gaussian component in the primary mass distribution
    mu_g_high: Mean of the higher mass Gaussian component in the primary mass distribution
    sigma_g_high: Width of the higher mass Gaussian component in the primary mass distribution
    lambda_g: Fraction of the model in the Gaussian component
    lambda_g_low: Fraction of the Gaussian component in the lower mass peak
    delta_m: Range of mass tapering on the lower end of the mass distribution
    beta: Spectral index for the PL of the mass ratio distribution

    ************
    NOTE: The spectral indices passed to PowerLawDoubleGaussian_math, and PowerLaw_math, are alpha=-self.alpha, and alpha=self.beta, according to eqs. A8,A11 in 2111.03604
    ************* 
    
    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self,alpha=3.78,beta=0.8,mminbh=4.98,mmaxbh=112.5,lambda_g=0.03,lambda_g_low= 0.5,mu_g_low=10.5,sigma_g_low=3.88,mu_g_high=32.27,sigma_g_high=5,delta_m=5):
        super().__init__()

        self.update_parameters(param_dict={'alpha':alpha,'beta':beta,'mminbh':mminbh,'mmaxbh':mmaxbh,'lambda_g':lambda_g,'lambda_g_low':lambda_g_low,'mu_g_low':mu_g_low,'sigma_g_low':sigma_g_low,'mu_g_high':mu_g_high,'sigma_g_high':sigma_g_high,'delta_m':delta_m})

    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum, 
        the minimum value of the secondary mass distribution mmin to mminbh, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxbh.
        It's called by update_parameters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''

        self.m1pr =_cmp.PowerLawDoubleGaussian_math(alpha=-self.alpha,min_pl=self.mminbh,max_pl=self.mmaxbh,lambda_g=self.lambda_g,lambda_g_low=self.lambda_g_low,
                                                    mu_g_low=self.mu_g_low,sigma_g_low=self.sigma_g_low,mu_g_high=self.mu_g_high,sigma_g_high=self.sigma_g_high,min_g=self.mminbh,max_g=self.mu_g_high+5*self.sigma_g_high)
        

        self.m2pr =_cmp.PowerLaw_math(alpha=self.beta,min_pl=self.mminbh,max_pl=np.max([self.mu_g_low+5*self.sigma_g_low,self.mmaxbh]))

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,bottom=self.mminbh,bottom_smooth=self.delta_m),
                      'mass_2':_cmp.SmoothedProb(origin_prob=self.m2pr,bottom=self.mminbh,bottom_smooth=self.delta_m)}

        self.mmax = self.mdis['mass_1'].maximum
        self.mmin = self.mminbh
        self.mmax2 = self.mmaxbh

    @staticmethod
    def grid_constraint(likelihood, values, parameter_grid, fixed_params):
        """
        Function to remove parameter space that does not follow constraint for multi-peak mass prior in gridded method
        """
        
        # constraint for mu_g_low sampling and mu_g_high fixed
        if ('mu_g_low' in parameter_grid.keys()) & ('mu_g_high' in fixed_params.keys()):
            if np.max(parameter_grid['mu_g_low']) > fixed_params['mu_g_high']:
                raise ValueError(f"Value for maximum lower peak {np.max(parameter_grid['mu_g_low'])} is greater than fixed upper peak {fixed_params['mu_g_high']}.")
        # constraint for mu_g_low fixed and mu_g_high sampling
        elif ('mu_g_high' in parameter_grid.keys()) & ('mu_g_low' in fixed_params.keys()):
            if np.min(parameter_grid['mu_g_high']) < fixed_params['mu_g_low']:
                raise ValueError(f"Value for minimum upper peak {np.min(parameter_grid['mu_g_high'])} is less than fixed lower peak {fixed_params['mu_g_low']}.")
        # constraint for mu_g_low fixed and mu_g_high fixed
        elif ('mu_g_low' in fixed_params.keys()) & ('mu_g_high' in fixed_params.keys()): 
            if fixed_params['mu_g_low'] > fixed_params['mu_g_high']:
                raise ValueError(f"Value for lower peak {fixed_params['mu_g_low']} is greater than upper peak {fixed_params['mu_g_high']}.")
        else:
            names  = list(parameter_grid.keys())
            idx_low = names.index('mu_g_low')
            idx_high = names.index('mu_g_high')
            mask = [x[idx_high] < x[idx_low] for x in values]
            shape = likelihood.shape
            reshaped_mask = np.array(mask).reshape(shape)
            likelihood[reshaped_mask] = -np.inf

    @staticmethod
    def sampling_constraint(prior_dict):
        """
        Function to set up constraint prior dictionary for multi-peak mass prior in sampling method
        """

        # constraint for mu_g_low sampling and mu_g_high fixed
        if (type(prior_dict['mu_g_low']) in {bilby.core.prior.Uniform, bilby.core.prior.Gaussian}) & (type(prior_dict['mu_g_high']) == float):
            if prior_dict['mu_g_low'].maximum > prior_dict['mu_g_high']:
                raise ValueError(f"Value for maximum lower peak {prior_dict['mu_g_low'].maximum} is greater than fixed upper peak {prior_dict['mu_g_high']}.")
        # constraint for mu_g_low fixed and mu_g_high sampling
        elif (type(prior_dict['mu_g_high']) in {bilby.core.prior.Uniform, bilby.core.prior.Gaussian}) & (type(prior_dict['mu_g_low']) == float):
            if prior_dict['mu_g_low'] > prior_dict['mu_g_high'].minimum:
                raise ValueError(f"Value for minimum upper peak {prior_dict['mu_g_high'].minimum} is lower than fixed lower peak {prior_dict['mu_g_low']}.")
        # constraint for mu_g_low fixed and mu_g_high fixed
        elif(type(prior_dict['mu_g_high']) == float) & (type(prior_dict['mu_g_low']) == float): 
            if prior_dict['mu_g_low'] > prior_dict['mu_g_high']:
                raise ValueError(f"Value for lower peak {prior_dict['mu_g_low']} is greater than upper peak {prior_dict['mu_g_high']}.")
        # constraint for mu_g_low sampling and mu_g_high sampling with constrained prior
        else :
            prior_dict = bilby.core.prior.PriorDict(prior_dict, conversion_function = multipeak_constraint)
            prior_dict['peak_constraint'] = bilby.core.prior.Constraint(minimum = 0, maximum = 5000)   
        return prior_dict


class NSBH_multi_peak_gaussian(m_priors):
    """
    Child class for NS-BH with powerlaw component and two gaussian peaks.

    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha: Spectral index for the PL of the primary mass distribution    
    mu_g_low: Mean of the lower mass Gaussian component in the primary mass distribution
    sigma_g_low: Width of the lower mass Gaussian component in the primary mass distribution
    mu_g_high: Mean of the higher mass Gaussian component in the primary mass distribution
    sigma_g_high: Width of the higher mass Gaussian component in the primary mass distribution
    lambda_g: Fraction of the model in the Gaussian component
    lambda_g_low: Fraction of the Gaussian component in the lower mass peak
    delta_m: Range of mass tapering on the lower end of the mass distribution
    mminns: Minimum mass of the neutron star distribution
    mmaxns: Maximum mass of the neutron star distribution
    alphans: Spectral index for the PL of the neutron star mass distribution

    ************
    NOTE: The spectral indices passed to PowerLawDoubleGaussian_math, and PowerLaw_math, are alpha=-self.alpha, and alpha=-self.alphans, according to eqs. A10,A11 in 2111.03604
    *************
    
    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """

    def __init__(self,alpha=3.78,mminbh=4.98,mmaxbh=112.5,lambda_g=0.03,lambda_g_low= 0.5,mu_g_low=10.5,sigma_g_low=3.88,mu_g_high=32.27,sigma_g_high=5,delta_m=5,mminns=1.0,mmaxns=3.0,alphans=0):
        super().__init__()

        self.update_parameters(param_dict={'alpha':alpha,'mminbh':mminbh,'mmaxbh':mmaxbh,'lambda_g':lambda_g,'lambda_g_low':lambda_g_low,'mu_g_low':mu_g_low,'sigma_g_low':sigma_g_low,'mu_g_high':mu_g_high,'sigma_g_high':sigma_g_high,'delta_m':delta_m,'mminns':mminns,'mmaxns':mmaxns,'alphans':alphans})

    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum, 
        the minimum value of the secondary mass distribution mmin to mminns, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxns.
        It's called by update_parameters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''


        self.m1pr =_cmp.PowerLawDoubleGaussian_math(alpha=-self.alpha,min_pl=self.mminbh,max_pl=self.mmaxbh,lambda_g=self.lambda_g,lambda_g_low=self.lambda_g_low,
                                                    mu_g_low=self.mu_g_low,sigma_g_low=self.sigma_g_low,mu_g_high=self.mu_g_high,sigma_g_high=self.sigma_g_high,min_g=self.mminbh,max_g=self.mu_g_high+5*self.sigma_g_high)
        

        self.m2pr = _cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,bottom=self.mminbh,bottom_smooth=self.delta_m),
                      'mass_2': self.m2pr}

        self.mmax = self.mdis['mass_1'].maximum
        self.mmin = self.mminns
        self.mmax2 = self.mmaxns

    @staticmethod
    def grid_constraint(constraint_grid, values, parameter_grid, fixed_params):
        """
        Function to remove parameter space that does not follow constraint for multi-peak mass prior in gridded method
        """
        
        # constraint for mu_g_low sampling and mu_g_high fixed
        if ('mu_g_low' in parameter_grid.keys()) & ('mu_g_high' in fixed_params.keys()):
            if np.max(parameter_grid['mu_g_low']) > fixed_params['mu_g_high']:
                raise ValueError(f"Value for maximum lower peak {np.max(parameter_grid['mu_g_low'])} is greater than fixed upper peak {fixed_params['mu_g_high']}.")
        # constraint for mu_g_low fixed and mu_g_high sampling
        elif ('mu_g_high' in parameter_grid.keys()) & ('mu_g_low' in fixed_params.keys()):
            if np.min(parameter_grid['mu_g_high']) < fixed_params['mu_g_low']:
                raise ValueError(f"Value for minimum upper peak {np.min(parameter_grid['mu_g_high'])} is less than fixed lower peak {fixed_params['mu_g_low']}.")
        # constraint for mu_g_low fixed and mu_g_high fixed
        elif ('mu_g_low' in fixed_params.keys()) & ('mu_g_high' in fixed_params.keys()): 
            if fixed_params['mu_g_low'] > fixed_params['mu_g_high']:
                raise ValueError(f"Value for lower peak {fixed_params['mu_g_low']} is greater than upper peak {fixed_params['mu_g_high']}.")
        else:
            names  = list(parameter_grid.keys())
            idx_low = names.index('mu_g_low')
            idx_high = names.index('mu_g_high')
            mask = [x[idx_high] < x[idx_low] for x in values]
            shape = constraint_grid.shape
            reshaped_mask = np.array(mask).reshape(shape)
            constraint_grid[reshaped_mask] = -np.inf

        return constraint_grid
    
    @staticmethod
    def sampling_constraint(prior_dict):
        """
        Function to set up constraint prior dictionary for multi-peak mass prior in sampling method
        """

        # constraint for mu_g_low sampling and mu_g_high fixed
        if (type(prior_dict['mu_g_low']) in {bilby.core.prior.Uniform, bilby.core.prior.Gaussian}) & (type(prior_dict['mu_g_high']) == float):
            if prior_dict['mu_g_low'].maximum > prior_dict['mu_g_high']:
                raise ValueError(f"Value for maximum lower peak {prior_dict['mu_g_low'].maximum} is greater than fixed upper peak {prior_dict['mu_g_high']}.")
        # constraint for mu_g_low fixed and mu_g_high sampling
        elif (type(prior_dict['mu_g_high']) in {bilby.core.prior.Uniform, bilby.core.prior.Gaussian}) & (type(prior_dict['mu_g_low']) == float):
            if prior_dict['mu_g_low'] > prior_dict['mu_g_high'].minimum:
                raise ValueError(f"Value for minimum upper peak {prior_dict['mu_g_high'].minimum} is lower than fixed lower peak {prior_dict['mu_g_low']}.")
        # constraint for mu_g_low fixed and mu_g_high fixed
        elif(type(prior_dict['mu_g_high']) == float) & (type(prior_dict['mu_g_low']) == float): 
            if prior_dict['mu_g_low'] > prior_dict['mu_g_high']:
                raise ValueError(f"Value for lower peak {prior_dict['mu_g_low']} is greater than upper peak {prior_dict['mu_g_high']}.")
        # constraint for mu_g_low sampling and mu_g_high sampling with constrained prior
        else :
            prior_dict = bilby.core.prior.PriorDict(prior_dict, conversion_function = multipeak_constraint)
            prior_dict['peak_constraint'] = bilby.core.prior.Constraint(minimum = 0, maximum = 5000)   
        return prior_dict

class BNS(m_priors):
    """
    Child class for BNS distribution.
    
    Parameters
    -----------
    mminns: Minimum mass of the neutron star distribution
    mmaxns: Maximum mass of the neutron star distribution
    alphans: Spectral index for the PL of the neutron star mass distribution

    The default values of the parameters are set to the corresponding values reported in section 4.2 (page 23) in 2111.03604

    ************
    NOTE: The spectral index passed to PowerLaw_math is alpha=-self.alphans according to eq. A10 in 2111.03604
    ************
    
    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self,mminns=1.0,mmaxns=3.0,alphans=0.0):
        super().__init__()

        self.update_parameters(param_dict={'alphans':alphans, 'mminns':mminns, 'mmaxns':mmaxns})
                    
    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to mmaxns, 
        and the minimum value of the secondary mass distribution mmin to mminns.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax and mmin definitions depend on the mass prior model.          
        '''

        self.mmax = self.mmaxns
        self.mmin = self.mminns        

        self.mdis={'mass_1':_cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns),
                  'mass_2':_cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns)}
        
    
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

        to_ret =self.mdis['mass_1'].prob(ms1)*self.mdis['mass_2'].prob(ms2)

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

        m1_trials = np.logspace(np.log10(self.mdis['mass_1'].minimum),np.log10(self.mdis['mass_1'].maximum),10000)
        m2_trials = np.logspace(np.log10(self.mdis['mass_2'].minimum),np.log10(self.mdis['mass_2'].maximum),10000)

        cdf_m1_trials = self.mdis['mass_1'].cdf(m1_trials)
        cdf_m2_trials = self.mdis['mass_2'].cdf(m2_trials)

        m1_trials = np.log10(m1_trials)
        m2_trials = np.log10(m2_trials)

        _,indxm1 = np.unique(cdf_m1_trials,return_index=True)
        _,indxm2 = np.unique(cdf_m2_trials,return_index=True)

        interpo_icdf_m1 = interp1d(cdf_m1_trials[indxm1],m1_trials[indxm1],bounds_error=False,fill_value=(m1_trials[0],m1_trials[-1]))
        interpo_icdf_m2 = interp1d(cdf_m2_trials[indxm2],m2_trials[indxm2],bounds_error=False,fill_value=(m2_trials[0],m2_trials[-1]))

        mass_1_samples = 10**interpo_icdf_m1(vals_m1)
        mass_2_samples = 10**interpo_icdf_m2(vals_m2)

        indx = np.where(mass_2_samples>mass_1_samples)[0]
        mass_1_samples[indx],mass_2_samples[indx] = mass_2_samples[indx],mass_1_samples[indx]

        return mass_1_samples, mass_2_samples
        


 