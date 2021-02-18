'''
This module collects analytical and numerical probability density functions.
'''

import numpy as _np
from scipy.stats import truncnorm as _truncnorm
import copy as _copy
from scipy.interpolate import interp1d as _interp1d
import bilby as _bilby

def _S_factor(mass, mmin,delta_m):

    if not isinstance(mass,_np.ndarray):
        mass = _np.array([mass])

    to_ret = _np.ones_like(mass)
    if delta_m == 0:
        return to_ret

    mprime = mass-mmin

    select_window = (mass>mmin) & (mass<(delta_m+mmin))
    select_one = mass>=(delta_m+mmin)
    select_zero = mass<=mmin

    effe_prime = _np.zeros_like(mass)
    effe_prime[select_window] = _np.exp(_np.nan_to_num((delta_m/mprime[select_window])+(delta_m/(mprime[select_window]-delta_m))))
    to_ret = 1./(effe_prime+1)
    to_ret[select_zero]=0.
    to_ret[select_one]=1.
    return to_ret

def get_PL_norm(alpha,min,max):
    '''
    This function returns the powerlaw normalization factor

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    '''

    if alpha == -1:
        return _np.log(max/min)
    else:
        return (_np.power(max,alpha+1) - _np.power(min,alpha+1))/(alpha+1)

class SmoothedProb(object):
    '''
    Class for smoothing the low part of a PDF. The smoothing follows Eq. B7 of
    2010.14533.

    Parameters
    ----------
    origin_prob: class
        Original prior class to smooth from this module
    bottom: float
        minimum cut-off. Below this, the window is 0.
    bottom_smooth: float
        smooth factor. The smoothing acts between bottom and bottom+bottom_smooth
    '''

    def __init__(self,origin_prob,bottom,bottom_smooth):

        self.origin_prob = _copy.deepcopy(origin_prob)
        self.bottom_smooth = bottom_smooth
        self.bottom = bottom
        self.maximum=self.origin_prob.maximum
        self.minimum=self.origin_prob.minimum

        int_array = _np.linspace(self.origin_prob.minimum,bottom+bottom_smooth,1000)
        integral_before = _np.trapz(self.origin_prob.prob(int_array),int_array)
        integral_now = _np.trapz(self.prob(int_array),int_array)
        self.norm = 1 - integral_before + integral_now

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """
        window = _S_factor(x, self.bottom,self.bottom_smooth)

        if hasattr(self,'norm'):
            prob_ret =self.origin_prob.prob(x)*window/self.norm
        else:
            prob_ret =self.origin_prob.prob(x)*window

        return prob_ret

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        if hasattr(self,'cached_cdf'):
            to_ret = self.cached_cdf(x)
        else:
            eval_array = _np.linspace(self.minimum,self.maximum,1000)
            prob_eval = self.prob(eval_array)
            dx = eval_array[1]-eval_array[0]
            cumulative_disc = _np.cumsum(prob_eval)*dx
            cumulative_disc[0]=0
            cumulative_disc[-1]=1
            self.cached_cdf=_interp1d(eval_array,cumulative_disc,bounds_error=False,fill_value=(0,1))
            to_ret = self.cached_cdf(x)

        return to_ret

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        to_ret = self.prob(x)
        new_norm = self.cdf(b)-self.cdf(a)

        new_norm[new_norm==0]=1

        to_ret/=new_norm
        to_ret *= (x<=b) & (x>=a)

        return to_ret


class PowerLaw_math(object):
    """
    Class for a powerlaw probability :math:`p(x) \\propto x^{\\alpha}` defined in
    [a,b]

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    """

    def __init__(self,alpha,min_pl,max_pl):

        self.minimum = min_pl
        self.maximum = max_pl
        self.min_pl = min_pl
        self.max_pl = max_pl
        self.alpha = alpha
        self.norm = get_PL_norm(alpha,min_pl,max_pl)

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        to_ret = _np.power(x,self.alpha)/self.norm
        to_ret *= (x<=self.max_pl) & (x>=self.min_pl)

        return to_ret

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        if hasattr(self,'cached_cdf'):
            to_ret = self.cached_cdf(x)
        else:
            eval_array = _np.linspace(self.minimum,self.maximum,1000)
            prob_eval = self.prob(eval_array)
            dx = eval_array[1]-eval_array[0]
            cumulative_disc = _np.cumsum(prob_eval)*dx
            cumulative_disc[0]=0
            cumulative_disc[-1]=1
            self.cached_cdf=_interp1d(eval_array,cumulative_disc,bounds_error=False,fill_value=(0,1))
            to_ret = self.cached_cdf(x)

        return to_ret

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        to_ret = self.prob(x)
        new_norm = self.cdf(b)-self.cdf(a)

        new_norm[new_norm==0]=1

        to_ret/=new_norm
        to_ret *= (x<=b) & (x>=a)

        return to_ret


class PowerLawGaussian_math(object):
    """
    Class for a powerlaw probability plus gausian peak
    :math:`p(x) \\propto (1-\\lambda)x^{\\alpha}+\\lambda \\mathcal{N}(\\mu,\\sigma)`. Each component is defined in
    a different interval

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    lambda_g: float
        fraction of prob coming from gaussian peak
    mean_g: float
        mean for the gaussian
    sigma_g: float
        standard deviation for the gaussian
    min_g: float
        minimum for the gaussian component
    max_g: float
        maximim for the gaussian component
    """

    def __init__(self,alpha,min_pl,max_pl,lambda_g,mean_g,sigma_g,min_g,max_g):

        self.minimum = _np.min([min_pl,min_g])
        self.maximum = _np.min([max_pl,max_g])

        self.min_pl = min_pl
        self.max_pl = max_pl
        self.alpha = alpha
        self.norm_PL = get_PL_norm(alpha,min_pl,max_pl)

        self.lambda_g = lambda_g
        self.mean_g=mean_g
        self.sigma_g=sigma_g
        self.min_g=min_g
        self.max_g=max_g

        #a, b = (self.min_g - self.mean_g) / self.sigma_g, (self.max_g - self.mean_g) / self.sigma_g
        #self.gg=_truncnorm(a,b,loc=self.mean_g,scale=self.sigma_g)
        self.gg = _bilby.core.prior.TruncatedGaussian(mu=self.mean_g,sigma=self.sigma_g,
        minimum=self.min_g,maximum=self.max_g)


    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        pl_part = (1-self.lambda_g)*_np.power(x,self.alpha)/self.norm_PL
        pl_part *= (x<=self.max_pl) & (x>=self.min_pl)
        #g_part =self.gg.pdf(x)*self.lambda_g
        g_part =self.gg.prob(x)*self.lambda_g

        return pl_part+g_part

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        if hasattr(self,'cached_cdf'):
            to_ret = self.cached_cdf(x)
        else:
            eval_array = _np.linspace(self.minimum,self.maximum,1000)
            prob_eval = self.prob(eval_array)
            dx = eval_array[1]-eval_array[0]
            cumulative_disc = _np.cumsum(prob_eval)*dx
            cumulative_disc[0]=0
            cumulative_disc[-1]=1
            self.cached_cdf=_interp1d(eval_array,cumulative_disc,bounds_error=False,fill_value=(0,1))
            to_ret = self.cached_cdf(x)

        return to_ret

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        to_ret = self.prob(x)
        new_norm = self.cdf(b)-self.cdf(a)

        new_norm[new_norm==0]=1

        to_ret/=new_norm
        to_ret *= (x<=b) & (x>=a)

        return to_ret



class BrokenPowerLaw_math(object):
    """
    Class for a broken powerlaw probability
    :math:`p(x) \\propto x^{\\alpha}` if :math:`min<x<b(max-min)`, :math:`p(x) \\propto x^{\\beta}` if :math:`b(max-min)<x<max`.

    Parameters
    ----------
    alpha_1: float
        Powerlaw slope for first component
    alpha_2: float
        Powerlaw slope for second component
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    b: float
        fraction in [0,1] at which the powerlaw breaks
    """

    def __init__(self,alpha_1,alpha_2,min_pl,max_pl,b):

        self.minimum = min_pl
        self.maximum = max_pl

        self.min_pl = min_pl
        self.max_pl = max_pl
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.break_point = min_pl+b*(max_pl-min_pl)
        self.b=b

        self.norm = get_PL_norm(alpha_1,min_pl,self.break_point) + _np.power(self.break_point,self.alpha_1-self.alpha_2)\
        *get_PL_norm(alpha_2,self.break_point,max_pl)

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        pl_1 = _np.power(x,self.alpha_1)
        pl_2 = _np.power(x,self.alpha_2)*_np.power(self.break_point,self.alpha_1-self.alpha_2)

        pl_1 *= (x<=self.break_point) & (x>=self.min_pl)
        pl_2 *= (x<=self.max_pl) & (x>self.break_point)

        pl = (pl_1+pl_2)/self.norm

        return pl

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        if hasattr(self,'cached_cdf'):
            to_ret = self.cached_cdf(x)
        else:
            eval_array = _np.linspace(self.minimum,self.maximum,1000)
            prob_eval = self.prob(eval_array)
            dx = eval_array[1]-eval_array[0]
            cumulative_disc = _np.cumsum(prob_eval)*dx
            cumulative_disc[0]=0
            cumulative_disc[-1]=1
            self.cached_cdf=_interp1d(eval_array,cumulative_disc,bounds_error=False,fill_value=(0,1))
            to_ret = self.cached_cdf(x)

        return to_ret

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        to_ret = self.prob(x)
        new_norm = self.cdf(b)-self.cdf(a)

        new_norm[new_norm==0]=1

        to_ret/=new_norm
        to_ret *= (x<=b) & (x>=a)

        return to_ret
