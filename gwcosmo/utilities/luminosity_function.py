"""
Module with Schechter magnitude function:
(C) Walter Del Pozzo (2014)
Rachel Gray
"""
from numpy import *
from scipy.integrate import quad

class SchechterMagFunction(object):
    """
    The Schechter function (as a function of absolute magnitude and H0)
    """
    
    def __init__(self, Mstar_obs=-19.70, alpha=-1.07, phistar=1.):
        """
        Parameters
        ----------
        Mstar_obs : float, optional
            Observed absolute magnitude, M* - 5log10(h) (default=-19.7)
        alpha : float, optional
            Faint-end slope parameter (default=-1.07)
        phistar : float, optional
            Normalisation factor defining density of galaxies, Mpc^-3 
            This factor cancels in numerator and denominator of the main 
            analysis, including any H0-dependence (default=1.)
        """
        
        self.Mstar_obs = Mstar_obs
        self.phistar = phistar
        self.alpha = alpha
        self.norm = None

    def evaluate(self, m, H0):
        """
        Evaluate the Schechter function for a given absolute magnitude and H0

        phi(m)=0.4*ln10*phistar*[10^(0.4(Mstar-m)*(alpha+1))]*exp[-10^(0.4(Mstar-m))]
        
        Parameters
        ----------
        m : float
            Absolute magnitude
        H0 : float
            Hubble constant value in kms-1Mpc-1
        
        Returns
        -------
        float
            phi(m)
        """
        
        Mstar = M_Mobs(H0, self.Mstar_obs)
        return 0.4*log(10.0)*self.phistar \
               * pow(10.0, -0.4*(self.alpha+1.0)*(m-Mstar)) \
               * exp(-pow(10, -0.4*(m-Mstar)))

    def normalise(self, mmin, mmax):
        if self.norm is None:
            self.norm = quad(self.evaluate, mmin, mmax)[0]

    def pdf(self, m):
        return self.evaluate(m)/self.norm
        
    def __call__(self, m, H0):
        return self.evaluate(m,H0)


def M_Mobs(H0, M_obs):
    """
    Given an observed absolute magnitude, returns absolute magnitude
    """
    return M_obs + 5.*log10(H0/100.)


# Distance modulus given luminosity distance
def DistanceModulus(dL):
    """
    Returns distance modulus given luminosity distance

    Parameters
    ----------
    dL : luminosity distance in Mpc

    Returns
    -------
    distance modulus = 5*np.log10(dL)+25
    """
    return 5*log10(dL)+25  # dL has to be in Mpc


def dl_mM(m, M, Kcorr=0.):
    """
    returns luminosity distance in Mpc given
    apparent magnitude and absolute magnitude

    Parameters
    ----------
    m : apparent magnitude
    M : absolute magnitude in the source frame
    Kcorr : (optional) K correction, to convert absolute magnitude from the
        observed band to the source frame band (default=0).  If fluxes are
        bolometric, this should be left as 0. If not, a K correction of 0 is
        only valid at low redshifts.

    Returns
    -------
    Luminosity distance in Mpc
    """
    return 10**(0.2*(m-M-Kcorr-25))


def L_M(M):
    """
    Returns luminosity when given an absolute magnitude.
    The constant used here corresponds to the conversion between bolometric mangitude and luminosity.
    It does not matter for the H0 inference, so please use with care when using with band specific magnitudes.

    Parameters
    ----------
    M : absolute magnitude in the source frame

    Returns
    -------
    Luminosity in Watts
    """
    # TODO: double check use of L0=3.0128e28
    return 3.0128e28*10**(-0.4*M)


def M_mdl(m, dl, Kcorr=0.):
    """
    Returns a source's absolute magnitude given
    apparent magnitude and luminosity distance
    If a K correction is supplied it will be applied

    Parameters
    ----------
    m : apparent magnitude
    dl : luminosity distance in Mpc
    Kcorr : (optional) K correction, to convert absolute magnitude from the
        observed band to the source frame band (default=0).  If fluxes are
        bolometric, this should be left as 0. If not, a K correction of 0 is
        only valid at low redshifts.

    Returns
    -------
    Absolute magnitude in the source frame
    """
    return m - DistanceModulus(dl) - Kcorr


def L_mdl(m, dl, Kcorr=0.):
    """
    Returns luminosity when given apparent magnitude and luminosity distance
    If a K correction is supplied it will be applied

    Parameters
    ----------
    m : apparent magnitude
    dl : luminosity distance in Mpc
    Kcorr : (optional) K correction, to convert absolute magnitude from the
        observed band to the source frame band (default=0).  If fluxes are
        bolometric, this should be left as 0. If not, a K correction of 0 is
        only valid at low redshifts.

    Returns
    -------
    Luminosity in the source frame
    """
    return L_M(M_mdl(m, dl, Kcorr=Kcorr))


