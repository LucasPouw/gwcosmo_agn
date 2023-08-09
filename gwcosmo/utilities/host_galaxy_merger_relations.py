"""
Classes to dictate the relationship between GW mergers and host galaxy
properties such as redshift and intrinsic luminosity

Rachel Gray
"""

from .luminosity_function import L_M


class UniformWeighting(object):
    """
    Host galaxy probability relation to luminosity: uniform
    """

    def __init__(self):
        self.luminosity_weights = False

    def model(self, M):
        """
        Uniform weighting

        Parameters
        ----------
        M : float
            absolute magnitude

        Returns
        -------
        float
            1.
        """
        return 1.

    def __call__(self, M):
        return self.model(M)


class LuminosityWeighting(UniformWeighting):
    """
    Host galaxy probability relation to luminosity: proportional to luminosity
    """

    def __init__(self):
        self.luminosity_weights = True

    def model(self, M):
        """
        Luminosity weighting

        Parameters
        ----------
        M : float
            absolute magnitude

        Returns
        -------
        float
            Luminosity
        """

        return L_M(M)



class RedshiftEvolutionConstant(object):
    """
    Parent class for CBC merger rate evolution with redshift, p(s|z)
    
    Each redshift evolution child class must have function called 'model'
    which takes only redshift as an input parameter. All model hyperparameters
    should be defined as attributes to the class
    """
        
    def model(self,z):
        """
        No rate evolution

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
        """
        return 1.
        
    def __call__(self,z):
        return self.model(z)/(1.+z)


class RedshiftEvolutionMadau(RedshiftEvolutionConstant):
    """
    Merger rate relation to redshift: Madau model
    Equation 2 in https://arxiv.org/pdf/2003.12152.pdf
    Default values are taken from https://arxiv.org/abs/2111.03604
    """
    
    def __init__(self, gamma=4.59, k=2.86, zp=2.47):
    
        """
        Parameters
        ----------
        gamma : float
            Powerlaw index for initial (low z) merger rate, R(z)∝(1+z)^gamma (default=4.59)
        k : float
            Powerlaw index for later (high z) merger rate, R(z)∝(1+z)^-k (default=2.86)
        zp : float
            Peak of merger rate (default=2.47)
        """

        self.gamma = gamma
        self.k = k
        self.zp = zp

    def model(self, z):
        """
        Madau rate evolution

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
        """

        C = 1+(1+self.zp)**(-self.gamma-self.k)
        return C*((1+z)**self.gamma)/(1+((1+z)/(1+self.zp))**(self.gamma+self.k))


class RedshiftEvolutionPowerLaw(RedshiftEvolutionConstant):
    """
    Merger rate relation to redshift: power-law model
    Default value taken from https://arxiv.org/abs/2111.03604
    """

    def __init__(self, gamma=4.59):
    
        """
        Parameters
        ----------
        gamma : float
            Powerlaw index for merger rate, R(z)∝(1+z)^gamma (default=4.59)
        """

        self.gamma = gamma

    def model(self, z):
        """
        Power-law rate evolution

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
        """
        return (1+z)**self.gamma


