"""
Module for late-time cosmology calculations.
Modified gravity models are included.
Currently implements only a flat universe.

Anson Chen

Constants
---------
c : speed of light in km/s
Omega_m : matter fraction
H0 : Hubble parameter


"""

import numpy as np
from scipy import integrate
from scipy.interpolate import splrep, splev, RegularGridInterpolator, interp1d
import lal

c = lal.C_SI/1000.  # 2.99792458e+05 # in km/s


def h(z, Omega_m=0.3065, w0=-1., wa=0.):
    """
    Returns dimensionless redshift-dependent hubble parameter.

    Parameters
    ----------
    z : redshift
    Omega_m : matter fraction
    Dynamical dark energy: w(a) = w0+wa(1-a)

    Returns
    -------
    dimensionless h(z) = sqrt(Omega_m*(1+z)^3 + Omega_Lambda
    *(1+z)^[3(1+w0+wa)]*e^[-3*wa*z/(1+z)])
    """
    Omega_Lambda = (1-Omega_m)
    return np.sqrt(Omega_m*(1+z)**3 + Omega_Lambda* (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z)))


def dcH0overc(z, Omega_m=0.3065, w0=-1., wa=0.):
    """
    Returns dimensionless combination dc*H0/c
    given redshift and matter fraction.

    Parameters
    ----------
    z : redshift
    Omega_m : matter fraction
    Dynamical dark energy: w(a) = w0+wa(1-a)

    Returns
    -------
    dimensionless combination dc*H0/c = \int_0^z dz'/h(z')
    """
    integrand = lambda zz: 1./h(zz, Omega_m, w0, wa)

    if np.size(z)>1:
        if np.size(np.where(z<=0))>0:
            raise ValueError('Negative redshift input!')
        result = np.array([integrate.quad(integrand, 0, zi)[0] for zi in z])
    else:
        if z<=0:
            raise ValueError('Negative redshift input!')
        result = integrate.quad(integrand, 0, z)[0]  # in km/s

    return result


def dLH0overc(z, Omega_m=0.3065, w0=-1., wa=0.):
    """
    Returns dimensionless combination dL*H0/c
    given redshift and matter fraction.

    Parameters
    ----------
    z : redshift
    Omega_m : matter fraction

    Returns
    -------
    dimensionless combination dL*H0/c = (1+z) * \int_0^z dz'/h(z')
    """
    return (1+z)*dcH0overc(z, Omega_m, w0, wa)



class standard_cosmology(object):

    def __init__(self, H0=70, Omega_m=0.3065, w0=-1., wa=0., zmax=10.0, zmin=1.e-5, zbin=5000):

        self.c = c
        self.H0 = H0
        self.Omega_m = Omega_m
        self.w0 = w0
        self.wa = wa
        self.zmax = zmax
        self.zmin = zmin
        self.zbin = zbin
        self.z_array = np.logspace(np.log10(self.zmin), np.log10(self.zmax), self.zbin)

        # Interpolation of z(dL)
        self.dlH0overc_z_arr = np.array([dLH0overc(z, Omega_m=self.Omega_m, w0=self.w0, wa=self.wa)
                        for z in self.z_array])
        self.dlH0overc_of_z = splrep(self.z_array, self.dlH0overc_z_arr)
        self.z_of_dlH0overc = splrep(self.dlH0overc_z_arr, self.z_array)

        # Interpolation of p(z)
        self.log_pz_arr = np.log10(np.array([self.volume_z(z) for z in self.z_array]))
        self.log_pz_of_z = splrep(np.log10(self.z_array), self.log_pz_arr)

    def update_parameters(self,param_dict):
        """
        Update values of cosmological parameters. 
        Key in param_dict: H0
        """
        if 'H0' in param_dict:
            if param_dict['H0'] != self.H0:
                self.H0 = param_dict['H0']

    def dl_zH0(self, z):
        """
        Returns luminosity distance given redshift

        Parameters
        ----------
        z : redshift

        Returns
        -------
        luminosity distance, dl (in Mpc)
        """
        return splev(z, self.dlH0overc_of_z, ext=3)*c/self.H0

    def dgw_z(self, z):
        """
        Returns GW luminosity distance given redshift

        Parameters
        ----------
        z : redshift

        Returns
        -------
        luminosity distance, dgw (in Mpc)
        """
        return self.dl_zH0(z) * self.dgw_dL_ratio(z)

    def z_dlH0(self,dl):
        """
        Returns redshift given luminosity distance

        Parameters
        ----------
        dl : luminosity distance in Mpc

        Returns
        -------
        redshift, z
        """
        return splev(dl*self.H0/c, self.z_of_dlH0overc, ext=3)


    def z_dgw(self,dgw):
        """
        Returns redshift given GW luminosity distance

        Parameters
        ----------
        dgw : GW luminosity distance in Mpc

        Returns
        -------
        redshift, z
        """
        return self.z_dlH0(dgw)

    def dgw_dL_ratio(self, z):
        """
        Returns ratio between GW distance and luminosity distance
        """
        return 1

    def dgw_dL_ratio_dbyz(self, z):
        """
        Returns the derivatives of the ratio between GW distance and 
        luminosity distance w.r.t. redshift
        """
        return 0

    def ddgw_dz(self, z):
        """
        Returns the derivative of the GW distance w.r.t. redshift

        Parameters
        ----------
        z : redshift
        """
        H_z = self.H0*h(z, self.Omega_m, self.w0, self.wa)

        return self.dgw_z(z)/(1+z) + c*(1+z)/H_z * self.dgw_dL_ratio(z) + self.dgw_z(z) * self.dgw_dL_ratio_dbyz(z)

    def volume_z(self, z):
        """
        Returns the cosmological volume at the given redshift.

        Parameters
        ----------
        z : redshift

        Returns
        -------
        volume element (\int_0^z dz'/h(z'))^2 / h(z): dimensionless
        """
        return dcH0overc(z, self.Omega_m, self.w0, self.wa)**2 / h(z, self.Omega_m, self.w0, self.wa)


    def volume_time_z(self, z):
        """
        Returns the cosmological volume time element at a given redshift.

        Parameters
        ----------
        z : redshift

        Returns
        -------
        volume time element (\int_0^z dz'/h(z'))^2 / (1+z)h(z)
        """
        return self.volume_z(z)/(1.0+z)


    def p_z(self, z):
        """
        p(z|Omega_m): Uniform in comoving volume distribution of galaxies
        """
        return 10.**splev(np.log10(z), self.log_pz_of_z, ext=3)


class local_cosmology(object):

    def __init__(self, H0=70):

        print('Using local cosmology. The result will deviate at high redshift.')

        self.H0 = H0

    def dgw_z(self, z):
        """
        Returns luminosity distance given redshift

        Parameters
        ----------
        z : redshift

        Returns
        -------
        luminosity distance, dl (in Mpc)
        """
        # Local cosmology
        return z*c/self.H0

    def z_dgw(self,dl):
        """
        Returns redshift given luminosity distance

        Parameters
        ----------
        dl : luminosity distance in Mpc

        Returns
        -------
        redshift, z
        """
        # Local cosmology
        return dl*self.H0/c

    def ddgw_dz(self, z):
        """
        Returns the derivative of the luminosity distance w.r.t. redshift

        Parameters
        ----------
        z : redshift
        """
        return c/self.H0

    def p_z(self,z):
        """
        p(z|Omega_m): Uniform in comoving volume distribution of galaxies
        """
        return z*z


class Xi0_n_cosmology(standard_cosmology):
    # Modified cosmology in (Xi0, n) parameterization, inheriting features from standard cosmology.

    def __init__(self, H0=70, Xi0=1, n=1.91, Omega_m=0.3065, w0=-1., wa=0., zmax=10.0, zmin=1.e-5, zbin=5000):

        super().__init__(H0, Omega_m, w0, wa, zmax, zmin, zbin)

        self.H0 = 0
        self.__Xi0 = 0
        self.__n = 0

        self.update_parameters(param_dict={'H0': H0, 'Xi0': Xi0, 'n': n})

    def update_parameters(self, param_dict):
        """
        Update values of cosmological parameters. 
        Keys in param_dict: H0, Xi0, n
        """
        if 'H0' in param_dict:
            if param_dict['H0'] != self.H0:
                self.H0 = param_dict['H0']

        update_dgw_dL_ratio = False
        if 'Xi0' in param_dict:
            if param_dict['Xi0'] != self.__Xi0:
                self.__Xi0 = param_dict['Xi0']
                update_dgw_dL_ratio = True
        if 'n' in param_dict:
            if param_dict['n'] != self.__n:
                self.__n = param_dict['n']
                update_dgw_dL_ratio = True
        if update_dgw_dL_ratio:
            self.z_of_dgw = interp1d(self.dgw_dL_ratio(self.z_array)*self.dlH0overc_z_arr, self.z_array, kind='cubic')

    def dgw_dL_ratio(self, z):
        """
        Returns the ratio of GW distance and luminosity distance
        in modified gravity.
        Parametrization introduced by the Geneva group:
        dgw(z) / dL(z) = \Xi(z) = \Xi_0 + (1-\Xi_0)/(1+z)^n

        Parameters
        ----------
        z : redshift
        Xi0, n : modified gravity parameters

        Returns
        -------
        dgw(z) / dL(z)
        """
        return self.__Xi0 + (1-self.__Xi0)/(1+z)**self.__n


    def dgw_dL_ratio_dbyz(self, z):
        """
        Returns the derivative of ratio of GW distance and 
        luminosity distance w.r.t. redshift in modified gravity.
        d\Xi(z) / dz = - n * (1-\Xi_0)/(1+z)^(n+1)

        Parameters
        ----------
        z : redshift
        Xi0, n : modified gravity parameters

        Returns
        -------
        d(dgw(z) / dL(z)) / dz
        """
        return -1 * self.__n*(1-self.__Xi0)/((1+z)**(self.__n+1))


    def z_dgw(self, dgw):
        """
        Returns redshift given GW distance

        Parameters
        ----------
        dgw : GW distance in Mpc

        Returns
        -------
        redshift, z
        """
        zgw = self.z_of_dgw(dgw*self.H0/c)

        return zgw

