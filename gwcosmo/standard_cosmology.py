"""
Module for basic late-time cosmology calculations. Currently implements only flat universe.

Constants
---------
c : speed of light in km/s
Omega_m : matter fraction
H0 : Hubble parameter

(c) Archisman Ghosh, 2013-Nov
"""

import numpy as np
from scipy import integrate
import lal

c = lal.C_SI/1000. #2.99792458e+05 # in km/s
Omega_m = 0.3 # 0.3175 # PLANCK best fit
H0 = 70 # 67.11 # in km/s/Mpc

def h(z, Omega_m=Omega_m):
  """
  Returns dimensionless redshift-dependent hubble parameter.
  
  Parameters
  ----------
  z : redshift
  Omega_m : matter fraction
  
  Returns
  -------
  dimensionless h(z) = 1/sqrt(Omega_m*(1+z)^3 + Omega_Lambda)
  """
  Omega_Lambda = (1-Omega_m)
  return np.sqrt(Omega_m*(1+z)**3 + Omega_Lambda) 

def dcH0overc(z, Omega_m=Omega_m):
  """
  Returns dimensionless combination dc*H0/c given redshift and matter fraction.
  
  Parameters
  ----------
  z : redshift
  Omega_m : matter fraction
  
  Returns
  -------
  dimensionless combination dc*H0/c = \int_0^z dz'/h(z')
  """
  Omega_Lambda = (1-Omega_m)
  integrand = lambda zz: 1./np.sqrt(Omega_m*(1+zz)**3 + Omega_Lambda)
  return integrate.quad(integrand, 0, z)[0] # in km/s

def dLH0overc(z, Omega_m=Omega_m):
  """
  Returns dimensionless combination dL*H0/c given redshift and matter fraction.
  
  Parameters
  ----------
  z : redshift
  Omega_m : matter fraction
  
  Returns
  -------
  dimensionless combination dL*H0/c = (1+z) * \int_0^z dz'/h(z')
  """
  return (1+z)*dcH0overc(z, Omega_m)

def volume_z(z, Omega_m=Omega_m):
  """
  Returns the cosmological volume at the given redshift.
  
  Parameters
  ----------
  z : redshift
  Omega_m : matter fraction
  
  Returns
  -------
  volume (\int_0^z dz'/h(z'))^2 / h(z)
  """
  return dcH0overc(z, Omega_m)**2/h(z, Omega_m)  

def prefactor_volume_dHoc(dHoc, Omega_m=Omega_m, tolerance_z=1e-06, z=None):
  """
  Returns the prefactor modifying dL^2*ddL for the cosmological volume element.
  
  Parameters
  ----------
  dLH0overc : dimensionless combination dL*H0/c
  Omega_m : matter fraction
  tolerance_z : (optional) tolerated error in redshift. default = 1e-06
  z : (optional) redshift, if it has been calculated already
  
  Returns
  -------
  prefactor, (1+z)^(-3) * (1 - 1 / (1 + (1+z)^2/(dHoc*h(z))))
  """
  if z is None:
    z = redshift(dHoc, Omega_m, tolerance_z)
  return (1+z)**(-3.) * (1 - 1. / (1 + (1+z)**2/(dHoc*h(z, Omega_m))))

def volume_dHoc(dHoc, Omega_m=Omega_m, tolerance_z=1e-06, z=None):
  """
  Returns cosmological volume at the given dL*H0/c.
  
  Parameters
  ----------
  dLH0overc : dimensionless combination dL*H0/c
  Omega_m : matter fraction
  tolerance_z : (optional) tolerated error in redshift. default = 1e-06
  z : (optional) redshift, if it has been calculated already
  
  Returns
  -------
  volume, dHoc^2 * (1+z)^(-3) * (1 - 1 / (1 + (1+z)^2/(dHoc*h(z))))
  """
  return dHoc**2*prefactor_volume_dHoc(dHoc, Omega_m, tolerance_z, z=z)

def redshift(dHoc, Omega_m=Omega_m, tolerance_z=1e-06):
  """
  Returns redshift given dimensionless combination dL*H0/c and matter fraction.
  
  Parameters
  ----------
  dLH0overc : dimensionless combination dL*H0/c
  Omega_m : matter fraction
  tolerance_z : (optional) tolerated error in redshift. default = 1e-06.
  
  Returns
  -------
  redshift, z
  """
  min_z = 0.
  max_z = 1.
  error_z = max_z-min_z
  while error_z>tolerance_z:
    if dLH0overc(max_z,Omega_m)<dHoc:
      min_z = max_z
      max_z *= 2
    elif dLH0overc((max_z+min_z)/2.,Omega_m)<dHoc:
      min_z = (max_z+min_z)/2.
    else:
      max_z = (max_z+min_z)/2.
    error_z = max_z-min_z
  return (max_z+min_z)/2.

# Distance modulus given luminosity distance
def mu(dL):
  """
  Returns distance modulus given luminosity distance
  
  Parameters
  ----------
  dL : luminosity distance in Mpc
  
  Returns
  -------
  distance modulus, mu = 5*np.log10(dL)+25
  """
  return 5*np.log10(dL)+25 # dL has to be in Mpc

def dl_mM(m,M):
    """
    returns luminosity distance in Mpc given apparent magnitude and absolute magnitude
    """
    return 10**(0.2*(m-M-25))


# Rachel: I've put dl_zH0 and z_dlH0 in as place holders.
def dl_zH0(z, H0=70., Omega_m=0.3, linear=False):
    """
    Returns luminosity distance given distance and cosmological parameters

    Parameters
    ----------
    z : redshift
    H0 : Hubble parameter in km/s/Mpc (default=70.)
    Omega_m : matter fraction (default=0.3) 
    linear : assumes local cosmology and suppresses non-linear effects (default=False)
    
    Returns
    -------
    luminosity distance, dl (in Mpc)
    """
    if linear:
        # Local cosmology
        return z*c/H0
    else:
        # Standard cosmology
        return dLH0overc(z, Omega_m=Omega_m)*c/H0
    
def z_dlH0(dl, H0=70., Omega_m=0.3, linear=False):
    """
    Returns redshift given luminosity distance and cosmological parameters

    Parameters
    ----------
    dl : luminosity distance in Mpc
    H0 : Hubble parameter in km/s/Mpc (default=70.)
    Omega_m : matter fraction (default=0.3)
    linear : assumes local cosmology and suppresses non-linear effects (default=False)
    
    Returns
    -------
    redshift, z
    """
    if linear:
        # Local cosmology
        return dl*H0/c
    else:
        # Standard cosmology
        return redshift(dl*H0/c, Omega_m=Omega_m)
  
