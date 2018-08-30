#!/usr/bin/env python

"""
Module with detection efficiency functions

(C) Ankan Sur, Anuradha Samajdar, Archisman Ghosh
"""

import os, math, numpy as np
from scipy import integrate, interpolate
try:
  from lal import C_SI
except ImportError:
  C_SI = 299792458.0

# simple cubic dependence
def cube(H0, *args, **kwargs):
  return H0**3.

# basic error function that is integrated over
def integrand(rho_0, rho_1):
  return np.sqrt(np.pi/2.) * math.erfc((rho_0-rho_1)/np.sqrt(2.))

# ### Simplified detection efficiency assuming optimal location and orientation
# ### Functions to calculate the detection efficiency (as suggested by Gair)
# ### References: Mandel, Farr, Gair: LIGO-P1600187
#                 Farr et al.       : https://git.ligo.org/will-farr/HubbleConstantNotes

# substitute in the basic integrand
def substituted_integrand(H0, z, d_horiz, rho_th):
  return integrand(rho_th, rho_th*d_horiz*H0/(C_SI*1e-3*z))

# integral over z to get detection efficiency 
def p_det_opt(H0, d_horiz=160., rho_th=8.0, z_min=0.001, z_max=0.04):
  return integrate.quad(lambda z: z**2.*substituted_integrand(H0, z, d_horiz, rho_th), z_min, z_max)[0]

# ### Detection efficiency with effect of antenna pattern included
# ### Functions to calculate the detection efficiency including p(w) as in 1709.08079

# ### P(w) is the "cumulative antenna pattern" defined in Section 4.2 of Chen et al. arXiv:1709.08079
# ### the file pw_hlv.txt is downloaded from: https://github.com/hsinyuc/distancetool/tree/master/data

# P(w) for HLV network
w, pw = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pw_hlv.txt'),unpack=True)
pw_interp = interpolate.interp1d(w,pw,bounds_error=False,fill_value=0.)

# substitute in the basic integrand
def substituted_integrand_with_w_alternate(H0, w, z, d_horiz, rho_th):
  return integrand(rho_th, w*rho_th*d_horiz*H0/(C_SI*1e-3*z))

# double integrand for integration over (z,w)
def double_integrand(w, z, H0, d_horiz, rho_th):
  return pw_interp(w)*z**2.*substituted_integrand_with_w_alternate(H0, w, z, d_horiz, rho_th)

# double integral over (z,w) giving detection efficiency
def p_det_ang_avg(H0, d_horiz=160., rho_th=8.0, z_min=0.001, z_max=0.04):
  return integrate.dblquad(double_integrand, z_min, z_max, lambda x: 0, lambda x: 1, args=(H0, d_horiz, rho_th))[0]

# ### Alternate approach to calculate the detection efficiency including p(w) as in 1709.08079

# integral over z
def integrand_with_pw_and_pz(H0, snr_ev, d_horiz=160., snr_max=100., rho_th=8.0, z_min=0.001, z_max=0.04):
  wnew = snr_ev/snr_max
  return integrate.quad(lambda z: z**2.*(np.sqrt(np.pi/2.) * math.erfc((rho_th-rho_th*d_horiz*H0*wnew/(C_SI*1e-3*z))/np.sqrt(2.))), z_min, z_max)[0]*pw_interp(wnew)

# integral over SNR to get detection efficiency
def p_det_ang_avg_alt(H0, d_horiz=160., snr_max=100., rho_th=8.0, z_min=0.001, z_max=0.04):
  return integrate.quad(lambda rho: integrand_with_pw_and_pz(H0, rho, d_horiz, snr_max, rho_th, z_min, z_max), rho_th, snr_max)[0]
