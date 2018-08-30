#!/usr/bin/env python

"""
Main script for Hubble parameter estimation

(C) Archisman Ghosh, Ankan Sur, Anuradha Samajdar
"""

import os, sys, commands, numpy as np, healpy as hp, h5py
from scipy import integrate, interpolate, random
from scipy.stats import gaussian_kde
from scipy.misc import logsumexp
try:
  from lal import C_SI
except ImportError:
  C_SI = 299792458.0
from selection_gw import cube, p_det_opt, p_det_ang_avg, p_det_ang_avg_alt
import cPickle as pickle

import standard_cosmology as sc
from populate_posteriors import SchectherMagFunction

# Wrapper around function defined in standard_cosmology to take into account linear cosmology
def dLH0overc(z, Omega_m=0.3, **kwargs):
  if kwargs.has_key('linear') and kwargs['linear']:
    return z
  else:
    return sc.dLH0overc(z, Omega_m=Omega_m)
  #linear * z + (1.-linear) * sc.dLH0overc(z, **kwargs)

# Wrapper around function defined in standard_cosmology to take into account linear cosmology
def volume_z(z, Omega_m=0.3, **kwargs):
  if kwargs.has_key('linear') and kwargs['linear']:
    return z**2.
  else:
    return sc.volume_z(z, Omega_m=Omega_m)
  #linear * z**2. + (1.-linear) * sc.volume_z(z, **kwargs) 

# Proportional to number of visible galaxies at redshift z
def numerical_incomplete_gamma(z, H0=70., Mmin=-30., Mmax=-16., mth=18., phistar=1., Mstar_obs=-20.37, alpha=-1.07, **kwargs):
  Mstar = Mstar_obs + 5.*np.log10(H0/100.)
  smf = SchectherMagFunction(Mstar, alpha, phistar).evaluate
  (integral, error) = integrate.quad(smf, Mmin, max(Mmin, min(mth-sc.mu(dLH0overc(z, **kwargs)*sc.c/H0), Mmax)))
  return volume_z(z, **kwargs) * integral

# Proportional to total number of galaxies at redshift z
def numerical_complete(z, H0=70., Mmin=-30., Mmax=-16., phistar=1., Mstar_obs=-20.37, alpha=-1.07, **kwargs):
  Mstar = Mstar_obs + 5.*np.log10(H0/100.)
  smf = SchectherMagFunction(Mstar, alpha, phistar).evaluate
  (integral, error) = integrate.quad(smf, Mmin, Mmax)
  return volume_z(z, **kwargs) * integral

# Proportional to fraction of galaxies in catalog up to redshift z_max
def fraction_complete(z_max, z_min=0.001, **kwargs):
  (nobs, err_nobs) = integrate.quad(lambda z: numerical_incomplete_gamma(z, **kwargs), z_min, z_max)
  (ntot, err_ntot) = integrate.quad(lambda z: numerical_complete(z, **kwargs), z_min, z_max)
  return nobs/ntot

# ### Module for credible level calculations ###

# ### NOTE: This module is reviewed and available on LALSuite in:
# ### lalinference/python/imrtgr_imr_consistency_test.py

class confidence(object):
  def __init__(self, counts):
    # Sort in descending order in frequency
    self.counts_sorted = np.sort(counts.flatten())[::-1]
    # Get a normalized cumulative distribution from the mode
    self.norm_cumsum_counts_sorted = np.cumsum(self.counts_sorted) / np.sum(counts)
    # Set interpolations between heights, bins and levels
    self._set_interp()
  def _set_interp(self):
    self._length = len(self.counts_sorted)
    # height from index
    self._height_from_idx = interpolate.interp1d(np.arange(self._length), self.counts_sorted, bounds_error=False, fill_value=0.)
    # index from height
    self._idx_from_height = interpolate.interp1d(self.counts_sorted[::-1], np.arange(self._length)[::-1], bounds_error=False, fill_value=self._length)
    # level from index
    self._level_from_idx = interpolate.interp1d(np.arange(self._length), self.norm_cumsum_counts_sorted, bounds_error=False, fill_value=1.)
    # index from level
    self._idx_from_level = interpolate.interp1d(self.norm_cumsum_counts_sorted, np.arange(self._length), bounds_error=False, fill_value=self._length)
  def level_from_height(self, height):
    return self._level_from_idx(self._idx_from_height(height))
  def height_from_level(self, level):
    return self._height_from_idx(self._idx_from_level(level))

# ### Utilities for HEALPix maps ###

# RA and dec from HEALPix index
def ra_dec_from_ipix(nside, ipix):
  (theta, phi) = hp.pix2ang(nside, ipix)
  return (phi, np.pi/2.-theta)

# HEALPix index from RA and dec
def ipix_from_ra_dec(nside, ra, dec):
  (theta, phi) = (np.pi/2.-dec, ra)
  return hp.ang2pix(nside, theta, phi)

# ### Functions specific to this pipeline ###

# Logarithm of the likelihood function given by Eqn. (1) of LIGO-T1700422-v2
def logLikelihood(x, weight_list, z_list, dist_likelihood, h_min, h_max, om_min=0., om_max=1., **kwargs):
  (h, om) = x
  if (h_min <= h <= h_max) & (om_min <= om <= om_max):
    dist_list = np.vectorize(dLH0overc, otypes=[np.float64])(z_list, Omega_m=om, **kwargs)*(C_SI/1e3)/(100.*h)
    return np.log(np.sum(weight_list*dist_likelihood(dist_list)))
  else:
    return -np.inf

# Contribution from undetected EM counterparts given a GW detection
def logIncompleteDiff(x, dist_likelihood, h_min, h_max, om_min=0., om_max=1., z_min=0.001, z_max=1., z_res=100, **kwargs):
  (h, om) = x
  z_list = np.linspace(z_min, z_max, z_res)
  weight_list = (np.vectorize(numerical_complete)(z_list, H0=100.*h, **kwargs) - np.vectorize(numerical_incomplete_gamma)(z_list, H0=100.*h, **kwargs))
  if (h_min <= h <= h_max) & (om_min <= om <= om_max):
    dist_list = np.vectorize(dLH0overc)(z_list, Omega_m=om, **kwargs)*(C_SI/1e3)/(100.*h)
    return np.log(np.sum(weight_list*dist_likelihood(dist_list)))
  else:
    return -np.inf

# Contribution from undetected EM counterparts given a GW detection
def logIncompleteFrac(x, dist_likelihood, h_min, h_max, om_min=0., om_max=1., z_min=0.001, z_max=1., z_res=100, **kwargs):
  (h, om) = x
  z_list = np.linspace(z_min, z_max, z_res)
  weight_list = (1. - np.vectorize(numerical_incomplete_gamma)(z_list, H0=100.*h, **kwargs)/np.vectorize(numerical_complete)(z_list, H0=100.*h, **kwargs)) * np.vectorize(volume_z)(z_list, **kwargs)
  if (h_min <= h <= h_max) & (om_min <= om <= om_max):
    dist_list = np.vectorize(dLH0overc)(z_list, Omega_m=om, **kwargs)*(C_SI/1e3)/(100.*h)
    return np.log(np.sum(weight_list*dist_likelihood(dist_list)))
  else:
    return -np.inf

# np.log(1.-frac_obs_arr)
# frac_obs_arr = np.vectorize(fraction_complete)(z_max, z_min, z_res=z_res, H0=H_arr, mth=mth, Mmin=Mmin, Mmax=Mmax, Mstar_obs=Mstar_obs, alpha=alpha, linear=linear)

# ### The following two functions are for diagnosis only (not used in code)

# Proportional to total number of possible EM counterparts given a GW detection; weighted with cosmological volume
def logCompleteVolume(x, dist_likelihood, h_min, h_max, om_min=0., om_max=1., z_min=0.001, z_max=1., z_res=100, **kwargs):
  (h, om) = x
  z_list = np.linspace(z_min, z_max, z_res)
  weight_list = np.vectorize(volume_z)(z_list, **kwargs)
  if (h_min <= h <= h_max) & (om_min <= om <= om_max):
    dist_list = np.vectorize(dLH0overc)(z_list, Omega_m=om, **kwargs)*(C_SI/1e3)/(100.*h)
    return np.log(np.sum(weight_list*dist_likelihood(dist_list)))
  else:
    return -np.inf

# Proportional to total number of possible EM counterparts given a GW detection; weighted with number of galaxies
def logCompleteTotal(x, dist_likelihood, h_min, h_max, om_min=0., om_max=1., z_min=0.001, z_max=1., z_res=100, **kwargs):
  (h, om) = x
  z_list = np.linspace(z_min, z_max, z_res)
  weight_list = np.vectorize(numerical_complete)(z_list, H0=100.*h, **kwargs)
  if (h_min <= h <= h_max) & (om_min <= om <= om_max):
    dist_list = np.vectorize(dLH0overc)(z_list, Omega_m=om, **kwargs)*(C_SI/1e3)/(100.*h)
    return np.log(np.sum(weight_list*dist_likelihood(dist_list)))
  else:
    return -np.inf

# Functions to convert z_helio to z_CMB
def sph2vec(ra,dec):
  ra=ra/180.*np.pi
  dec=dec/180.*np.pi
  return np.array([np.sin(np.pi/2.-dec)*np.cos(ra),np.sin(np.pi/2.-dec)*np.sin(ra),np.cos(np.pi/2.-dec)])

def zhelio_to_zcmb(ra,dec,z_helio):
  ra_cmb = 167.99
  dec_cmb = -7.22
  v_cmb = 369.
  z_gal_cmb = v_cmb*np.dot(sph2vec(ra_cmb,dec_cmb),sph2vec(ra,dec))/(C_SI/1e3)
  z_cmb = (1.+z_gal_cmb)*(1.+z_helio) - 1.
  return z_cmb

if __name__ == '__main__':
  
  from optparse import OptionParser
  
  parser = OptionParser()
  
  parser.add_option('--pos', type='string', dest='possamp_file', help='posterior samples (ASCII or HDF5)')
  parser.add_option('--sky', type='string', dest='skymap_file', help='skymap (FITS)')
  parser.add_option('--gal', type='string', dest='galcat_file', help='galaxy catalog')
  parser.add_option('--fmt', type='string', dest='cat_format', default='GLADE', help='catalog format: [GLADE, F2Y_MDC, F2Y_MDC_v2, other]')
  parser.add_option('--lev', type='float', dest='sky_level', default=0.99, help='credible region in skymap to choose galaxies from (default = 0.99; i.e. 99%)')
  parser.add_option('--luminosity-cut', type='float', dest='lum_cut', default=0.01, help='cut on galaxies brighter than this fraction of characteristic luminosity (default = 0.01)')
  parser.add_option('--z-min', type='float', default=0.001, help='minimum redshift of galaxies to look at (default = 0.001)')
  parser.add_option('--z-max', type='float', default=2.0, help='maximum redshift of galaxies to look at (default = 2.0)')
  parser.add_option('--z-res', type='int', default=100, help='resolution in redshift for integration (default = 100)')
  parser.add_option('--H0-min', type='float', default=25., help='minimum Hubble parameter H0 (default = 10 km/s/Mpc)')
  parser.add_option('--H0-max', type='float', default=150., help='maximum Hubble parameter H0 (default = 200 km/s/Mpc)')
  parser.add_option('--H0-res', type='int', default=100, help='resolution Hubble parameter H0 (default = 100)')
  parser.add_option('--omega-m', type='float', dest='om', default=0.3, help='matter fraction Omega_m [global declaration] (default = 0.3)')
  parser.add_option('--linear', action='store_true', help='use a linear relationship for a local cosmology', default=False)
  parser.add_option('--mth', type='float', default=18., help='apparent magnitude threshold for optical telescope')
  parser.add_option('--Mstar-obs', type='float', default=-20.37, help='characteristic (absolute) magnitude for Schechter function (dafault = -20.37)')
  parser.add_option('--alpha', type='float', default=-1.07, help='slope of Schechter function (default = -1.07)')
  parser.add_option('--Mmax', type='float', default=-16., help='dimmest absolute magnitude for completion (default = -16)')
  parser.add_option('--Mmin', type='float', default=-30., help='brightest absolute magnitude for limit of numerical integration (default = -30)')
  parser.add_option('--no-dist-prior-corr', action='store_false', dest='dist_prior_corr', help='no distance prior correction (applied by default)', default=True)
  parser.add_option('--selection-corr', type='string', default=None, help='choose from: [cube, gair, chen] (default = None)')
  parser.add_option('--completion', action='store_true', help='correct for incompleteness of galaxy catalog', default=False)
  parser.add_option('--do-not-normalize', action='store_true', help='do not normalize before adding terms', default=False)
  parser.add_option('--luminosity-weighting', action='store_true', help='luminosity weighting (default = False)', default=False)
  parser.add_option('--helio-cmb-conv', action='store_true', help='heliocentric to CMB-frame conversion (default = False)', default=False)
  parser.add_option('--d-horiz', type='float', default=190., help='horizon distance of IFO network in Mpc (default = 190.)')
  parser.add_option('--force', action='store_true', help='force rewriting of output galaxy catalog subset file', default=False)
  parser.add_option('--disable-plot', action='store_false', dest='plotting', help='turn off plotting', default=True)
  parser.add_option('--plot-label', type='string', default=r'$H_0$', help='plot label')
  parser.add_option('--out', type='string', dest='out_folder', help='output folder')
  
  (options, args) = parser.parse_args()

  possamp_file = options.possamp_file
  skymap_file = options.skymap_file
  galcat_file = options.galcat_file
  cat_format = options.cat_format
  sky_level = options.sky_level
  lum_cut = options.lum_cut
  z_min = options.z_min
  z_max = options.z_max
  z_res = options.z_res
  H0_min = options.H0_min
  H0_max = options.H0_max
  H0_res = options.H0_res
  om = options.om
  linear = options.linear
  mth = options.mth
  Mstar_obs = options.Mstar_obs
  alpha = options.alpha
  Mmax = options.Mmax
  Mmin = options.Mmin
  out_folder = options.out_folder
  dist_prior_corr = options.dist_prior_corr
  selection_corr = options.selection_corr
  completion = options.completion
  do_not_normalize = options.do_not_normalize
  luminosity_weighting = options.luminosity_weighting
  helio_cmb_conv = options.helio_cmb_conv
  d_horiz = options.d_horiz
  force = options.force
  plotting = options.plotting
  plot_label = options.plot_label
  
  os.system('mkdir -p %s'%(out_folder))
  
  gal_file_suffix = ''.join((os.path.basename(galcat_file).split('.')[0], helio_cmb_conv*'_helio-cmb-conv'))
  H0_file_suffix = ''.join((gal_file_suffix, linear*'_linear', linear*'_linear', (selection_corr!=None)*('_sel-corr-%s'%(selection_corr)), completion*'_completion-corr', (not dist_prior_corr)*'_no-dist-prior-corr'))
  if cat_format == 'GLADE':
    H0_file_suffix = ''.join((H0_file_suffix, '_lum-cut-%s'%(str(round(lum_cut, 2)).replace('.', 'p')), luminosity_weighting*'_lum-weight'))
  
  out_gal_file = os.path.join(out_folder, 'galaxies_%s.txt'%(gal_file_suffix))
  out_plot_file = os.path.join(out_folder, 'H0_%s.png'%(H0_file_suffix))
  pickle_file = os.path.join(out_folder, 'H0_%s.p'%(H0_file_suffix))
  
  # Read posterior samples and make KDE of distance samples
  if possamp_file[-5:] == '.hdf5':
    possamp = h5py.File(possamp_file).get('lalinference_mcmc/posterior_samples').value
    dist_samp = possamp['dist']
  else:
    possamp = np.genfromtxt(possamp_file, names=True, dtype=None)
    dist_samp = possamp['distance']
  dist_kde = gaussian_kde(dist_samp)
  if dist_prior_corr:
    # Change of prior from uniform in volume to uniform in distance
    xx = np.linspace(0.9*np.min(dist_samp), 1.1*np.max(dist_samp), 2000.)
    yy = dist_kde(xx)/xx**2.
    yy /= np.sum(yy)*(xx[1]-xx[0])
    # Interpolation of normalized prior-corrected distribution
    try:
      # The following works only on recent python versions
      dist_support = interpolate.InterpolatedUnivariateSpline(xx, yy, ext=1)
    except TypeError:
      # A workaround to prevent bounds error in earlier python versions
      dist_interp = interpolate.InterpolatedUnivariateSpline(xx, yy)
      def dist_support(x):
        if (x>=xx[0]) and (x<=xx[-1]):
          return dist_interp(x)
        return 0.
      dist_support = np.vectorize(dist_support, otypes=[np.float64])
  else:
    dist_support = dist_kde
  
  # Select the galaxies and write them to a file
  if not os.path.isfile(out_gal_file) or force:

    # Load skymap
    skymap = hp.read_map(skymap_file)
    nside = hp.npix2nside(len(skymap))

    # Load galaxy catalog (note column ordering)
    if cat_format == 'GLADE':
      (ra_gal, dec_gal, z_gal, mag_B_gal, abs_mag_B_gal, mag_K_gal) = np.genfromtxt(galcat_file, usecols=(6, 7, 10, 11, 13, 18), comments='%', unpack='True')
      if helio_cmb_conv:
        #convert z_gal_valid to CMB frame
        z_gal = zhelio_to_zcmb(ra_gal, dec_gal, z_gal)
      ra_gal *= np.pi/180.
      dec_gal *= np.pi/180.
    elif cat_format == 'F2Y_MDC':
      (ra_gal, dec_gal, z_gal) = np.genfromtxt(galcat_file, usecols=(2, 3, 4), comments='#', unpack='True')
    elif cat_format == 'F2Y_MDC_v2':
      (ra_gal, dec_gal, z_gal) = np.genfromtxt(galcat_file, usecols=(1, 2, 3), comments='#', unpack='True')
    else:
      (ra_gal, dec_gal, z_gal) = np.genfromtxt(galcat_file, usecols=(0, 1, 2), comments='%', unpack='True')

    # Map each galaxy to a pixel on the skymap
    ipix_gal = ipix_from_ra_dec(nside, ra_gal, dec_gal)

    # Height of probability contour corresponding to confidence level set above
    skyconf_obj = confidence(skymap)
    sky_height = skyconf_obj.height_from_level(sky_level)

    # Pixels of skymap inside the probability contour
    ipix_above_height, = np.where(skymap > sky_height)

    # Indices of galaxies inside the probability contour
    idx_gal_above_height = np.array([ig in ipix_above_height for ig in ipix_gal])
    
    # FIXME
    if cat_format == 'GLADE':
      
      # Indices of objects that are not globular clusters
      
      # B-mag cut
      Mmax_cut = Mstar_obs + 5.*np.log10(0.7) - np.log(lum_cut) #+ 2.*np.log(10.)
      
      # Impose a cut on z and B-mag (min and max values chosen above)
      valid_idx, = np.where((z_min<z_gal)&(z_gal<z_max)&(idx_gal_above_height)&(abs_mag_B_gal<Mmax_cut))
    
    else:
    
      # Impose a cut on z (min and max values chosen above)
      valid_idx, = np.where((z_min<z_gal)&(z_gal<z_max)&(idx_gal_above_height))
    
    sys.stderr.write('%d galaxies in %d%% sky.\n'%(len(valid_idx), int(100*sky_level)))
    valid_gal_ra_arr = ra_gal[valid_idx]
    valid_gal_dec_arr = dec_gal[valid_idx]
    valid_gal_z_arr = z_gal[valid_idx]
    valid_gal_sky_prob_arr = skymap[ipix_gal[valid_idx]]
    valid_gal_sky_conf_arr = np.vectorize(skyconf_obj.level_from_height, otypes=[np.float64])(valid_gal_sky_prob_arr)
    
    # FIXME
    if cat_format == 'GLADE':
      valid_gal_abs_mag_B_arr = abs_mag_B_gal[valid_idx]
    else:
      valid_gal_abs_mag_B_arr = (Mstar_obs + 5.*np.log10(0.7)) * np.ones(len(valid_idx))

    # Sort in order of decreasing probability
    sorted_idx = np.argsort(valid_gal_sky_prob_arr)[::-1]

    # Write selected "valid" galaxies to a file
    os.system('mkdir -p %s'%(os.path.dirname(out_gal_file)))
    np.savetxt(out_gal_file, np.array([valid_gal_ra_arr[sorted_idx], valid_gal_dec_arr[sorted_idx], valid_gal_z_arr[sorted_idx], valid_gal_sky_prob_arr[sorted_idx], valid_gal_sky_conf_arr[sorted_idx], valid_gal_abs_mag_B_arr[sorted_idx]]).T, delimiter='\t', header='ra\tdec\tz\tskyl\tconf\tabs_mag_B')

  # ### MAIN: Measure over a grid on H0 ###

  # Read galaxy parameters back from selected galaxy file
  # Note: skyl is the likelihood on the 2D sky
  data_gal = np.genfromtxt(out_gal_file, names=True)
  
  if np.shape(data_gal) == ():
    data_gal = np.array([data_gal])
  
  Mmax_cut = Mstar_obs + 5.*np.log10(0.7) - np.log(lum_cut) #+ 2.*np.log(10.)
  valid_idx, = np.where((z_min<data_gal['z'])&(data_gal['z']<z_max)&(data_gal['abs_mag_B']<Mmax_cut))
  
  ra_gal = data_gal['ra'][valid_idx]
  dec_gal = data_gal['dec'][valid_idx]
  z_gal = data_gal['z'][valid_idx]
  l_gal = data_gal['skyl'][valid_idx]
  conf_gal = data_gal['conf'][valid_idx]
  abs_mag_B_gal = data_gal['abs_mag_B'][valid_idx]

  # Create an array of H0 to measure over
  H_arr = np.linspace(H0_min, H0_max, H0_res)

  # FIXME: Apply weights proportional to magnitude / luminosity
  if luminosity_weighting:
    #smf = SchectherMagFunction(Mstar_obs + 5.*np.log10(0.7), alpha, 1.).evaluate
    #w_gal = l_gal * np.vectorize(smf)(abs_mag_B_gal)
    Mstar = Mstar_obs + 5.*np.log10(0.7) #FIXME
    w_gal = l_gal * 10.**(-0.4*(abs_mag_B_gal-Mstar))
  else:
    w_gal = l_gal

  # Calculate the likelihood of the data for the given H0
  log_like_H_arr_obs = np.array([logLikelihood((H/100., om), w_gal, z_gal, dist_support, H0_min/100., H0_max/100., linear=linear) for H in H_arr])
  
  # Completion
  if completion:
  
    # Term to account for missing galaxies in catalog
    log_like_H_arr_unobs = np.array([logIncompleteFrac((H/100., om), dist_support, H0_min/100., H0_max/100., z_min=z_min, z_max=z_max, z_res=z_res, Mmin=Mmin, Mmax=Mmax, mth=mth, Mstar_obs=Mstar_obs, alpha=alpha, linear=linear) for H in H_arr])
    
    # Calculate fraction of galaxies which are there in the catalog
    frac_obs_arr = np.vectorize(fraction_complete)(z_max, z_min, z_res=z_res, H0=H_arr, mth=mth, Mmin=Mmin, Mmax=Mmax, Mstar_obs=Mstar_obs, alpha=alpha, linear=linear)
    
    # Add the two terms with appropriate relative weights
    if not do_not_normalize:
      # Normalize the distributions before adding
      if not np.all(log_like_H_arr_obs==-np.inf):
        log_like_H_arr_obs -= logsumexp(log_like_H_arr_obs)+np.log(H_arr[1]-H_arr[0])
      if not np.all(log_like_H_arr_unobs==-np.inf):
        log_like_H_arr_unobs -= logsumexp(log_like_H_arr_unobs)+np.log(H_arr[1]-H_arr[0])
    log_like_H_arr = np.logaddexp(np.log(frac_obs_arr)+log_like_H_arr_obs, log_like_H_arr_unobs)
    
  else:
    
    log_like_H_arr = log_like_H_arr_obs
  
  # Apply correction for selection effects
  if selection_corr:
    p_det = {'cube': cube, 'gair': p_det_opt, 'chen': p_det_ang_avg}[selection_corr]
    log_like_H_arr -= np.log(np.vectorize(p_det)(H_arr, d_horiz=d_horiz, z_max=z_max))
  
  if np.all(log_like_H_arr==-np.inf):
    log_like_H_arr = np.zeros(len(H_arr))
  
  # Normalize the distribution
  log_like_H_arr -= logsumexp(log_like_H_arr)+np.log(H_arr[1]-H_arr[0])
  
  # Pickle dump the result
  pickle.dump((H_arr, log_like_H_arr), open(pickle_file, 'wb'))


  if plotting:

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(H_arr, np.exp(log_like_H_arr), 'g-', label=plot_label)
    #plt.plot(H_arr, np.exp(log_like_H_arr_unobs),'g-',label=plot_label)
    plt.xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]')
    plt.axvline(70., color='k', linestyle='--')
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(out_plot_file)
    #plt.plot(ra_gal)
    #plt.savefig(out_plot_file)
