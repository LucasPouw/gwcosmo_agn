#!/usr/bin/env python

# script to calculate the likelihood for a single MDC event. Can specify event number, MDC type, KDE type (2 or 3D for MDC 1).  
# Currently all methods use samples - in future will also be compatible with skymaps.
# Currently relies on reading in precomputed parameters. In future there will be the option to calculate those from scratch.
# Currently relies on precomputed galaxy catalogues for each events' 99% confidence interval.

import numpy as np
#import time
from astropy import units as u
#from astropy.coordinates import SkyCoord
#from astropy import constants as const
from astropy.time import Time
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
#from scipy.integrate import quad, dblquad, nquad
from scipy.interpolate import splev #, splrep, interp1d, griddata
#from scipy import interp
import h5py
#import pickle
#import lal
#from lal import ComputeDetAMResponse
#import healpy as hp
#from matplotlib.font_manager import FontProperties
from gwcosmo import *
import argparse

parser = argparse.ArgumentParser(description='Compute the likelihood for a single MDC event.')

parser.add_argument('mdc_version', type=int, default=None, help='specify the version of the MDC to run (1,2,3)')
parser.add_argument('-H0min', '--H0min', type=float, default=0.25, help='set minimum value of H0/100 (default is 0.25)')
parser.add_argument('-H0max', '--H0max', type=float, default=1.5, help='set maximum value of H0/100 (default is 1.5)')
parser.add_argument('-H0step', '--H0step', type=int, default=50, help='set number of H0 values to calculate (default is 50)')
parser.add_argument('-n', '--event_no', type=int, default=None, help='specify the number of the event to analyse (1 to 250)')
parser.add_argument('-kde', '--kde_type', type=str, default='2D', help='specify whether the 3D kde or 2+1D kde should be used (3D,2D)') # currently only applicable for MDCv1
parser.add_argument('-mth', '--appm_threshold', type=float, default=None, help='specify the apparent magnitude threshold, if applicable (16,18,19.5)')
parser.add_argument('-plt', '--plot', type=str, default=False, help='if true, plot the likelihood as a function of H0')
parser.add_argument('-plt_path', '--plot_path', type=str, default=None, help='specify the location at which to save a plot of the likelihood')
parser.add_argument('-txt_path', '--txt_path', type=str, default=None, help='specify the location at which to save the likelihood')

args = parser.parse_args()
#parameter set up - work out which of these are necessary

H0_real = 0.7                         # the true Hubble constant
H0min = args.H0min
H0max = args.H0max
H0step = args.H0step
H0vec = np.linspace(H0min,H0max,H0step)      # a vector of H0 values
dH0 = H0vec[1]-H0vec[0]               # the H0 vec spacing 
alpha = -1.07                         # Schechter function parameter

n = args.event_no
mdc = args.mdc_version
kde = args.kde_type
mth = args.appm_threshold
plot = args.plot
plt_path = args.plot_path
txt_path = args.txt_path




if mdc == 1:
    pnW = np.zeros((H0vec.size, 1))
    den = splev(H0vec,mdc1_den)
    
    filename = '../mdc_prep/home/jveitch/src/first2years-data/2016/lalinference_mcmc/{}/posterior_samples.hdf5'.format(n) 
    group_name = 'lalinference_mcmc'
    dataset_name = 'posterior_samples'
    f1 = h5py.File(filename, 'r')
    group = f1[group_name]
    post = group[dataset_name]

    t = Time(np.mean(post['time']),format='gps')
    gmst = t.sidereal_time('mean','greenwich')
    gmstrad = gmst.to_value(u.rad)
  
    # read in the galaxy data for galaxies in this event's 99% credible region
    temp = np.genfromtxt('MDC_v1/sample_galaxies/event{}_gal.txt'.format(n))
    
    mdc_z = temp[:,4]
    mdc_ra = temp[:,2]
    mdc_dec = temp[:,3]
    
    if kde == '2D':
        skykernel = gaussian_kde([post['ra'],post['dec']])
        distkernel = gaussian_kde(post['dist'])
        
        for i,h in enumerate(H0vec):
            pnW[i] = px_H0G_num(mdc_z,None,h,gmstrad,distkernel=distkernel,skykernel=skykernel,catm=None,catra=mdc_ra,catdec=mdc_dec)/den[i]
            
    if kde == '3D':
        distskykernel = gaussian_kde([post['dist'],post['ra'],post['dec']]) 
        
        for i,h in enumerate(H0vec):
            pnW[i] = px_H0G_num_3D(mdc_z,None,h,gmstrad,distskykernel=distskykernel,catm=None,catra=mdc_ra,catdec=mdc_dec)/den[i]
            
    else:
        print 'The KDE type you have entered is invalid.  Please specify.'
            
    if txt_path != None:
        np.savetxt('{}/event{}_post.txt'.format(txt_path,n), pnW)



elif mdc == 2:
    #pnW = np.zeros((H0vec.size, 1))
    pxGnW = np.zeros(H0vec.size)
    pxnG = np.zeros(H0vec.size)
    
    
    filename = '../mdc_prep/home/jveitch/src/first2years-data/2016/lalinference_mcmc/{}/posterior_samples.hdf5'.format(n) 
    group_name = 'lalinference_mcmc'
    dataset_name = 'posterior_samples'
    f1 = h5py.File(filename, 'r')
    group = f1[group_name]
    post = group[dataset_name]

    t = Time(np.mean(post['time']),format='gps')
    gmst = t.sidereal_time('mean','greenwich')
    gmstrad = gmst.to_value(u.rad)

    # generate kd estimates for the distributions
    skykernel = gaussian_kde([post['ra'],post['dec']])
    distkernel = gaussian_kde(post['dist'])
    #distskykernel = gaussian_kde([post['dist'],post['ra'],post['dec']]) 
    
    if mth == 18:
        pG = splev(H0vec,pGtemp18)
        pxG_den = splev(H0vec,pxG_den_18)
        pxnG_den = splev(H0vec,pxnG_den21D_18)
        # read in the galaxy data for galaxies in this event's 99% credible region
        temp = np.genfromtxt('MDC_v2/sample_galaxies/event{}_gal.txt'.format(n))
        
    
    elif mth == 16:
        pG = splev(H0vec,pGtemp16)
        pxG_den = splev(H0vec,pxG_den_16)
        pxnG_den = splev(H0vec,pxnG_den21D_16)
        temp = np.genfromtxt('MDC_v2-2/sample_galaxies/event{}_gal.txt'.format(n))
    
    elif mth == 19.5:
        pG = splev(H0vec,pGtemp195)
        pxG_den = splev(H0vec,pxG_den_195)
        pxnG_den = splev(H0vec,pxnG_den21D_195)
        temp = np.genfromtxt('MDC_v2-3/sample_galaxies/event{}_gal.txt'.format(n))
    
    else:
        print('For MDC 2 you must specify an apparent magnitude threshold')
        exit()
    
    pnG = pnocat_detH0(None,pG)

    
    if not temp.any():
        print 'using an empty catalogue'
        for i,h in enumerate(H0vec):
            pxGnW[i] = 0
            pxnG[i] = px_H0nG_num(None,h,alpha,mth,distkernel=distkernel)/pxnG_den[i]
        
        
    elif temp.ndim == 1:
        print 'using a galaxy catalogue with 1 galaxy'
        mdc_z = np.array([temp[3]])
        mdc_ra = np.array([temp[1]])
        mdc_dec = np.array([temp[2]])
        
        for i,h in enumerate(H0vec):
            pxGnW[i] = px_H0G_num(mdc_z,None,h,gmstrad,distkernel=distkernel,skykernel=skykernel,catm=None,catra=mdc_ra,catdec=mdc_dec)/pxG_den[i]
            pxnG[i] = px_H0nG_num(None,h,alpha,mth,distkernel=distkernel)/pxnG_den[i]

        
    else:
        print 'using a galaxy catalogue with multiple galaxies'
        mdc_z = temp[:,3]
        mdc_ra = temp[:,1]
        mdc_dec = temp[:,2]

        for i,h in enumerate(H0vec):
            pxGnW[i] = px_H0G_num(mdc_z,None,h,gmstrad,distkernel=distkernel,skykernel=skykernel,catm=None,catra=mdc_ra,catdec=mdc_dec)/pxG_den[i]
            pxnG[i] = px_H0nG_num(None,h,alpha,mth,distkernel=distkernel)/pxnG_den[i]

    pnW = pG*pxGnW + pnG*pxnG
    

    if txt_path != None:
        np.savetxt('{}/event{}_post.txt'.format(txt_path,n), pnW)


##### Under construction. Test rigorously before use. #####
elif mdc == 3:
    pxGnW = np.zeros(H0vec.size)
    pxnG = np.zeros(H0vec.size)
    
    
    filename = '../mdc_prep/home/jveitch/src/first2years-data/2016/lalinference_mcmc/{}/posterior_samples.hdf5'.format(n) 
    group_name = 'lalinference_mcmc'
    dataset_name = 'posterior_samples'
    f1 = h5py.File(filename, 'r')
    group = f1[group_name]
    post = group[dataset_name]

    t = Time(np.mean(post['time']),format='gps')
    gmst = t.sidereal_time('mean','greenwich')
    gmstrad = gmst.to_value(u.rad)

    # generate kd estimates for the distributions
    skykernel = gaussian_kde([post['ra'],post['dec']])
    distkernel = gaussian_kde(post['dist'])


    ##### TO DO: Change (if needed) and test code for calculating luminosity weighted pxG_den, pxnG_den #####
    ##### TO DO: Change (if needed) and test code for calculating luminosity weighted px_H0G_num, px_H0nG_num #####
    
    pG = splev(H0vec,pGtemp14_weight)
    pxG_den = splev(H0vec,pxG_den_14_Lweights)
    pxnG_den = splev(H0vec,pxnG_den21D_14_Lweights)
    
    # read in the galaxy data for galaxies in this event's 99% credible region
    temp = np.genfromtxt('MDC_v3/sample_galaxies/event{}_gal.txt'.format(n))
    
    pnG = pnocat_detH0(None,pG)

    
    if not temp.any():
        print 'using an empty catalogue'
        for i,h in enumerate(H0vec):
            pxGnW[i] = 0
            
            ##### TO DO: Updata to include luminosity weighting #####
            pxnG[i] = px_H0nG_num(None,h,alpha,mth,distkernel=distkernel,weight=True)/pxnG_den[i]
        
        
    elif temp.ndim == 1:
        print 'using a galaxy catalogue with 1 galaxy'
        mdc_z = np.array([temp[3]])
        mdc_ra = np.array([temp[1]])
        mdc_dec = np.array([temp[2]])
        mdc_m = np.array([temp[4]])
        
        for i,h in enumerate(H0vec):
            ##### TO DO: Updata px_H0G_num, px_H0nG_num to include luminosity weighting #####
            pxGnW[i] = px_H0G_num(mdc_z,None,h,gmstrad,distkernel=distkernel,skykernel=skykernel,catm=mdc_m,catra=mdc_ra,catdec=mdc_dec)/pxG_den[i]
            pxnG[i] = px_H0nG_num(None,h,alpha,mth,distkernel=distkernel,weight=True)/pxnG_den[i]

        
    else:
        print 'using a galaxy catalogue with multiple galaxies'
        mdc_z = temp[:,3]
        mdc_ra = temp[:,1]
        mdc_dec = temp[:,2]
        mdc_m = temp[:,4]

        for i,h in enumerate(H0vec):
            ##### TO DO: Updata px_H0G_num, px_H0nG_num to include luminosity weighting #####
            pxGnW[i] = px_H0G_num(mdc_z,None,h,gmstrad,distkernel=distkernel,skykernel=skykernel,catm=mdc_m,catra=mdc_ra,catdec=mdc_dec)/pxG_den[i]
            pxnG[i] = px_H0nG_num(None,h,alpha,mth,distkernel=distkernel,weight=True)/pxnG_den[i]

    pnW = pG*pxGnW + pnG*pxnG
    

    if txt_path != None:
        np.savetxt('{}/event{}_post.txt'.format(txt_path,n), pnW)

#####





else:
    print('The MDC version entered is not a valid version number.  Please specify 1 or 2')


exit()

plt.figure()
plt.plot(H0vec,pnW/np.sum(pnW)/dH0,'-.r',label='no weights')
plt.axvline(0.7, label='True H0')
plt.xlabel('Hubble constant $H_{0}$')
plt.ylabel('$p(H_{0})$')
plt.xlim([H0vec[0],H0vec[-1]])
if plt_path != None:
    plt.savefig('{}/event{}_post.png'.format(plt_path,n))
if plot == 'True':
    plt.show()
plt.close()


