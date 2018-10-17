#!/usr/bin/python
"""
This script computes H0 for the MDC using the counterparts.
Ignacio Magana
"""
# System imports
import os
import sys
from optparse import Option, OptionParser

#Global Imports
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']= 'Times New Roman'
matplotlib.rcParams['font.sans-serif']= ['Bitstream Vera Sans']
matplotlib.rcParams['text.usetex']= True
matplotlib.rcParams['mathtext.fontset']= 'stixsans'

import seaborn as sns
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')

import numpy as np
from scipy.stats import gaussian_kde
import gwcosmo

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Command line options
parser = OptionParser(
    description = __doc__,
    usage = "%prog [options]",
    option_list = [
        Option("-u", "--min_h0", metavar="MINH0", default='30.0', type=float,
            help="MINH0: Set minimum value of H0 Posterior"),
        Option("-v", "--max_h0", metavar="MAXH0", default='210.0', type=float,
            help="MAXH0: Set maximum value of H0 Posterior"),
        Option("-x", "--bins_h0", metavar="BINSH0", default='200', type=int,
            help="BINSH0: Set number of H0 Posterior bins"),
        Option("-d", "--min_dist", metavar="MINDIST", default='0.1', type=float,
            help="MINDIST: Set minimum value of luminosity distance"),
        Option("-e", "--max_dist", metavar="MAXDIST", default='400.0', type=float,
            help="MAXDIST: Set maximum value of luminosity distance"),
        Option("-f", "--bins_dist", metavar="BINSDIST", default='200', type=int,
            help="BINSDIST: Set number of luminosity distance bins"),
        Option("-z", "--z_max", metavar="ZMAX", default='1.0', type=float,
            help="ZMAX: Maximal detectable redshift"),
        Option("-y", "--use_3d_kde", metavar="KDE", default='False',
            help="KDE: Specify if 3D KDE is to be used. True by default."),
        Option("-p", "--plot", metavar="PLOT", default='True',
            help="PLOT: Plot results"),
        Option("-s", "--save", metavar="SAVE", default='True',
            help="SAVE: Save results"),
        Option("-b", "--outputfile", metavar="OUTPUTFILE", default='event_',
            help="OUTPUTFILE: Name of output file")
    ])
opts, args = parser.parse_args()
print(opts)
print(args)

# Check for missing required arguments
missing = []
for option in parser.option_list:
    if 'required' in option.help and eval('opts.' + option.dest) == None:
        missing.extend(option._long_opts)
if len(missing) > 0:
    parser.error('Missing required options: {0}'.format(str(missing)))
        
# Set command line arguments
min_h0 = float(opts.min_h0)
max_h0 = float(opts.max_h0)
bins_h0 = float(opts.bins_h0)

max_dist = float(opts.max_dist)
min_dist = float(opts.min_dist)
bins_dist = float(opts.bins_dist)

z_max = float(opts.z_max)

plot = str2bool(opts.plot)
save = str2bool(opts.save)

outputfile = str(opts.outputfile)

galaxy_weighting = False
completion = True
use_3d_kde = str2bool(opts.use_3d_kde)

counterparts = np.loadtxt('/home/ignacio.magana/src/gwcosmo/gwcosmo/data/catalog_data/mdc_counterparts.txt')
def main():
    "Compute P(H0)"
    H0 = np.linspace(min_h0, max_h0, bins_h0)
    dH0 = H0[1] - H0[0]
    
    #set up array of luminosity distance values
    dl = np.linspace(min_dist,max_dist,bins_dist)
    
    #set up detection probability for BNSs over the range dl
    dp = gwcosmo.likelihood.detection_probability.DetectionProbability(1.35,0.1,1.35,0.1,dl)
    
    galaxies_list = []
    for k in range(0,250):
        catalog = gwcosmo.catalog.galaxyCatalog()
        catalog.dictionary = {'0':gwcosmo.catalog.galaxy()}
        catalog.get_galaxy(0).ra = counterparts[k][2]
        catalog.get_galaxy(0).dec = counterparts[k][3]
        catalog.get_galaxy(0).z = counterparts[k][4]
        galaxies_list.append(catalog)
    
    # compute likelihood
    mth=25
    likelihoods = []
    for k in range(0,250):
        samples = gwcosmo.likelihood.posterior_samples.posterior_samples()
        samples.load_posterior_samples_hdf5('/home/ignacio.magana/first2years-data/2016/lalinference_mcmc/' \
                                                                +str(k+1)+'/posterior_samples.hdf5')
        me = gwcosmo.master.MasterEquation(H0,galaxies_list[k],dp,mth,linear=True,weighted=galaxy_weighting)
        likelihood = me.likelihood(samples,complete=completion,skymap2d=None,use_3d_kde=use_3d_kde)
        likelihoods.append(likelihood)

    prior = me.pH0_D(prior='jeffreys')
    
    posteriors=[] 
    posteriors_norm=[]
    priors_norm=[]
    for k in range(0,250):
        posterior = prior*likelihoods[k]
        posteriors.append(posterior)
        posteriors_norm.append(posterior/np.sum(posterior)/dH0)
        priors_norm.append(prior/np.sum(prior)/dH0)

    if plot == True:
        for k in range(0,250):
            plt.figure()
            plt.plot(H0,priors_norm[k],ls=':', linewidth = 3.0, label='prior')
            plt.plot(H0,posteriors_norm[k],linewidth = 3.0, label='posterior')
            plt.plot(H0,likelihoods[k],linewidth = 3.0, ls = '--', label='likelihood')
            plt.axvline(70,ls='--', c='k', label = r'$H_0 = 70$ (km s$^{-1}$ Mpc$^{-1}$)')
            plt.xlim(min_h0,max_h0)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=16)
            plt.ylabel(r'$p(H_0)$ (km$^{-1}$ s Mpc)', fontsize=16)
            plt.legend(loc='best',fontsize=16)
            plt.tight_layout()
            plt.savefig(outputfile+str(k+1)+'.png',dpi=400)
            plt.close()

    if save == True:
        for k in range(0,250):
            np.savez(outputfile+str(k+1)+'.npz',[H0,likelihoods[k],prior,posteriors_norm[k]])

if __name__ == "__main__":
    main()