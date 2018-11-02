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
        Option("-u", "--min_H0", metavar="MINH0", default='30.0', type=float,
            help="MINH0: Set minimum value of H0 Posterior"),
        Option("-v", "--max_H0", metavar="MAXH0", default='210.0', type=float,
            help="MAXH0: Set maximum value of H0 Posterior"),
        Option("-x", "--bins_H0", metavar="BINSH0", default='200', type=int,
            help="BINSH0: Set number of H0 Posterior bins"),
        Option("-d", "--min_dist", metavar="MINDIST", default='0.1', type=float,
            help="MINDIST: Set minimum value of luminosity distance"),
        Option("-e", "--max_dist", metavar="MAXDIST", default='400.0', type=float,
            help="MAXDIST: Set maximum value of luminosity distance"),
        Option("-f", "--bins_dist", metavar="BINSDIST", default='200', type=int,
            help="BINSDIST: Set number of luminosity distance bins"),
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
min_H0 = float(opts.min_H0)
max_H0 = float(opts.max_H0)
bins_H0 = float(opts.bins_H0)

max_dist = float(opts.max_dist)
min_dist = float(opts.min_dist)
bins_dist = float(opts.bins_dist)

plot = str2bool(opts.plot)
save = str2bool(opts.save)

outputfile = str(opts.outputfile)

galaxy_weighting = False
completion = True
use_3d_kde = str2bool(opts.use_3d_kde)

def main():
    "Compute P(H0)"
    H0 = np.linspace(min_H0, max_H0, bins_H0)
    dH0 = H0[1] - H0[0]
    
    #set up array of luminosity distance values
    dl = np.linspace(min_dist,max_dist,bins_dist)
    
    #set up detection probability for BNSs over the range dl
    dp = gwcosmo.likelihood.detection_probability.DetectionProbability('BNS',dl)
    
    counterpart = gwcosmo.catalog.galaxyCatalog()
    counterpart.dictionary = {'0':gwcosmo.catalog.galaxy()}
    counterpart.get_galaxy(0).ra = 197.4500*np.pi/180.
    counterpart.get_galaxy(0).dec = -23.3839*np.pi/180.
    counterpart.get_galaxy(0).z = 0.009727
    
    # compute likelihood
    samples = gwcosmo.likelihood.posterior_samples.posterior_samples()
    samples.load_posterior_samples('GW170817')
    me = gwcosmo.master.MasterEquation(H0,counterpart,dp,linear=True,weighted=galaxy_weighting,use_3d_kde=use_3d_kde,counterparts=True)
    likelihood = me.likelihood(samples,complete=completion,skymap2d=None)

    prior = me.pH0(prior='log')
    
    posterior = prior*likelihood
    posterior_norm = posterior/np.sum(posterior*dH0)
    prior_norm = prior/np.sum(prior*dH0)

    if plot == True:
        plt.figure()
        plt.plot(H0,prior_norm,ls=':', linewidth = 3.0, label='prior')
        plt.plot(H0,posterior_norm,linewidth = 3.0, label='posterior')
        plt.plot(H0,likelihood,linewidth = 3.0, ls = '--', label='likelihood')
        plt.axvline(70,ls='--', c='k', label = r'$H_0 = 70$ (km s$^{-1}$ Mpc$^{-1}$)')
        plt.xlim(min_H0,max_H0)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=16)
        plt.ylabel(r'$p(H_0)$ (km$^{-1}$ s Mpc)', fontsize=16)
        plt.legend(loc='best',fontsize=16)
        plt.tight_layout()
        plt.savefig(outputfile+'.png',dpi=400)
        plt.close()

    if save == True:
        np.savez(outputfile+'.npz',[H0,likelihood,prior,posterior_norm])

if __name__ == "__main__":
    main()