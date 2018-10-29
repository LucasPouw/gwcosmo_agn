#!/usr/bin/python
"""
This script computes H0 as a function of H0 bins.
Ignacio Magana, Rachel Gray, Ankan Sur
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
        Option("-m", "--method", metavar="METHOD", default=None,
            help="METHOD: Select counterpart/statistical (required)"),
        Option("-u", "--min_h0", metavar="MINH0", default='30.0', type=float,
            help="MINH0: Set minimum value of H0 Posterior"),
        Option("-v", "--max_h0", metavar="MAXH0", default='210.0', type=float,
            help="MAXH0: Set maximum value of H0 Posterior"),
        Option("-x", "--bins_h0", metavar="BINSH0", default='50', type=int,
            help="BINSH0: Set number of H0 Posterior bins"),
        Option("-d", "--min_dist", metavar="MINDIST", default='0.1', type=float,
            help="MINDIST: Set minimum value of luminosity distance"),
        Option("-e", "--max_dist", metavar="MAXDIST", default='400.0', type=float,
            help="MAXDIST: Set maximum value of luminosity distance"),
        Option("-f", "--bins_dist", metavar="BINSDIST", default='50', type=int,
            help="BINSDIST: Set number of luminosity distance bins"),
        Option("-z", "--z_max", metavar="ZMAX", default='1.0', type=float,
            help="ZMAX: Maximal detectable redshift"),
        Option("-k", "--posterior_samples", metavar="SAMPLES", default=None,
            help="SAMPLES: LALinference posterior samples file in format (.dat or hdf5) or use GW170817, GW170814, GW170818"),
        Option("-t", "--mass_distribution", metavar="MASS_DISTRIBUTION", default=None,
            help="MASS_DISTRIBUTION: Choose between BNS or BBH mass distributions for Pdet calculation."),
        Option("-y", "--use_3d_kde", metavar="KDE", default='True',
            help="KDE: Specify if 3D KDE is to be used. True by default."),
        Option("-i", "--skymap", metavar="SKYMAP", default=None,
            help="SKYMAP: LALinference 3D skymap file in format (.fits)"),
        Option("-g", "--galaxy_catalog", metavar="CATALOG", default=None,
            help="CATALOG: Galaxy catalog file in hdf5 format"),
        Option("-l", "--galaxy_catalog_default", metavar="CATALOG_DEFAULT", default=None,
            help="CATALOG_DEFAULT: Load default galaxy catalog. Options: glade or mdc"),
        Option("-n", "--glade_version", metavar="GLADE", default=None,
            help="GLADE: Version of GLADE catalog available. Options: corrected, original, maya, hong."),
        Option("-q", "--mdc_version", metavar="MDC", default=None,
            help="MDC: Version of MDC catalogs available. Options: 1.0, 2.1, 2.2, 2.3, 3.1"),  
        Option("-w", "--galaxy_weighting", metavar="WEIGHTING", default='False',
            help="WEIGHTING: Set galaxy catalog weighting"),
        Option("-c", "--completion", metavar="COMPLETENESS", default='False',
            help="COMPLETENESS: Set galaxy catalog completion function."),
        Option("-r", "--counterpart_ra", metavar="RA", default=None,
            help="RA: Right ascension of counterpart"),
        Option("-o", "--counterpart_dec", metavar="DEC", default=None,
            help="DEC: Declination of counterpart"),
        Option("-j", "--counterpart_z", metavar="REDSHIFT", default=None,
            help="REDSHIFT: Redshift of counterpart"),
        Option("-p", "--plot", metavar="PLOT", default='True',
            help="PLOT: Plot results"),
        Option("-s", "--save", metavar="SAVE", default='True',
            help="SAVE: Save results"),
        Option("-b", "--outputfile", metavar="OUTPUTFILE", default='Posterior',
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

print('Selected method is:', opts.method)

if (opts.posterior_samples is None and
    opts.skymap is None):
        parser.error('Provide either posterior samples or skymap.')
        
if opts.mass_distribution is None:
        parser.error('Provide a mass distribution to use for Pdet calculation.')
        
if opts.method == 'statistical':
    if (opts.galaxy_catalog is None and 
        opts.galaxy_catalog_default is None):
        parser.error('The statistical method requires a galaxy catalog. Provide one or use a default catalog.')
    if opts.galaxy_catalog_default == 'glade':
        if opts.glade_version is None:
            parser.error('Provide version of GLADE catalog to use.')
    if opts.galaxy_catalog_default == 'mdc':
        if opts.mdc_version is None:
            parser.error('Provide version of MDC catalog to use.')
            
if opts.method == 'counterpart':
    if (opts.counterpart_ra is None or
    opts.counterpart_dec is None or
    opts.counterpart_z is None):
        parser.error('The counterpart method requires the ra, dec, and z of the galaxy.')
        
if opts.posterior_samples is not None:
        samples_file_path = str(opts.posterior_samples)
        use_3d_kde = str2bool(opts.use_3d_kde)
if opts.skymap is not None:
        skymap_file_path = str(opts.skymap)

if opts.mass_distribution is not None:
        mass_distribution = str(opts.mass_distribution)

if opts.method == 'statistical':
    if opts.galaxy_catalog is not None:
        galaxy_catalog_path = str(opts.galaxy_catalog)
    if opts.galaxy_catalog_default is not None:
        galaxy_catalog_default = str(opts.galaxy_catalog_default)
        if opts.glade_version is not None:
            glade_version = str(opts.glade_version)
        if opts.mdc_version is not None:
            mdc_version = str(opts.mdc_version)

    galaxy_weighting = str2bool(opts.galaxy_weighting)
    completion = str2bool(opts.completion)

if opts.method == 'counterpart':
    galaxy_weighting = False
    completion = True
    if opts.counterpart_ra is not None:
        counterpart_ra = float(opts.counterpart_ra)
    if opts.counterpart_dec is not None:
        counterpart_dec = float(opts.counterpart_dec)
    if opts.counterpart_z is not None:
        counterpart_z = float(opts.counterpart_z)
        
# Set command line arguments
min_h0 = float(opts.min_h0)
max_h0 = float(opts.max_h0)
bins_h0 = float(opts.bins_h0)

max_dist = float(opts.max_dist)
min_dist = float(opts.min_dist)
bins_dist = float(opts.bins_dist)

z_max = float(opts.z_max)

options_string = opts.method

plot = str2bool(opts.plot)
save = str2bool(opts.save)

outputfile = str(opts.outputfile)

def main():
    "Compute P(H0)"
    H0 = np.linspace(min_h0, max_h0, bins_h0)
    dH0 = H0[1] - H0[0]
    
    samples = gwcosmo.likelihood.posterior_samples.posterior_samples()
    if samples_file_path == 'GW170817':
        samples.load_posterior_samples(event=samples_file_path)
    if samples_file_path == 'GW170818':
        samples.load_posterior_samples(event=samples_file_path)
    else:    
        samples.load_posterior_samples_hdf5(samples_file_path)
    
    catalog = gwcosmo.catalog.galaxyCatalog()
    if opts.method == 'counterpart':
        catalog.load_counterpart_catalog(counterpart_ra, counterpart_dec, counterpart_z)
        
    if opts.method == 'statistical':
        if galaxy_catalog_default == 'glade':
            catalog.load_glade_catalog(version=glade_version)
        if galaxy_catalog_default == 'mdc':
            catalog.load_mdc_catalog(version=mdc_version)
            
    #set up array of luminosity distance values
    dl = np.linspace(min_dist,max_dist,bins_dist)
    
    #set up detection probability for BNSs over the range dl
    dp = gwcosmo.likelihood.detection_probability.DetectionProbability(mass_distribution,dl)
    
    # compute likelihood
    me = gwcosmo.master.MasterEquation(H0,catalog,dp,linear=True,weighted=galaxy_weighting,use_3d_kde=use_3d_kde)
    
    likelihood = me.likelihood(samples,complete=completion,skymap2d=None)

    prior = me.pH0_D(prior='jeffreys')

    posterior = prior*likelihood

    posterior_norm = posterior/np.sum(posterior)/dH0
    prior_norm = prior/np.sum(prior)/dH0

    if plot == True:
        plt.figure()
        plt.plot(H0,prior_norm,ls=':', linewidth = 3.0, label='prior')
        plt.plot(H0,posterior_norm,linewidth = 3.0, label='posterior')
        plt.plot(H0,likelihood,linewidth = 3.0, ls = '--', label='likelihood')
        plt.axvline(70,ls='--', c='k', label = r'$H_0 = 70$ (km s$^{-1}$ Mpc$^{-1}$)')
        plt.xlim(min_h0,max_h0)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=16)
        plt.ylabel(r'$p(H_0)$ (km$^{-1}$ s Mpc)', fontsize=16)
        plt.legend(loc='best',fontsize=16)
        plt.tight_layout()
        plt.savefig(outputfile+'.png',dpi=400)

    if save == True:
        np.savez(outputfile+'.npz',[H0,likelihood,prior,posterior_norm])

if __name__ == "__main__":
    main()