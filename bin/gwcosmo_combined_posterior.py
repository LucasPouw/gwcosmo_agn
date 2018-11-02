#!/usr/bin/python
"""
This script combines individual H0 posteriors.
Ignacio Magana, Rachel Gray
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
import imageio

# Command line options
parser = OptionParser(
    description = __doc__,
    usage = "%prog [options]",
    option_list = [
        Option("-d", "--dir", metavar="DIR", default=None,
            help="DIR: Directory to .npz gwcosmo_single_posterior files to be combined. (required)"),
        Option("-y", "--plot", metavar="PLOT", default='True',
            help="PLOT: Plot results"),
        Option("-c", "--makeconvergeplot", metavar="CONVERGENCE", default='True',
            help="CONVERGENCE: Make convergence plot for the mdc"),
        Option("-m", "--makemovie", metavar="MAKEMOVIE", default='True',
            help="MAKEMOVIE: Make movie for the mdc"),
        Option("-k", "--save", metavar="SAVE", default='True',
            help="SAVE: Save results"),
        Option("-b", "--outputfile", metavar="OUTPUTFILE", default='Posterior_mdc',
            help="OUTPUTFILE: Name of output file")
    ])
opts, args = parser.parse_args()

# Check for missing required arguments
missing = []
for option in parser.option_list:
    if 'required' in option.help and eval('opts.' + option.dest) == None:
        missing.extend(option._long_opts)
if len(missing) > 0:
    parser.error('Missing required options: {0}'.format(str(missing)))

dir = str(opts.dir)

dir_list = []
for path, subdirs, files in os.walk(dir):
    for name in files:
        filepath = os.path.join(path, name)
        if filepath[-4:] == '.npz':
            dir_list.append(filepath)
            
plot = bool(opts.plot)
makemovie = bool(opts.makemovie)
makeconvergeplot = bool(opts.makeconvergeplot)

save = bool(opts.save)
outputfile = str(opts.outputfile)

def main():
    "Compute combined P(H0)"

    Nevents = len(dir_list)
    
    H0 = np.load(dir_list[0])['arr_0'][0]
    min_H0 = H0[0]
    max_H0 = H0[-1]
    dH0 = H0[1] - H0[0]
    
    prior = np.load(dir_list[0])['arr_0'][2]
    
    likelihoods=[]
    for path in dir_list:
        likelihoods.append(np.load(path)['arr_0'][1])

    likelihood_comb = np.ones(H0.size)
    likelihood_comb_list = []
    for k in range(Nevents):
        likelihood_comb *= likelihoods[k]
        likelihood_comb = likelihood_comb/np.sum(likelihood_comb) #normalise within loop, else will go to 0.
        likelihood_comb_list.append(likelihood_comb)

    posterior = prior*likelihood_comb

    posterior_norm = posterior/np.sum(posterior*dH0)
    prior_norm = prior/np.sum(prior*dH0)

    if plot == True:
        plt.figure()
        plt.plot(H0,prior_norm,ls=':',linewidth = 3.0, label='prior')
        plt.plot(H0,posterior_norm,linewidth = 3.0, label='posterior')
        plt.plot(H0,likelihood_comb,linewidth = 3.0, ls = '--', label='likelihood')
        plt.axvline(70,ls='--', c='k', label = r'$H_0 = 70$ (km s$^{-1}$ Mpc$^{-1}$)')
        for n in range(Nevents):
            plt.plot(H0,likelihoods[n], alpha = 0.3)
        plt.xlim(min_H0,max_H0)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=16)
        plt.ylabel(r'$p(H_0)$ (km$^{-1}$ s Mpc)', fontsize=16)
        plt.legend(loc='best',fontsize=16)
        plt.tight_layout()
        plt.savefig(outputfile+'.png',dpi=400)
        plt.close()
        
        plt.figure()
        plt.plot(H0,prior_norm,ls=':',linewidth = 3.0, label='prior')
        plt.plot(H0,posterior_norm,linewidth = 3.0, label='posterior')
        plt.plot(H0,likelihood_comb,linewidth = 3.0, ls = '--', label='likelihood')
        plt.axvline(70,ls='--', c='k', label = r'$H_0 = 70$ (km s$^{-1}$ Mpc$^{-1}$)')
        for n in range(Nevents):
            plt.plot(H0,likelihoods[n], alpha = 0.3)
        plt.xlim(60,80)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=16)
        plt.ylabel(r'$p(H_0)$ (km$^{-1}$ s Mpc)', fontsize=16)
        plt.legend(loc='best',fontsize=16)
        plt.tight_layout()
        plt.savefig(outputfile+'_zoom.png',dpi=400)
        plt.close()
        
    if save == True:
        np.savez(outputfile+'.npz',[H0,likelihood_comb,prior,posterior_norm])
    
    if makeconvergeplot == True:
        error=[]
        sigma=[]
        N = np.arange(0,len(likelihood_comb_list))
        for i in range(0,len(likelihood_comb_list)):
            posterior = prior*likelihood_comb_list[i]
            posterior_norm = posterior/np.sum(posterior)/dH0
            sigma_H0_vals = H0[np.where(posterior_norm>max(posterior_norm)-1*np.std(posterior_norm))]
            error.append( sigma_H0_vals[-1] - sigma_H0_vals[0])
            sigma.append(np.mean(sigma_H0_vals))
        error=np.array(error)
        sigma=np.array(sigma)
        plt.figure()
        plt.errorbar(N, sigma, yerr=error, color='red', fmt='.', markersize='2', ecolor='red',capsize=1, elinewidth=1)
        plt.axhline(70,ls='--', c='k', label = r'$H_0 = 70$ (km s$^{-1}$ Mpc$^{-1}$)')
        plt.xlabel('Number of Events',fontsize=16)
        plt.ylabel(r'$H_0$ (km$^{-1}$ s Mpc)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc='best',fontsize=16)
        plt.tight_layout()
        plt.savefig(outputfile+'_convergence.png',dpi=400)
        plt.close()
        
        plt.figure()
        plt.errorbar(N, sigma, yerr=error, color='red', fmt='.', markersize='2', ecolor='red',capsize=1, elinewidth=1)
        plt.axhline(70,ls='--', c='k', label = r'$H_0 = 70$ (km s$^{-1}$ Mpc$^{-1}$)')
        plt.xlabel('Number of Events',fontsize=16)
        plt.ylabel(r'$\sigma_{H_0}$ (km$^{-1}$ s Mpc)', fontsize=16)
        plt.yscale('log')
        plt.xscale('log')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc='best',fontsize=16)
        plt.tight_layout()
        plt.savefig(outputfile+'_convergence_log.png',dpi=400)
        plt.close()
        
    if makemovie == True:
        for i in range(0,len(likelihood_comb_list)):
            plt.figure()
            plt.plot(H0,prior_norm/max(prior_norm), ls=':', linewidth = 3.0, label='prior')
            plt.plot(H0,prior*likelihood_comb_list[i]/max(prior*likelihood_comb_list[i]),linewidth = 3.0, label='posterior')
            plt.plot(H0,likelihood_comb_list[i]/max(likelihood_comb_list[i]),linewidth = 3.0, ls = '--', label='likelihood')
            plt.axvline(70,ls='--', c='k', label = r'$H_0 = 70$ (km s$^{-1}$ Mpc$^{-1}$)')
            plt.xlim(min_H0,max_H0)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=16)
            plt.ylabel(r'$p(H_0)$ (km$^{-1}$ s Mpc)', fontsize=16)
            plt.legend(loc='upper right',fontsize=16)
            plt.tight_layout()
            plt.savefig(dir+'/'+outputfile+'_'+str(i)+'.png',dpi=400)
            plt.close()
            
        filenames=[]
        for i in range(0,len(likelihood_comb_list)):
            filenames.append(dir+'/'+outputfile+'_'+str(i)+'.png')

        with imageio.get_writer(outputfile+'_movie.gif', mode='I', duration=0.2) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

if __name__ == "__main__":
    main()