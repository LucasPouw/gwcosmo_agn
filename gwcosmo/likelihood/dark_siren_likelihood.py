"""
Multi-event likelihood Module
Rachel Gray

"""

import numpy as np

from scipy.integrate import simpson
from scipy.interpolate import interp1d
from gwcosmo.utilities.zprior_utilities import get_zprior_full_sky, get_zprior
from gwcosmo.likelihood.posterior_samples import *
import gwcosmo

from .skymap import ra_dec_from_ipix,ipix_from_ra_dec
import healpy as hp
import bilby
import h5py
import json
import ast
import sys
# import pickle
import copy

class PixelatedGalaxyCatalogMultipleEventLikelihood(bilby.Likelihood):
    """
    Class for preparing and carrying out the computation of the likelihood on 
    H0 for a single GW event
    """
    def __init__(self, posterior_samples_dictionary, injections, LOS_catalog_path, zrates, cosmo, mass_priors, min_pixels=30, sky_area=0.999, network_snr_threshold=11.,ifar_cut=0):

        """
        Parameters
        ----------
        samples : object
            GW samples
        skymap : gwcosmo.likelihood.skymap.skymap object
            provides p(x|Omega) and skymap properties
        LOS_prior :
        """
        super().__init__(parameters={'H0': None, 'gamma':None, 'Madau_k':None, 'Madau_zp':None, 'alpha':None, 'delta_m':None, 'mu_g':None, 'sigma_g':None, 'lambda_peak':None, 'alpha_1':None, 'alpha_2':None, 'b':None, 'mminbh':None, 'mmaxbh':None, 'alphans':None, 'mminns':None, 'mmaxns':None, 'beta':None, 'Xi0':None, 'n':None})

        self.zrates = zrates
        
        #TODO make min_pixels an optional dictionary
        LOS_catalog = h5py.File(LOS_catalog_path, 'r')
        temp = LOS_catalog.attrs['opts']
        catalog_opts = ast.literal_eval(temp.decode('utf-8'))
        nside = catalog_opts['nside']
        print(f'Chosen resolution nside: {nside}')
        self.z_array = LOS_catalog['z_array'][:]
        self.zprior_full_sky = get_zprior_full_sky(LOS_catalog)

        self.mass_priors = mass_priors
        self.cosmo = cosmo

        self.zprior_times_pxOmega_dict = {}
        self.pixel_indices_dictionary = {}
        self.samples_dictionary = {}
        self.samples_indices_dictionary = {}
        self.keys = []
                    
        for key, value in posterior_samples_dictionary.items():            
            try:
                samples = load_posterior_samples(posterior_samples_dictionary[key])
            except ValueError as ve:
                print("Error when loading posterior samples from file {}: {}".format(posterior_samples_dictionary[key],ve))
                sys.exit()

            if samples.skip_me:
                print("Skip event {} as requested by the user.".format(key))
                continue
            
            skymap = gwcosmo.likelihood.skymap.skymap(samples.skymap_path)
            low_res_skyprob = hp.pixelfunc.ud_grade(skymap.prob, nside, order_in='NESTED', order_out='NESTED')
            low_res_skyprob = low_res_skyprob/np.sum(low_res_skyprob)
            
            pixelated_samples = make_pixel_px_function(samples, skymap, npixels=min_pixels, thresh=sky_area)
            nside_low_res = pixelated_samples.nside
            if nside_low_res > nside:
                raise ValueError(f'Low resolution nside {nside_low_res} is higher than high resolution nside {nside}. Try decreasing min_pixels for event {key}.')

            # identify which samples will be used to compute p(x|z,H0) for each pixel
            pixel_indices = pixelated_samples.indices
            samp_ind ={}
            minsamps = 100 # default value
            for i,pixel_index in enumerate(pixel_indices):
                samp_ind[pixel_index] = pixelated_samples.identify_samples(pixel_index, minsamps=minsamps)
            
            no_sub_pix_per_pixel = int(4**(np.log2(nside/nside_low_res)))

            # Get the coordinates of the hi-res pixel centres
            pixra, pixdec = ra_dec_from_ipix(nside, np.arange(hp.pixelfunc.nside2npix(nside)), nest=True)
            # compute the low-res index of each of them
            ipix = ipix_from_ra_dec(nside_low_res, pixra, pixdec, nest=True)

            print('Loading the redshift prior')
            zprior_times_pxOmega = np.zeros((len(pixel_indices),len(self.z_array)))
            for i,pixel_index in enumerate(pixel_indices):
                # Find the hi res indices corresponding to the current coarse pixel
                hi_res_pixel_indices = np.arange(hp.pixelfunc.nside2npix(nside))[np.where(ipix==pixel_index)[0]]
                # load pixels, weight by GW sky area, and combine
                for j, hi_res_index in enumerate(hi_res_pixel_indices):
                	zprior_times_pxOmega[i,:] +=  get_zprior(LOS_catalog, hi_res_index)*low_res_skyprob[hi_res_index]
            print(f"Identified {len(pixel_indices)*no_sub_pix_per_pixel} pixels in the galaxy catalogue which correspond to {key}'s {sky_area*100}% sky area")
            self.zprior_times_pxOmega_dict[key] = zprior_times_pxOmega
            self.pixel_indices_dictionary[key] = pixel_indices
            self.samples_dictionary[key] = samples
            self.samples_indices_dictionary[key] = samp_ind
            self.keys.append(key)

        LOS_catalog.close()

        self.injections = injections
        # get the actual number of selected GW events entering the analysis, used for the check Neff >= 4Nobs inside the injection class
        self.injections.Nobs = len(self.keys)
        #TODO: add check that the snr threshold is valid for this set of injections
        self.injections.update_cut(snr_cut=network_snr_threshold,ifar_cut=ifar_cut)

        print(self.keys)


    def log_likelihood_numerator_single_event(self,event_name):

        pixel_indices = self.pixel_indices_dictionary[event_name]
        samples = self.samples_dictionary[event_name]
        samp_ind = self.samples_indices_dictionary[event_name]
        zprior = self.zprior_times_pxOmega_dict[event_name]
                
        # set up KDEs for this value of the parameter to be analysed
        px_zOmegaparam = np.zeros((len(pixel_indices),len(self.z_array)))
        for i,pixel_index in enumerate(pixel_indices):

            z_samps,m1_samps,m2_samps = self.reweight_samps.compute_source_frame_samples(samples.distance[samp_ind[pixel_index]],
                                                                                         samples.mass_1[samp_ind[pixel_index]],
                                                                                         samples.mass_2[samp_ind[pixel_index]])
            PEprior = samples.pe_priors[samp_ind[pixel_index]]

            zmin_temp = np.min(z_samps)*0.5
            zmax_temp = np.max(z_samps)*2.
            z_array_temp = np.linspace(zmin_temp,zmax_temp,100)
            
            kde,norm = self.reweight_samps.marginalized_redshift_reweight(z_samps,m1_samps,m2_samps,PEprior)

            if norm != 0: # px_zOmegaH0 is initialized to 0
                px_zOmegaparam_interp = interp1d(z_array_temp,kde(z_array_temp),kind='cubic',bounds_error=False,fill_value=0)
                px_zOmegaparam[i,:] = px_zOmegaparam_interp(self.z_array)*norm
        
        # make p(s|z) have the same shape as p(x|z,Omega,param) and p(z|Omega,s)
        ps_z_array = np.tile(self.zrates(self.z_array),(len(pixel_indices),1))
        
        Inum_vals = np.sum(px_zOmegaparam*zprior*ps_z_array,axis=0)
        num = simpson(Inum_vals,self.z_array)

        return np.log(num)
        
    def log_likelihood_denominator_single_event(self):

        z_prior = interp1d(self.z_array,self.zprior_full_sky*self.zrates(self.z_array),bounds_error=False,fill_value=(0,(self.zprior_full_sky*self.zrates(self.z_array))[-1]))
        dz=np.diff(self.z_array)
        z_prior_norm = np.sum((z_prior(self.z_array)[:-1]+z_prior(self.z_array)[1:])*(dz)/2)
        injections = copy.deepcopy(self.injections)
        
        # Update the sensitivity estimation with the new model
        injections.update_VT(self.cosmo,self.mass_priors,z_prior,z_prior_norm)
        Neff, Neff_is_ok, var = injections.calculate_Neff()
        if Neff_is_ok: # Neff >= 4*Nobs    
            log_den = np.log(injections.gw_only_selection_effect())
        else:
            print("Not enough Neff ({}) compared to Nobs ({}) for current mass-model {}, z-model {}, zprior_norm {}"
                  .format(Neff,injections.Nobs,self.mass_priors,z_prior,z_prior_norm))
            print("mass prior dict: {}, cosmo_prior_dict: {}".format(self.mass_priors_param_dict,self.cosmo_param_dict))
            print("returning infinite denominator")
            log_den = np.inf

        return log_den, np.log(z_prior_norm)
                       
    def log_combined_event_likelihood(self):
        
        # carry norm to apply to numerator as well
        den_single, zprior_norm_log = self.log_likelihood_denominator_single_event()
        den = den_single*len(self.keys)
        
        num = 1.
        for event_name in self.keys:
            num += self.log_likelihood_numerator_single_event(event_name)-zprior_norm_log

        return num-den
        
    def log_likelihood(self):

        self.zrates.gamma = self.parameters['gamma']
        self.zrates.k = self.parameters['Madau_k']
        self.zrates.zp = self.parameters['Madau_zp']
        
        self.mass_priors_param_dict = {'alpha':self.parameters['alpha'], 'delta_m':self.parameters['delta_m'], 
                                         'mu_g':self.parameters['mu_g'], 'sigma_g':self.parameters['sigma_g'], 
                                         'lambda_peak':self.parameters['lambda_peak'],
                                         'alpha_1':self.parameters['alpha_1'], 
                                         'alpha_2':self.parameters['alpha_2'], 'b':self.parameters['b'], 
                                         'mminbh':self.parameters['mminbh'], 'mmaxbh':self.parameters['mmaxbh'], 
                                         'beta':self.parameters['beta'], 'alphans':self.parameters['alphans'],
                                         'mminns':self.parameters['mminns'], 'mmaxns':self.parameters['mmaxns']}

        self.mass_priors.update_parameters(self.mass_priors_param_dict)

        self.cosmo_param_dict = {'H0': self.parameters['H0'], 'Xi0': self.parameters['Xi0'], 'n': self.parameters['n']}
        self.cosmo.update_parameters(self.cosmo_param_dict)

        self.reweight_samps = reweight_posterior_samples(self.cosmo,self.mass_priors)

        return self.log_combined_event_likelihood()
        
    def __call__(self):
        return np.exp(self.log_likelihood())


