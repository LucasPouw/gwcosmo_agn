'''
Multi-parameter likelihood Module for GW events with 
electromagnetic counterpart

Tathagata Ghosh
'''

import numpy as np
from scipy.integrate import simpson, quad
from scipy.interpolate import interp1d
from gwcosmo.utilities.cosmology import standard_cosmology
from gwcosmo.likelihood.posterior_samples import *
from gwcosmo.likelihood.skymap import *
import gwcosmo
import bilby
import pickle
import copy
from scipy.stats import truncnorm
import sys

   
class MultipleEventLikelihoodEM(bilby.Likelihood):

    def __init__(self, counterpart_dictionary, injections, zrates, cosmo, mass_priors, posterior_samples_dictionary=None, posterior_samples_field=None, skymap_dictionary=None,  network_snr_threshold=12., post_los_dictionary=None, nsamps=1000, skymap_prior_distance="dlSquare", skymap_H0=70, skymap_Omega_m=0.3065):

        """
        Class to calculate log-likelihood on cosmological and population hyper-parameters.

        parameters
        ------------------

        counterpart_dictionary : dictionary
            Dictionary of elctomagnetic counterpart information of GW events.
                Structure of the dictionary: {"GW170817": {"redshift": [3017,166]/c,"ra_dec": [3.44602385, -0.40813555]}}
                                             [3017,166]/c=[mu/c, sigma/c] : mu, sigma from Gaussian distribution; c: speed of light in km/s.
        injections : injection object
            The injection object from gwcosmo to calculate selection effects.
                Pre-computed by using gwcosmo.gwcosmo.create_injections.py 
        zrates : gwcosmo.utilities.host_galaxy_merger_relations object
            Object of merger rate evolution with redshift.
        cosmo : gwcosmo.utilities.cosmology object
            Object of cosmological model.
        mass_priors : gwcosmo.prior.priors object
            Object of mass model: For example, BNS, NSBH_powerlaw.
        posterior_samples_dictionary : dictionary
            Dictionary to store the name of files containing posterior samples for GW events.
                Structure of the dictionary: {"GW170817": "GW170817_GWTC-1.hdf5"}
        skymap_dictionary : dictionary
            Dictionary to store the name of skymap files corresponding to GW events.
                Structure of the dictionary: {"GW170817": "GW170817_skymap.fits.gz"}
        network_snr_threshold : float
            Network SNR threshold of GW events which are used for analysis.
        post_los_dictionary : dictionary
            Dictionary to store the information whether posterior samples of GW events are conditioned over line of sight or not.
                Structure of the dictionary: {"GW170817": "True"}
        nsamps : dictionary
            Number of samples are considered from posterior samples when post_los is False.
        skymap_prior_distance : str 
            Prior used during construction of skymap. Default is "dlSquare". Other options are "UniformComoving" and "UniformComoving".
        skymap_H0 : float
            Assumed H0 while constructing skymap if skymap_prior_distance is "UniformComoving".
        skymap_Omega_m : float
            Assumed Omega_m (if Flat Lambda CDM cosmology)  while constructing skymap if skymap_prior_distance is "UniformComoving".
        """

        super().__init__(parameters={'H0': None, 'Xi0': None, 'n': None, 'gamma':None, 'Madau_k':None, 'Madau_zp':None, 'alpha':None, 'delta_m':None, 'mu_g':None, 'sigma_g':None, 'lambda_peak':None, 'alpha_1':None, 'alpha_2':None, 'b':None, 'mminbh':None, 'mmaxbh':None, 'beta':None, 'alphans':None, 'mminns':None, 'mmaxns':None})

        # em couterpart information
        self.counterpart_dictionary = counterpart_dictionary
        #redshift evolution model
        self.zrates = zrates

        #selection effect
        self.injections = injections
        self.injections.update_cut(snr_cut=network_snr_threshold,ifar_cut=ifar_cut))
        try:
            self.injections.Nobs = len(list(posterior_samples_dictionary.keys())) # it's the number of GW events entering the analysis, used for the check Neff >= 4Nobs inside the injection class
        except:
            self.injections.Nobs = len(list(skymap_dictionary.keys()))
        
        #mass distribution
        self.mass_priors = mass_priors

        #cosmology
        self.cosmo = cosmo

        # prior redshift distribution: uniform in comoving volume
        self.zprior = self.cosmo.p_z


        self.post_los_dictionary = post_los_dictionary
        self.sample_index_dictionary = {}
        self.nsamps = nsamps

        # counterpart information
        self.counterpart_zmin_zmax_dictionary = {}
        self.counterpart_pdf_dictionary = {}

        self.samples_dictionary = {}

        self.keys = []

        # likelihood in distance from skymap for em counterpart
        self.posterior_dl_skymap = {}
        self.skymap_prior_distance = skymap_prior_distance
        self.dlarray = {}


        # sample and skymap dictionary
        self.posterior_samples_dictionary = posterior_samples_dictionary
        self.skymap_dictionary = skymap_dictionary

        for key in self.counterpart_dictionary.keys():

            counterpart_muz, counterpart_sigmaz = self.counterpart_dictionary[key]["redshift"]
            zmin = counterpart_muz - 5*counterpart_sigmaz
            if zmin<0: zmin=0
            zmax = counterpart_muz + 5*counterpart_sigmaz
            a = (zmin - counterpart_muz) / counterpart_sigmaz
            b = (zmax - counterpart_muz) / counterpart_sigmaz
            counterpart_pdf = truncnorm(a,b,counterpart_muz,counterpart_sigmaz)
            self.counterpart_pdf_dictionary[key] = counterpart_pdf
            self.counterpart_zmin_zmax_dictionary[key] = np.array([zmin,zmax])
 
        if posterior_samples_dictionary is not None:
            for key, value in posterior_samples_dictionary.items():

                self.keys.append(key)

                samples = load_posterior_samples(posterior_samples_dictionary[key])
                self.samples_dictionary[key] = samples
                if self.post_los_dictionary[key] is False:
                    ra_los, dec_los = self.counterpart_dictionary[key]["ra_dec"]
                    if type(self.nsamps) is dict and key in self.nsamps :
                        nsamp_event = int(self.nsamps [key])
                    else :
                        nsamp_event = int(self.nsamps)
                    sample_index, ang_rad_max = identify_samples_from_posterior (ra_los,dec_los,samples.ra,samples.dec,nsamp_event)
                    self.sample_index_dictionary [key] = sample_index
                    print (f"Considering {nsamp_event} samples around line of sight for {key}")

        elif skymap_dictionary is not None:

            for key, value in skymap_dictionary.items():
					
                skymap = gwcosmo.likelihood.skymap.skymap(skymap_dictionary[key])
                ra_los, dec_los = self.counterpart_dictionary[key]["ra_dec"]
                dlmin, dlmax, dlpost = skymap.lineofsight_posterior_dl(ra_los,dec_los)
                self.posterior_dl_skymap [key] = dlpost 
                if dlmin>0:
                    dl_array = np.linspace (dlmin,dlmax,10000)
                else :
                    dl_array = np.linspace (0,dlmax,10000)
                self.dlarray [key] = dl_array
                self.keys.append(key)

            if self.skymap_prior_distance=="UniformComoving" :
                zmin = 0 
                zmax = 10 
                cosmo_skymap = standard_cosmology (skymap_H0, skymap_Omega_m)
                z_array = np.linspace(zmin, zmax, 10000)
                dl_array = cosmo_skymap.dgw_z (z_array)
                z_prior_skymap = cosmo_skymap.p_z(z_array)
                self.dl_prior_skymap = interp1d (dl_array, z_prior_skymap)

        # get the actual number of selected GW events entering the analysis, used for the check Neff >= 4Nobs inside the injection class
        self.injections.Nobs = len(list(self.samples_dictionary.keys()))


    def log_likelihood_numerator_single_event_from_samples(self,event_name):
	
        samples = self.samples_dictionary[event_name]
        z_samps,m1_samps,m2_samps = self.reweight_samps.compute_source_frame_samples(samples.distance, samples.mass_1, samples.mass_2)
        PEprior = samples.pe_priors
        if self.post_los_dictionary[event_name]:
            kde,norm = self.reweight_samps.marginalized_redshift_reweight(z_samps,m1_samps,m2_samps,PEprior)
        else :
            sample_index = self.sample_index_dictionary [event_name]
            kde,norm = self.reweight_samps.marginalized_redshift_reweight(z_samps[sample_index],m1_samps[sample_index],m2_samps[sample_index],
                                                                          PEprior[sample_index])

        redshift_bins = 1000
        zmin = self.cosmo.z_dgw(np.amin(samples.distance))*0.5
        zmax = self.cosmo.z_dgw(np.amax(samples.distance))*2.
        z_array_temp = np.linspace(zmin,zmax,redshift_bins)
        px_zOmegaH0_interp = interp1d(z_array_temp,kde(z_array_temp),kind='linear',bounds_error=False,fill_value=0)  # interpolation may produce some -ve values when kind='cubic'  
        num_x = lambda x : px_zOmegaH0_interp (x)*self.zrates(x)*self.counterpart_pdf_dictionary [event_name].pdf (x)
        zmin, zmax = self.counterpart_zmin_zmax_dictionary [event_name]
        num, _ = quad(num_x,zmin,zmax)

        return np.log(num*norm)
		
    def log_likelihood_numerator_single_event_from_skymap(self,event_name):

        redshift_bins = 10000
        zmin = self.cosmo.z_dgw(self.dlarray [event_name] [0])*0.5
        zmax = self.cosmo.z_dgw(self.dlarray [event_name] [-1])*2.
        z_array_temp = np.linspace(zmin,zmax,redshift_bins)

        dlarr_given_H0 = self.cosmo.dgw_z(z_array_temp)

        if self.skymap_prior_distance=="dlSquare" :
            likelihood_x_z_H0= self.posterior_dl_skymap[event_name].pdf(dlarr_given_H0)/dlarr_given_H0**2
        elif self.skymap_prior_distance=="Uniform" :
            likelihood_x_z_H0= self.posterior_dl_skymap[event_name].pdf(dlarr_given_H0)
        elif self.skymap_prior_distance=="UniformComoving" :
            likelihood_x_z_H0= self.posterior_dl_skymap[event_name].pdf(dlarr_given_H0)/self.dl_prior_skymap(dlarr_given_H0)
        likelihood_x_z_H0 /= simpson(likelihood_x_z_H0, z_array_temp)

        px_zOmegaH0_interp = interp1d(z_array_temp,likelihood_x_z_H0,kind='linear',bounds_error=False,fill_value=0)

        num_x = lambda x : px_zOmegaH0_interp(x)*self.zrates(x)*self.counterpart_pdf_dictionary[event_name].pdf(x) 
        zmin, zmax = self.counterpart_zmin_zmax_dictionary[event_name]
        num, _ = quad(num_x, zmin, zmax)

        return np.log(num)
	

    def log_likelihood_denominator_single_event(self):

        zmin = 0 
        zmax = 10 
        z_array = np.linspace(zmin, zmax, 10000)
        z_prior = interp1d(z_array,self.zprior(z_array)*self.zrates(z_array))
        dz=np.diff(z_array)
        z_prior_norm = np.sum((z_prior(z_array)[:-1]+z_prior(z_array)[1:])*(dz)/2)
        injections = copy.deepcopy(self.injections)
        
        # Update the sensitivity estimation with the new model
        injections.update_VT(self.cosmo,self.mass_priors,z_prior,z_prior_norm)
        Neff, Neff_is_ok, var = injections.calculate_Neff()
        if Neff_is_ok: # Neff >= 4*Nobs    
            log_den = np.log(injections.gw_only_selection_effect())
        else:
            print("Not enough Neff ({}) compared to Nobs ({}) for current mass-model {} and z-model {}".format(Neff,injections.Nobs,self.mass_priors,z_prior))
            print("mass prior dict: {}, cosmo_prior_dict: {}".format(self.mass_priors_param_dict,self.cosmo_param_dict))
            print("returning infinite denominator")
            print("exit!")
            log_den = np.inf
            #sys.exit()
            
        return log_den, np.log(z_prior_norm)

                       
    def log_combined_event_likelihood(self):
        
        num = 1.

        den_single, zprior_norm_log = self.log_likelihood_denominator_single_event()
        den = den_single*len(self.keys)

        if self.posterior_samples_dictionary is not None:
            for event_name in self.keys:
                num += self.log_likelihood_numerator_single_event_from_samples(event_name)-zprior_norm_log
        elif self.posterior_samples_dictionary is None and self.skymap_dictionary is not None :
            for event_name in self.keys:
                num += self.log_likelihood_numerator_single_event_from_skymap(event_name)-zprior_norm_log

        return num-den

    def log_likelihood(self):

        self.cosmo_param_dict = {'H0': self.parameters['H0'], 'Xi0': self.parameters['Xi0'], 'n': self.parameters['n']}
        self.cosmo.update_parameters(self.cosmo_param_dict)

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


        if self.posterior_samples_dictionary is not None:
            self.reweight_samps = reweight_posterior_samples(self.cosmo,self.mass_priors)
            return self.log_combined_event_likelihood()
            
        elif self.posterior_samples_dictionary is None and self.skymap_dictionary is not None :
            return self.log_combined_event_likelihood()	

    def __call__(self):
        return np.exp(self.log_likelihood())
