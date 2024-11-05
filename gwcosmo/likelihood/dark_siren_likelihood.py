"""
Multi-event likelihood Module
Rachel Gray

"""

import numpy as np

from scipy.integrate import simpson, quad
from scipy.interpolate import interp1d
from gwcosmo.utilities.zprior_utilities import get_zprior_full_sky, get_zprior
from gwcosmo.likelihood.posterior_samples import *
from gwcosmo.utilities.mass_prior_utilities import extract_parameters_from_instance
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
from lal import C_SI
import gwcosmo.utilities.posterior_utilities as pu # get the PE samples keys
import inspect

light_speed_in_km_per_sec = C_SI/1000.

class PixelatedGalaxyCatalogMultipleEventLikelihood(bilby.Likelihood):
    """
    Class for preparing and carrying out the computation of the likelihood on
    H0 for a single GW event
    """
    def __init__(self, posterior_samples_dictionary,
                 injections,
                 LOS_catalog_path,
                 zrates,
                 cosmo,
                 mass_priors,
                 min_pixels=30,
                 min_samps_in_pixel=100.,
                 sky_area=0.999,
                 network_snr_threshold=11.,
                 ifar_cut=0):

        """
        Parameters
        ----------
        samples : object
            GW samples
        skymap : gwcosmo.likelihood.skymap.skymap object
            provides p(x|Omega) and skymap properties
        LOS_prior : object 
        """

        mass_prior_params = extract_parameters_from_instance(mass_priors)
        
        for param_name in mass_prior_params:
            mass_prior_params[param_name] = None

        super().__init__(parameters={
            'H0': None,
            'gamma':None,
            'Madau_k':None,
            'Madau_zp':None,
            'Xi0':None,
            'n':None,
            'D':None,
            'logRc':None,
            'nD':None,
            'cM':None,
            **mass_prior_params })
        
        
        self.zrates = zrates
        self.mass_prior_params = mass_prior_params

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
            # check average dL for this event
            avg_dL = np.mean(samples.distance)
            std_dL = np.std(samples.distance)
            
            if not pu.PE_min_pixels in posterior_samples_dictionary[key]: # search for the "min_pixels" key
                posterior_samples_dictionary[key][pu.PE_min_pixels] = min_pixels # add key and use the global value for min_pixels if not specific to the current GW event
            else:
                posterior_samples_dictionary[key][pu.PE_min_pixels] = int(posterior_samples_dictionary[key][pu.PE_min_pixels])

            print("Event {}: requested min_pixels = {}".format(key,posterior_samples_dictionary[key][pu.PE_min_pixels]))

            skymap = gwcosmo.likelihood.skymap.skymap(samples.skymap_path)
            print("Got GW event skymap with nside: {}".format(skymap.nside))

            low_res_skyprob = hp.pixelfunc.ud_grade(skymap.prob, nside, order_in='NESTED', order_out='NESTED')
            low_res_skyprob = low_res_skyprob/np.sum(low_res_skyprob)
            
            # compute the sky area covered by the GW event and find an estimate of the min_pixels value to help the user
            skyprob = low_res_skyprob
            if skymap.nside > 4*nside:
                # when the GW skymap has a much higher resolution than the LOS skymap, degrade it but not too much, use new_nside = 4*LOS_nside
                # to get a quite precise estimation of the GW area on the sky and avoid at the same time to work on the full resolution skymap that may take some time
                skyprob = hp.pixelfunc.ud_grade(skymap.prob, 4*nside, order_in='NESTED', order_out='NESTED')
                skyprob = skyprob/np.sum(skyprob)
            
            gw_probs = np.sort(skyprob)[::-1] # sort with decreasing order
            csp = np.cumsum(gw_probs)
            dd = np.abs(csp-sky_area)
            npix_area = np.where(dd == np.min(dd))[0] # number of pixels needed to cover the GW's sky_area
            sky_area_square_rad = npix_area[0]*4*np.pi/len(skyprob)
            sky_area_square_deg = sky_area_square_rad*(180/np.pi)**2            
            n_pixels_at_LOS_resolution = 12*nside**2*sky_area_square_rad/(4*np.pi)
            print("{}: avg(dL): {:.2f} Mpc, std(dL): {:.2f} Mpc. Sky area with proba closest to {}: {:.2f} sq. deg., corresponding to {:.2f} pixels at LOS file resolution (nside: {}).".format(key,avg_dL,std_dL,sky_area,sky_area_square_deg,n_pixels_at_LOS_resolution,nside))
            print("A reasonable value for min_pixels is around {0:.2f} (mayber smaller), depending on the exact GW skymap.".format(n_pixels_at_LOS_resolution))
            n_pixels_warning = 15
            if n_pixels_at_LOS_resolution < n_pixels_warning:
                print("Warning: the GW sky area corresponds to less than {} pixels at the LOS resolution (nside:{}). You could use a higher resolution LOS if available.".format(n_pixels_warning,nside))
            
            pixelated_samples = make_pixel_px_function(samples, skymap, npixels=posterior_samples_dictionary[key][pu.PE_min_pixels], thresh=sky_area)
            nside_low_res = pixelated_samples.nside
            if nside_low_res > nside:
                raise ValueError(f'Low resolution nside {nside_low_res} is higher than high resolution nside {nside}. Try decreasing min_pixels for event {key}, or use a higher resolution LOS if available.')

            # identify which samples will be used to compute p(x|z,H0) for each pixel
            pixel_indices = pixelated_samples.indices
            samp_ind ={}
            for i,pixel_index in enumerate(pixel_indices):
                samp_ind[pixel_index] = pixelated_samples.identify_samples(pixel_index, minsamps=min_samps_in_pixel)

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

        Nobs = len(self.keys)
        if Nobs == 0:
            raise ValueError("No events to analyse.")

        # take care of the SNR/FAR selections
        self.snr_cut = network_snr_threshold
        self.ifar_cut = ifar_cut
        self.injections = injections
        # set the actual number of selected GW events entering the analysis, used for the check Neff >= 4Nobs inside the injection class
        self.injections.Nobs = Nobs
        self.injections.update_cut(self.snr_cut,self.ifar_cut)
        print(self.keys)
        print("Analysing {} GW events...".format(len(self.keys)))


    def MergersPerYearPerGpc3_z_UniformComoving(self,z,H0):

        """
        This function computes the differential number of mergers at a redshift z
        assuming a uniform in comoving volume distribution of galaxies.
        number density of mergers per redshift bin per time bin (detector frame):
        dN/(d t_det dz) = dN/(dVc dts) x dVc/dz / (1+z)
        dN/(dVc dts) is R0 x  Madau
        we fix here R0 = 1 (arbitrary normalization at z=0)
        in gwcosmo:
        1) p_z is (dVc/dz)/(4pi (c/H0)^3) so that dVc/dz = p_z x (4pi (c/H0)^3)
        2) zrates is Madau/(1+z)
        the result is in Gpc^{-3} yr^{-1}
        """
        return self.zrates(z)*self.cosmo.p_z(z)*4*np.pi*(light_speed_in_km_per_sec/H0)**3/1e9


    def NtotMergers_UniformComoving(self,H0,R0=1,Tobs=1):

        """
        Compute the true number of mergers occurring during time Tobs with a rate R0 at z=0, given H0,
        between z=cosmo.zmin (=1e-6 by default) and z = cosmo.zmax (=10 by default)
        the galaxies are assumed to be distributed uniformly in comoving volume
        default: return the number of mergers in the universe during 1 year with R0=1 (1 merger per Gpc3 per yr)
        """

        return R0*Tobs*quad(self.MergersPerYearPerGpc3_z_UniformComoving,
                            self.cosmo.zmin,
                            self.cosmo.zmax,args=(H0))[0]

    def NtotMergers_LOS(self,H0,R0=1,Tobs=1):
        
        """
        Compute the true number of mergers occurring during time Tobs with a rate R0 at z=0, given H0,
        between z=cosmo.zmin (=1e-6 by default) and z = cosmo.zmax (=10 by default)
        the galaxies are assumed to be distributed following the LOS used in the analysis
        default: return the number of mergers in the universe during 1 year with R0=1 (1 merger per Gpc3 per yr)
        """

        # the input LOS is not normalized. We normalize it by requiring that the high-z part
        # must coincide with a uniform in comoving distribution, for insance in the range z [3;cosmo.zmax (=10)]
        z_uniform_min = 3
        wz = np.where( (self.z_array>=z_uniform_min) & (self.z_array<=self.cosmo.zmax))[0]
        uniform_Vc = self.cosmo.p_z(self.z_array)*4*np.pi*(light_speed_in_km_per_sec/H0)**3/1e9 # per Gpc per year
        # compute the scaling factor
        scaling_LOS_to_uniform_comoving = simpson(uniform_Vc[wz],x=self.z_array[wz])/simpson(self.zprior_full_sky[wz],x=self.z_array[wz])
        nmergers_LOS = simpson(scaling_LOS_to_uniform_comoving*self.zprior_full_sky*self.zrates(self.z_array),x=self.z_array)
        nmergers_uniform = simpson(uniform_Vc*self.zrates(self.z_array),x=self.z_array)
        return  scaling_LOS_to_uniform_comoving, R0*Tobs*nmergers_LOS, R0*Tobs*nmergers_uniform
    
    def Get_Nmergers_Nexp_UniformComoving(self,H0):

        uniform_Vc = self.cosmo.p_z(self.z_array)*4*np.pi*(light_speed_in_km_per_sec/H0)**3/1e9 # per Gpc per year
        values = uniform_Vc*self.zrates(self.z_array)
        z_prior = interp1d(self.z_array,values,bounds_error=False,fill_value=(0,values[-1]))
        dz = np.diff(self.z_array)
        z_prior_norm = np.sum((values[:-1]+values[1:])*(dz)/2)
        # no need for deepcopy (mattermost bilby.help channel, 20240619), Colm Talbot wrote:
        # "Each thread has it's own copy of the likelihood object, so there's no need for copying."
        # injections = copy.deepcopy(self.injections)
        Nmergers = self.NtotMergers_UniformComoving(H0,R0=1,Tobs=1)
        cosmo = copy.deepcopy(self.cosmo)
        cosmo.H0 = H0
        self.injections.update_VT(cosmo,self.mass_priors,z_prior,z_prior_norm)
        Nexp = self.injections.VT_sens*Nmergers/z_prior_norm # for R0=1 and Tobs=1

        Neff, Neff_is_ok, var = self.injections.calculate_Neff()
        if not Neff_is_ok: # Neff >= 4*Nobs
            print("Not enough Neff ({}) compared to Nobs ({}) for current mass-model {}, z-model {}, zprior_norm {}"
                  .format(Neff,self.injections.Nobs,self.mass_priors,z_prior,z_prior_norm))
            print("mass prior dict: {}, cosmo_prior_dict: {}".format(self.mass_priors_param_dict,self.cosmo_param_dict))

        return Nexp, Nmergers

    def Get_Nmergers_Nexp_LOS(self,H0):

        """
        Computes the expected number of mergers using the actual LOS
        For this we need to properly normalize the LOS (which is provided with an unknown normalization)
        """
        scale_LOS_to_uniform, Nmergers_LOS, Nmergers_uniform = self.NtotMergers_LOS(H0)
        values = scale_LOS_to_uniform*self.zprior_full_sky*self.zrates(self.z_array) # normalized LOS
        z_prior = interp1d(self.z_array,values,bounds_error=False,fill_value=(0,values[-1]))
        dz = np.diff(self.z_array)
        z_prior_norm = np.sum((values[:-1]+values[1:])*(dz)/2)
        cosmo = copy.deepcopy(self.cosmo)
        cosmo.H0 = H0
        self.injections.update_VT(cosmo,self.mass_priors,z_prior,z_prior_norm)
        Nexp_LOS = self.injections.VT_sens*Nmergers_LOS/z_prior_norm # for R0=1 and Tobs=1

        Neff, Neff_is_ok, var = self.injections.calculate_Neff()
        if not Neff_is_ok: # Neff >= 4*Nobs
            print("Not enough Neff ({}) compared to Nobs ({}) for current mass-model {}, z-model {}, zprior_norm {}"
                  .format(Neff,self.injections.Nobs,self.mass_priors,z_prior,z_prior_norm))
            print("mass prior dict: {}, cosmo_prior_dict: {}".format(self.mass_priors_param_dict,self.cosmo_param_dict))

        return Nexp_LOS, Nmergers_LOS


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
            kde, norm, status = self.reweight_samps.marginalized_redshift_reweight(z_samps,m1_samps,m2_samps,PEprior)

            if norm != 0: # px_zOmegaH0 is initialized to 0
                zmin_temp = np.min(z_samps)*0.5
                zmax_temp = np.max(z_samps)*2.
                z_array_temp = np.linspace(zmin_temp,zmax_temp,100)

                # interp1d is deprecated see
                # https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#legacy-interface-for-1-d-interpolation-interp1d
                # The next line could be replaced by
                # px_zOmegaparam_interp = interpolate.CubicSpline(z_array_temp, kde(z_array_temp))
                px_zOmegaparam_interp = interp1d(z_array_temp,kde(z_array_temp),kind='cubic')
                mask = (zmin_temp < self.z_array) & (self.z_array < zmax_temp)
                px_zOmegaparam[i,mask] = px_zOmegaparam_interp(self.z_array[mask]) * norm

            else: # get information on the GW event and pixel that gave norm == 0
                if status == False: # norm is 0 and status = False ie KDE not reliable
                    print("KDE problem was for GW id {} and pixel {} with cosmo params: {}".format(event_name,pixel_index,self.cosmo_param_dict))

        # make p(s|z) have the same shape as p(x|z,Omega,param) and p(z|Omega,s)
        ps_z_array = np.tile(self.zrates(self.z_array),(len(pixel_indices),1))

        Inum_vals = np.sum(px_zOmegaparam*zprior*ps_z_array,axis=0)
        num = simpson(Inum_vals, x=self.z_array)

        return np.log(num)

    def log_likelihood_denominator_single_event(self):

        # uniform_Vc = self.cosmo.p_z(self.z_array)#*4*np.pi*(light_speed_in_km_per_sec/self.cosmo.H0)**3/1e9
        # values = uniform_Vc*self.zrates(self.z_array)
        values = self.zprior_full_sky*self.zrates(self.z_array)
        z_prior = interp1d(self.z_array,values,bounds_error=False,fill_value=(0,values[-1]))
        dz = np.diff(self.z_array)
        z_prior_norm = np.sum((values[:-1]+values[1:])*(dz)/2)
        # no need for deepcopy (mattermost bilby.help channel, 20240619), Colm Talbot wrote:
        # "Each thread has it's own copy of the likelihood object, so there's no need for copying."
        # injections = copy.deepcopy(self.injections)
        # Update the sensitivity estimation with the new model
        self.injections.update_VT(self.cosmo,self.mass_priors,z_prior,z_prior_norm)
        Neff, Neff_is_ok, var = self.injections.calculate_Neff()
        if Neff_is_ok: # Neff >= 4*Nobs
            log_den = np.log(self.injections.gw_only_selection_effect())
        else:
            print("Not enough Neff ({}) compared to Nobs ({}) for current mass-model {}, z-model {}, zprior_norm {}"
                  .format(Neff,self.injections.Nobs,self.mass_priors,z_prior,z_prior_norm))
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
            #Nexp, Nmergers = self.Get_Nmergers_Nexp(self.cosmo_param_dict['H0'])
            #print(self.cosmo_param_dict['H0'],num-den,num,den,Nexp,Nmergers,Nexp/Nmergers)

        return num-den

    def log_likelihood(self):

        # update cosmo parameters
        self.cosmo_param_dict = {par: self.parameters[par] for par in ["H0", "Xi0", "n", "D", "logRc", "nD", "cM"]}
        self.cosmo.update_parameters(self.cosmo_param_dict)
        
        # update redshift evo parameters
        self.zrates.gamma = self.parameters['gamma']
        self.zrates.k = self.parameters['Madau_k']
        self.zrates.zp = self.parameters['Madau_zp']
        
        # update mass prior parameters
        self.mass_priors_param_dict = {name: self.parameters[name] for name in self.mass_prior_params.keys()}
        self.mass_priors.update_parameters(self.mass_priors_param_dict)

        self.reweight_samps = reweight_posterior_samples(self.cosmo,self.mass_priors)

        return self.log_combined_event_likelihood()

    def __call__(self):
        return np.exp(self.log_likelihood())
