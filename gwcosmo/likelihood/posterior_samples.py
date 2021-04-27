"""
LALinference posterior samples class and methods
Ignacio Magana, Ankan Sur
"""
import numpy as np
from scipy.stats import gaussian_kde
from astropy import units as u
import h5py
from ..utilities.standard_cosmology import z_dlH0, fast_cosmology
from ..prior.priors import distance_distribution, mass_prior
import json

from scipy.interpolate import interp1d, interp2d


class posterior_samples(object):
    """
    Posterior samples class and methods.

    Parameters
    ----------
    cosmo : Fast cosmology class
    posterior_samples : Path to posterior samples file to be loaded.
    field : Internal field of the json or the h5 file
    """
    
    def __init__(self,cosmo, posterior_samples,field=None):
        self.posterior_samples = posterior_samples
        
        self.cosmo=cosmo
        self.field=field
        self.load_posterior_samples()

    def jacobian_times_prior(self,z,H0):
        
        jacobian = np.power(1+z,2)*self.cosmo.dL_by_z_H0(z,H0)
        dl = self.cosmo.dl_zH0(z, H0)
        return jacobian*(dl**2)

    def load_posterior_samples(self):
        """
        Method to handle different types of posterior samples file formats.
        Currently it supports .dat (LALinference), .hdf5 (GWTC-1),
        .h5 (PESummary) and .hdf (pycbcinference) formats.
        """
        if self.posterior_samples[-3:] == 'dat':
            samples = np.genfromtxt(self.posterior_samples, names = True)
           
            self.distance = np.array([var for var in samples['luminosity_distance']])
            self.ra =  np.array([var for var in samples['ra']])
            self.dec =  np.array([var for var in samples['dec']])
            self.mass_1 =  np.array([var for var in samples['mass_1']])
            self.mass_2 =  np.array([var for var in samples['mass_2']])
            self.nsamples = len(self.distance)

        if self.posterior_samples[-4:] == 'hdf5':
            if self.posterior_samples[-11:] == 'GWTC-1.hdf5':
                if self.posterior_samples[-20:] == 'GW170817_GWTC-1.hdf5':
                    dataset_name = 'IMRPhenomPv2NRT_lowSpin_posterior'
                else:
                    dataset_name = 'IMRPhenomPv2_posterior'
                file = h5py.File(self.posterior_samples, 'r')
                data = file[dataset_name]
                self.distance = data['luminosity_distance_Mpc']
                self.ra = data['right_ascension']
                self.dec = data['declination']
                self.mass_1 = data['m1_detector_frame_Msun']
                self.mass_2 = data['m2_detector_frame_Msun']
                self.nsamples = len(self.distance)
                file.close()

        if self.posterior_samples.endswith('.json'):
            with open(self.posterior_samples) as f:
                data = json.load(f)

            PE_struct=data['posterior_samples'][self.field]

            m1_ind=PE_struct['parameter_names'].index('mass_1')
            m2_ind=PE_struct['parameter_names'].index('mass_2')
            dl_ind=PE_struct['parameter_names'].index('luminosity_distance')
            ra_ind=PE_struct['parameter_names'].index('ra')
            dec_ind=PE_struct['parameter_names'].index('dec')
                        
            nsamp=len(PE_struct['samples'])
            
            self.distance = np.array(PE_struct['samples'])[:,dl_ind].reshape(-1)
            self.ra = np.array(PE_struct['samples'])[:,ra_ind].reshape(-1)
            self.dec = np.array(PE_struct['samples'])[:,dec_ind].reshape(-1)
            self.mass_1 = np.array(PE_struct['samples'])[:,m1_ind].reshape(-1)
            self.mass_2 = np.array(PE_struct['samples'])[:,m2_ind].reshape(-1)
            self.nsamples = len(self.distance)


        if self.posterior_samples[-2:] == 'h5':
            file = h5py.File(self.posterior_samples, 'r')

            if self.field is None:
                approximants = ['C01:PhenomPNRT-HS', 'C01:NRSur7dq4',
                                'C01:IMRPhenomPv3HM', 'C01:IMRPhenomPv2',
                                'C01:IMRPhenomD']
                for approximant in approximants:
                    try:
                        data = file[approximant]
                        print("Using "+approximant+" posterior")
                        break
                    except KeyError:
                        continue
            else:
                data=file[self.field]

            self.distance = data['posterior_samples']['luminosity_distance']
            self.ra = data['posterior_samples']['ra']
            self.dec = data['posterior_samples']['dec']
            self.mass_1 = data['posterior_samples']['mass_1']
            self.mass_2 = data['posterior_samples']['mass_2']
            self.nsamples = len(self.distance)
            file.close()

        if self.posterior_samples[-3:] == 'hdf':
            file = h5py.File(self.posterior_samples, 'r')
            self.distance = file['samples/distance'][:]
            self.ra = file['samples/ra'][:]
            self.dec = file['samples/dec'][:]
            self.mass_1 = file['samples/mass_1'][:]
            self.mass_2 = file['samples/mass_2'][:]
            self.nsamples = len(self.distance)
            file.close()

    def marginalized_sky(self):
        """
        Computes the marginalized sky localization posterior KDE.
        """
        return gaussian_kde(np.vstack((self.ra, self.dec)))

    def compute_source_frame_samples(self, H0):
        
        zmin = 0.0001
        zmax = 10
        zs = np.linspace(zmin, zmax, 10000)
        dLs = self.cosmo.dl_zH0(zs,H0)
        z_at_dL = interp1d(dLs,zs)
        redshift = z_at_dL(self.distance)
        mass_1_source = self.mass_1/(1+redshift)
        mass_2_source = self.mass_2/(1+redshift)
        return redshift, mass_1_source, mass_2_source


    def marginalized_redshift_reweight(self, H0, hyper_params_dict, name):
        """
        Computes the marginalized distance posterior KDE.
        """
        # Prior distribution used in this work
        new_prior = mass_prior(name=name, hyper_params_dict=hyper_params_dict)

        # Get source frame masses
        redshift, mass_1_source, mass_2_source = self.compute_source_frame_samples(H0)

        # Re-weight
        weights = new_prior.joint_prob(mass_1_source,mass_2_source)/self.jacobian_times_prior(redshift,H0)
        if len(np.argwhere(weights==0.))>0.999*len(weights):
            norm = 0.
            weights = np.ones(len(weights))
        else:
            norm = np.sum(weights)

        return gaussian_kde(redshift,weights=weights), norm

    def marginalized_redshift(self, H0):
        """
        Computes the marginalized distance posterior KDE.
        """
        
        # Get source frame masses
        redshift, mass_1_source, mass_2_source = self.compute_source_frame_samples(H0)

        # remove dl^2 prior and include dz/ddL jacobian
        weights = 1/(self.cosmo.dL_by_z_H0(redshift,H0)*self.cosmo.dl_zH0(redshift,H0)**2)
        norm = np.sum(weights)
        return gaussian_kde(redshift,weights=weights), norm



class make_px_function(object):
    """
    Make a line of sight, or sky-marginalised, function of the GW data in
    redshift and H0.

    Parameters
    ----------
    samples : posterior_samples object
        The GW samples
    H0 : array
    reweight_samples : bool, optional
        Should the samples be reweighted to the same source-frame mass prior
        as used in the selection effects? (Default=True)
    """
    def __init__(self, samples,H0,hyper_params_dict,name='BBH-powerlaw',reweight_samples=True):
        redshift_bins = 500
        vals = np.zeros((len(H0),redshift_bins))
        zmin = z_dlH0(np.amin(samples.distance),H0[0])*0.5
        zmax = z_dlH0(np.amax(samples.distance),H0[-1])*2.
        for i,H in enumerate(H0):
            if reweight_samples == True:
                zkernel, norm = samples.marginalized_redshift_reweight(H, hyper_params_dict,name=name)
            else:
                zkernel, norm = samples.marginalized_redshift(H)
            z_array = np.linspace(zmin, zmax, redshift_bins)
            vals[i,:] = zkernel(z_array)*norm
        self.temps = interp2d(z_array,H0,vals,bounds_error=False, fill_value=0)
        
    def px_zH0(self,z,H0):
        """
        Returns p(x|z,H0)
        BE CAREFUL ABOUT THE ORDERING OF z and H0! 
        Interp2d returns results corresponding to an ordered input array
        """
        return self.temps(z,H0)
        
    def __call__(self, z,H0):
        return self.px_zH0(z,H0)



