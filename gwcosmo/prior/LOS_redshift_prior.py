"""
Line-of-sight redshift prior creation module

Rachel Gray
Freija Beirnaert
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import norm, truncnorm
from scipy.interpolate import interp1d
from gwcosmo.utilities.zprior_utilities import create_zarray
from gwcosmo.utilities.luminosity_function import M_Mobs,M_mdl
from gwcosmo.prior.catalog import color_names, color_limits
from gwcosmo.likelihood.skymap import ra_dec_from_ipix, ipix_from_ra_dec
from gwcosmo.maps.create_norm_map import calc_norm
from optparse import make_option
import healpy as hp

import logging
logger = logging.getLogger(__name__)

class LineOfSightRedshiftPrior(object):
    """
    Calculate the likelihood of cosmological parameters from one GW event, 
    using the galaxy catalogue method.

    Parameters
    ----------
    pixel_index : Index of the healpy pixel to analyse
    galaxy_catalog : gwcosmo.prior.catalog.galaxyCatalog object
        The galaxy catalogue
    schechter_params : gwcosmo.utilities.schechter_params.SchechterParams class
        Class that stores the schechter function parameters alpha, Mstar, Mmin, Mmax
    cosmology : gwcosmo.utilities.cosmology.standard_cosmology object
        Cosmological model
    Kcorr : bool, optional
        Should K corrections be applied to the analysis? (default=False)
        Will raise an error if used in conjunction with a galaxy catalogue
        without sufficient color information.
    mth : string, optional
        Read the apparent magnitude threshold from a (lower resolution) precomputed mth map located at this path
        (default=None). If None, mth is estimated from the galaxy catalogue. If "inf", the pixel is empty
    galaxy_norm : string, optional
        Read the galaxy_normalization from a (lower resolution) precomuted galaxy norm map located at this path
        (default=None). If None, galaxy_norm is computed on the fly for this pixel and resolution.
    zcut : float, optional
        An artificial redshift cut to the galaxy catalogue (default=None)
    zmax : float, optional
        The upper redshift limit for integrals (default=10.). Should be well
        beyond the highest redshift reachable by GW data or selection effects.
    zuncert : bool, optional
        Should redshift uncertainties be marginalised over? (Default=True).

    """


    def __init__(self, pixel_index, galaxy_catalog, observation_band, nside, schechter_params, cosmology, zprior, luminosity_prior, luminosity_weights, Kcorr=False, mth=None, galaxy_norm=None, zcut=None, zmax=10., catalog_is_complete=False, min_gals_for_threshold=10):

        """
        Parameters
        ----------
        pixel_index : int
            The healpy index of the pixel being analysed
        galaxy_catalog : objecthttps://git.ligo.org/cbc-cosmo/gwcosmo
            The galaxy catalogue
        observation_band : str
            Observation band (eg. 'B', 'K', 'u', 'g')
        nside : int
            The resolution value of nside
        cosmology : object
            Standard cosmology
        zprior : object
            redshift prior, p(z)
        zrates : object
            rate evolution function, p(s|z)
        luminosity_prior : object
            absolute magnitude prior, p(M|H0)
        luminosity_weights : object
            luminosity weighting function, p(s|M)
        Kcorr : bool, optional
            Should K corrections be applied to the analysis? (default=False)
        mth : string, optional
            Path to the precomputed (lower resolution) apparent magnitude threshold map
        galaxy_norm : string, optional
            Path to the precomputed (lower resolution) galaxy_norm map
        zcut : float, optional
            An artificial redshift cut to the galaxy catalogue (default=None)
        zmax : float, optional
            The upper redshift limit for the universe (default=10.)
        zuncert : bool, optional
            Should redshift uncertainties be marginalised over? (Default=True)
        complete_catalog : bool, optional
            is the galaxy catalogue already complete? (Default=False)
        """
        
        self.pixel_index = pixel_index
        self.nside = nside
        self.zprior = zprior
        self.luminosity_prior = luminosity_prior
        self.luminosity_weights = luminosity_weights
        self.zcut = zcut
        self.sp = schechter_params
        self.Kcorr = Kcorr
        self.band = observation_band
        self.zmax = zmax
        self.cosmo = cosmology
        self.catalog_is_complete = catalog_is_complete
        self.galaxy_catalog = galaxy_catalog
        # load in the galaxy catalogue data from this pixel
        subcatalog = galaxy_catalog.select_pixel(nside, pixel_index, nested=True)
        
        # Set colour limits based on whether Kcorrections are applied
        if Kcorr == True:
            self.color_limit = color_limits[color_names[observation_band]]
        else:
            self.color_limit = [-np.inf,np.inf]

        if mth is None:
            # Calculate mth on the fly
            self.mth = subcatalog.magnitude_thresh(observation_band,min_gals=min_gals_for_threshold)
        elif mth == "inf":
            # For empty catalogue calculation
            self.mth = np.inf
        else:
            # Use precomputed mth map
            # Load precomputed map
            m = hp.fitsfunc.read_map(mth, nest=True)   
            # Get the coordinates of the pixel centre
            pixra, pixdec = ra_dec_from_ipix(nside, pixel_index, nest=True)  
            # Compute the corresponding low-res index
            ipix = ipix_from_ra_dec(hp.pixelfunc.get_nside(m), pixra, pixdec, nest=True)
            # Look up mth in precomputed map
            self.mth = m[ipix]
        logger.info(f"pixel {pixel_index}: mth = {self.mth}")     
        
        # select galaxies which are below the redshift cut and magnitude threshold, and have the relevant color info
        logger.info(f"pixel {pixel_index}: Selecting galaxies below redshift {self.zcut}, with color information between {self.color_limit[0]} and {self.color_limit[1]} in the {observation_band} band, and below apparent magnitude threshold {self.mth}")
        subcatalog = subcatalog.apply_redshift_cut(self.zcut).apply_color_limit(observation_band, self.color_limit[0], self.color_limit[1]).apply_magnitude_limit(observation_band, self.mth)

        self.zs = subcatalog['z'] 
        #ras = subcatalog['ra']
        #decs = subcatalog['dec']
        self.ms = subcatalog.get_magnitudes(observation_band)
        self.sigmazs = subcatalog['sigmaz']
        self.colors = subcatalog.get_color(observation_band)

        self.galaxy_norm = 0.
        if not np.isinf(self.mth):
            if galaxy_norm is None:
                logger.warn("galaxy_norm is None, which should not be used at high resolutions")
                # calculate on the fly 
                galaxy_norm = calc_norm(self.galaxy_catalog, self.band, nside, pixel_index, self.zmax, self.zcut, self.mth, self.cosmo, self.sp.Mmax, self.Kcorr)
                self.galaxy_norm = galaxy_norm
            else:
                # Use precomputed galaxy norm map
                # Load precomputed map
                m = hp.fitsfunc.read_map(galaxy_norm, nest=True)   
                # Get the coordinates of the pixel centre
                pixra, pixdec = ra_dec_from_ipix(nside, pixel_index, nest=True)  
                # Compute the corresponding low-res index
                ipix = ipix_from_ra_dec(hp.pixelfunc.get_nside(m), pixra, pixdec, nest=True)
                # Look up mth in precomputed map
                self.galaxy_norm = m[ipix]
                # Adjust for different resolutions    
                self.galaxy_norm /= (hp.nside2npix(self.nside) / hp.get_map_size(m))
        logger.info(f"galaxy norm: {self.galaxy_norm}")

    def get_ngals(self):
        return len(self.zs)
        

    def uninformative_host_galaxy_prior(self,M,z,H0):
        """
        evaluates p(z)*p(M|H0)*p(s|M)
        
        This is the prior distribution of all GW host galaxies in the universe,
        neglecting rate evolution of mergers with redshift

        Parameters
        ----------
        M : float
            absolute magnitude
        z : float
            redshift
        H0 : float
            Hubble constant value in kms-1Mpc-1

        Returns
        -------
        float
        """
        
        return self.zprior(z)*self.luminosity_prior(M,H0)*self.luminosity_weights(M)


    def uninformative_galaxy_prior(self,M,z,H0):
        """
        evaluates p(z)*p(M|H0)
        
        This is the prior distribution of all galaxies in the universe

        Parameters
        ----------
        M : float
            absolute magnitude
        z : float
            redshift
        H0 : float
            Hubble constant value in kms-1Mpc-1

        Returns
        -------
        float
        """
        return self.zprior(z)*self.luminosity_prior(M,H0)   


    def create_redshift_prior(self):
        logger.info(f"nside {self.nside} pixel {self.pixel_index}: Start creation of redshift prior")
        z_array = create_zarray(self.galaxy_catalog, self.band)
        z_array2 = np.logspace(-6,np.log10(self.zmax),600) # low-res redshift array for smooth functions

        dz = np.ediff1d(z_array,to_end=1)
        
        H0 = self.cosmo.H0
        Mmin = M_Mobs(H0,self.sp.Mmin)
        Mmax = M_Mobs(H0,self.sp.Mmax)

        # if the pixel has no galaxy catalogue support, compute redshift
        if np.isinf(self.mth):
            pz_G = 0
            pG = 0
            pz_Gbar = np.zeros(len(z_array2))
            for i in range(len(z_array2)):
                pz_Gbar[i] = quad(self.uninformative_host_galaxy_prior,Mmin,Mmax,args=(z_array2[i],H0), epsabs=0, epsrel=1.49e-6)[0]
        else:    
            pz_G = np.zeros(len(z_array))
            for i in range(len(self.zs)):
                z_array3 = np.linspace(self.zs[i]-5*self.sigmazs[i],self.zs[i]+5*self.sigmazs[i],50) # redshift array for truncnorm interpolations in redshiftuncertainties
                trunk = np.empty_like(z_array3)
                # set redshift limits so that galaxies can't have negative z
                low_z_lim, high_z_lim = (0 - self.zs[i])/self.sigmazs[i], (self.zmax - self.zs[i])/self.sigmazs[i]
                if self.Kcorr:
                    Kcorr = self.galaxy_catalog.get_k_correction(self.band, z_array, color_names[self.band], self.colors[i])
                else:
                    Kcorr = 0.
                trunk = truncnorm.pdf(z_array3, low_z_lim, high_z_lim, self.zs[i], self.sigmazs[i])
                interpolate_trunk = interp1d(z_array3, trunk, bounds_error=False, fill_value=0) 
                kill_weights = M_mdl(self.ms[i],self.cosmo.dl_zH0(z_array),Kcorr=Kcorr) < Mmax
                pz_G += interpolate_trunk(z_array) * self.luminosity_weights(M_mdl(self.ms[i],self.cosmo.dl_zH0(z_array),Kcorr=Kcorr)) * kill_weights
            if self.galaxy_norm != 0:
                pz_G = pz_G/self.galaxy_norm
                pz_G[np.where(z_array>self.zcut)[0]]=0
            else:
                pz_G = 0
            
            pG = dblquad(self.uninformative_galaxy_prior,0,self.zcut,lambda x: Mmin, lambda x: max(min(M_mdl(self.mth,self.cosmo.dl_zH0(x)),Mmax),Mmin), args=[H0], epsabs=0, epsrel=1.49e-6)[0]

            # compute p(z|Gbar,H0)
            pz_Gbar = np.zeros(len(z_array2))
            for i in range(len(z_array2)):
                if z_array2[i] < self.zcut:
                    pz_Gbar[i] = quad(self.uninformative_host_galaxy_prior, min(max(M_mdl(self.mth,self.cosmo.dl_zH0(z_array2[i])),Mmin),Mmax), Mmax, args=(z_array2[i],H0), epsabs=0, epsrel=1.49e-6)[0]
                else:
                    pz_Gbar[i] = quad(self.uninformative_host_galaxy_prior, Mmin, Mmax, args=(z_array2[i],H0), epsabs=0, epsrel=1.49e-6)[0]
            
        pz_Gbar_smooth = interp1d(z_array2,pz_Gbar,kind='linear')

        pz_total = (pz_G*pG + pz_Gbar_smooth(z_array))

        return pz_total, z_array
