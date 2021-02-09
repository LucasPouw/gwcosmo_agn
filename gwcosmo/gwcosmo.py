"""
gwcosmoLikelihood Module
Rachel Gray, Archisman Ghosh, Ignacio Magana, John Veitch, Ankan Sur

"""
from __future__ import absolute_import
import lal
import numpy as np
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import healpy as hp
import warnings
warnings.filterwarnings("ignore")

from scipy.integrate import quad, dblquad
from scipy.stats import ncx2, norm, truncnorm
from scipy.interpolate import splev, splrep, interp1d
from astropy import constants as const
from astropy import units as u
from ligo.skymap.moc import rasterize
from ligo.skymap.core import uniq2ang

import astropy.constants as constants



import gwcosmo

from .utilities.standard_cosmology import *
from .utilities.schechter_function import *
from .utilities.schechter_params import *
from .utilities.calc_kcor import *
from .prior.catalog import color_names
from .likelihood.skymap import ra_dec_from_ipix,ipix_from_ra_dec

import time
import progressbar


################################################################################
################################# THE MAIN CLASSES #############################
################################################################################



class gwcosmoLikelihood(object):
    """
    """
    def __init__(self, px_zH0, pD_zH0, zprior, zrates, zmax=10.):
        self.px_zH0 = px_zH0
        self.pD_zH0 = pD_zH0
        self.zprior = zprior
        self.zrates = zrates
        self.zmax = zmax
        
    def px_zH0_times_pz_times_ps_z(self, z, H0, Lambda=0.):
        return self.px_zH0(z,H0)*self.zprior(z)*self.zrates(z,Lambda=Lambda)
        
    def pD_zH0_times_pz_times_ps_z(self, z, H0, Lambda=0.):
        return self.pD_zH0(z,H0)*self.zprior(z)*self.zrates(z,Lambda=Lambda)

    def px_OH0(self, H0, skyprob=1., Lambda=0.):
        """
        Evaluate p(x|O,H0).
        
        Defined as a single integral over z (instead of over z and M) as it is 
        equivalent to the empty catalogue case.
        NOTE: this is only possible as the ratio px_OH0/pD_OH0 is taken later.
        
        Parameters
        ----------
        H0 : float
            Hubble constant value in kms-1Mpc-1
        Lambda : float, optional
            Redshift evolution parameter (default=0)
        zmax : float, optional
            The upper redshift limit for integrals (default=10.)

        Returns
        -------
        float
            p(x|O,H0)
        """
        
        integral = quad(self.px_zH0_times_pz_times_ps_z,0.,self.zmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        return integral * skyprob
        
    def pD_OH0(self, H0, skyprob=1., Lambda=0.):
        """
        Evaluate p(x|O,H0).
        
        Defined as a single integral over z (instead of over z and M) as it is 
        equivalent to the empty catalogue case.
        NOTE: this is only possible as the ratio px_OH0/pD_OH0 is taken later.
        
        Parameters
        ----------
        H0 : float
            Hubble constant value in kms-1Mpc-1
        Lambda : float, optional
            Redshift evolution parameter (default=0)
        zmax : float, optional
            The upper redshift limit for integrals (default=10.)

        Returns
        -------
        float
            p(x|O,H0)
        """
        
        integral = quad(self.pD_zH0_times_pz_times_ps_z,0.,self.zmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        return integral * skyprob


class GalaxyCatalogLikelihood(gwcosmoLikelihood):
    """
    Calculate the likelihood of H0 from one GW event, using the galaxy 
    catalogue method.
    
    Parameters
    ----------
    base_functions : gwcosmo.gwcosmo.BaseFunctions object
        p(x|z,H0)*p(z)*p(s|z)*p(M)*p(s|M) and p(D|z,H0)*p(z)*p(s|z)*p(M)*p(s|M)
    skymap : gwcosmo.likelihood.skymap.skymap object
        provides p(x|Omega) and skymap properties
    galaxy_catalog : gwcosmo.prior.catalog.galaxyCatalog object
        The galaxy catalogue
    fast_cosmology : gwcosmo.utilities.standard_cosmology.fast_cosmology object
        Cosmological model
    Kcorr : bool, optional
        Should K corrections be applied to the analysis? (default=False)
        Will raise an error if used in conjunction with a galaxy catalogue 
        without sufficient color information.
    mth : float, optional
        Specify an apparent magnitude threshold for the galaxy catalogue
        (default=None). If none, mth is estimated from the galaxy catalogue.
    zcut : float, optional
        An artificial redshift cut to the galaxy catalogue (default=None)
    zmax : float, optional
        The upper redshift limit for integrals (default=10.). Should be well 
        beyond the highest redshift reachable by GW data or selection effects.
    zuncert : bool, optional
        Should redshift uncertainties be marginalised over? (Default=True).
    
    """
    def __init__(self, skymap, observation_band, fast_cosmology, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights, Kcorr=False, zmax=10.):
        super().__init__(px_zH0, pD_zH0, zprior, zrates, zmax=zmax)
        
        self.skymap = skymap
        self.cosmo = fast_cosmology
        self.luminosity_prior = luminosity_prior
        self.luminosity_weights = luminosity_weights
        
        self.Kcorr = Kcorr
        self.band = observation_band
        sp = SchechterParams(self.band)
        self.Mmin_obs = sp.Mmin
        self.Mmax_obs = sp.Mmax

        
    def px_zH0_times_pz_times_ps_z_times_pM_times_ps_M(self, M, z, H0, Lambda=0.):
        return self.px_zH0(z,H0)*self.zprior(z)*self.zrates(z,Lambda=Lambda) \
                *self.luminosity_prior(M,H0)*self.luminosity_weights(M)
        
    def pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M(self, M, z, H0, Lambda=0.):
        return self.pD_zH0(z,H0)*self.zprior(z)*self.zrates(z,Lambda=Lambda) \
                *self.luminosity_prior(M,H0)*self.luminosity_weights(M)
        
    def pxD_GH0(self, H0, sampz, sampm, sampra, sampdec, sampcolor, count, Lambda=0.):
        """
        Evaluate p(x|G,H0) and p(D|G,H0).

        Parameters
        ----------
        H0 : array of floats
            Hubble constant value(s) in kms-1Mpc-1
        sampz, sampm, sampra, sampdec, sampcolor : arrays of floats
            redshift, apparent magnitude, right ascension, declination and 
            colour samples
        count : the number of samples which belong to 1 galaxy
        Lambda : float, optional
            redshift evolution parameter (default=0)

        Returns
        -------
        arrays
            numerator and denominator
        """      
            
        if self.Kcorr == True:
            Kcorr = calc_kcor(self.band,sampz,color_names[self.band],colour_value=sampcolor)
        else:
            Kcorr = 0.
            
        tempsky = self.skymap.skyprob(sampra, sampdec)*self.skymap.npix
        
        zweights = self.zrates(sampz,Lambda=Lambda)
        
        tempnum = np.zeros([len(H0)])
        tempden = np.zeros([len(H0)])
        for k,h in enumerate(H0):
            numinner = self.px_zH0(sampz,h)
            deninner = self.pD_zH0(sampz,h)
            sampAbsM = M_mdl(sampm, self.cosmo.dl_zH0(sampz, h), Kcorr=Kcorr)
            
            # for samples which are fainter than the faint end of the Schechter function
            # set contribution to zero.
            Mmax = M_Mobs(h,self.Mmax_obs)
            sel = np.where(sampAbsM > Mmax)[0] # identify samples fainter than model allows
            Lweights = self.luminosity_weights(sampAbsM)
            if self.luminosity_weights.luminosity_weights == False:
                Lweights = np.ones(len(sampAbsM))*Lweights
            Lweights[sel] = 0 # set faint sample contribution to zero
            
            normsamp = 1./count

            tempnum[k] = np.sum(numinner*tempsky*Lweights*zweights*normsamp)
            tempden[k] = np.sum(deninner*Lweights*zweights*normsamp)

        return tempnum,tempden
        
    def pGB_DH0(self, H0, mth, skyprob, Lambda=0., zcut=10.):
        """
        Evaluate p(G|D,H0) and p(B|D,H0).
        
        The probability that the host galaxy of a detected GW event is inside
        or beyond the galaxy catalogue.
        
        Parameters
        ----------
        H0 : float
            Hubble constant value in kms-1Mpc-1
        mth : float
            Apparent magnitude threshold
        Lambda : float, optional
            Redshift evolution parameter (default=0)
        zcut : float, optional
            An artificial redshift cut to the galaxy catalogue (default=10.)
        zmax : float, optional
            The upper redshift limit for integrals (default=10.)

        Returns
        -------
        floats
            p(G|D,H0), p(B|D,H0), num, den
            where num/den = p(G|D,H0)
        """
        
        Mmin = M_Mobs(H0,self.Mmin_obs)
        Mmax = M_Mobs(H0,self.Mmax_obs)

        num = dblquad(self.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,zcut,lambda x: Mmin,lambda x: min(max(M_mdl(mth,self.cosmo.dl_zH0(x,H0)),Mmin),Mmax),args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]

        den = dblquad(self.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,self.zmax,lambda x: Mmin,lambda x: Mmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]

        integral = num/den
        
        pG = integral*skyprob
        pB = (1.-integral)*skyprob
        return pG, pB, num, den
        
    def px_BH0(self, H0, mth, skyprob, Lambda=0., zcut=10.):
        """
        Evaluate p(x|B,H0).
        
        If zcut >= zmax then a single integral is performed.
        If zcut < zmax then an additional integral is performed.
        
        Parameters
        ----------
        H0 : float
            Hubble constant value in kms-1Mpc-1
        mth : float
            Apparent magnitude threshold
        Lambda : float, optional
            Redshift evolution parameter (default=0)
        zcut : float, optional
            An artificial redshift cut to the galaxy catalogue (default=10.)
        zmax : float, optional
            The upper redshift limit for integrals (default=10.)

        Returns
        -------
        float
            p(x|B,H0)
        """
        
        Mmin = M_Mobs(H0,self.Mmin_obs)
        Mmax = M_Mobs(H0,self.Mmax_obs)
            
        below_zcut_integral = dblquad(self.px_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,zcut,lambda x: min(max(M_mdl(mth,self.cosmo.dl_zH0(x,H0)),Mmin),Mmax), lambda x: Mmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        
        above_zcut_integral = 0.
        if zcut < self.zmax:
            above_zcut_integral = dblquad(self.px_zH0_times_pz_times_ps_z_times_pM_times_ps_M,zcut,self.zmax,lambda x: Mmin, lambda x: Mmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        
        integral = below_zcut_integral + above_zcut_integral

        return integral * skyprob

    def pD_BH0(self, H0, mth, skyprob, Lambda=0., zcut=10.):
        """
        Evaluate p(D|B,H0).
        
        If zcut >= zmax then a single integral is performed.
        If zcut < zmax then an additional integral is performed.
        
        Parameters
        ----------
        H0 : float
            Hubble constant value in kms-1Mpc-1
        mth : float
            Apparent magnitude threshold
        Lambda : float, optional
            Redshift evolution parameter (default=0)
        zcut : float, optional
            An artificial redshift cut to the galaxy catalogue (default=10.)
        zmax : float, optional
            The upper redshift limit for integrals (default=10.)

        Returns
        -------
        float
            p(x|B,H0)
        """
        
        Mmin = M_Mobs(H0,self.Mmin_obs)
        Mmax = M_Mobs(H0,self.Mmax_obs)
            
        below_zcut_integral = dblquad(self.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,zcut,lambda x: min(max(M_mdl(mth,self.cosmo.dl_zH0(x,H0)),Mmin),Mmax), lambda x: Mmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        
        above_zcut_integral = 0.
        if zcut < self.zmax:
            above_zcut_integral = dblquad(self.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,zcut,self.zmax,lambda x: Mmin, lambda x: Mmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        
        integral = below_zcut_integral + above_zcut_integral

        return integral * skyprob


class SinglePixelGalaxyCatalogLikelihood(GalaxyCatalogLikelihood):
    """
    Calculate the likelihood of H0 from one GW event, using the galaxy 
    catalogue method.
    
    Parameters
    ----------
    base_functions : gwcosmo.gwcosmo.BaseFunctions object
        p(x|z,H0)*p(z)*p(s|z)*p(M)*p(s|M) and p(D|z,H0)*p(z)*p(s|z)*p(M)*p(s|M)
    skymap : gwcosmo.likelihood.skymap.skymap object
        provides p(x|Omega) and skymap properties
    galaxy_catalog : gwcosmo.prior.catalog.galaxyCatalog object
        The galaxy catalogue
    fast_cosmology : gwcosmo.utilities.standard_cosmology.fast_cosmology object
        Cosmological model
    Kcorr : bool, optional
        Should K corrections be applied to the analysis? (default=False)
        Will raise an error if used in conjunction with a galaxy catalogue 
        without sufficient color information.
    mth : float, optional
        Specify an apparent magnitude threshold for the galaxy catalogue
        (default=None). If none, mth is estimated from the galaxy catalogue.
    zcut : float, optional
        An artificial redshift cut to the galaxy catalogue (default=None)
    zmax : float, optional
        The upper redshift limit for integrals (default=10.). Should be well 
        beyond the highest redshift reachable by GW data or selection effects.
    zuncert : bool, optional
        Should redshift uncertainties be marginalised over? (Default=True).
    
    """
    def __init__(self, pixel_index, galaxy_catalog, skymap, observation_band, fast_cosmology, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights, Kcorr=False, mth=None, zcut=None, zmax=10.,zuncert=True, complete_catalog=False, nside=32):
        super().__init__(skymap, observation_band, fast_cosmology, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights, Kcorr=Kcorr, zmax=zmax)
        
        self.zcut = zcut
        self.complete_catalog = complete_catalog
        
        # Set redshift and colour limits based on whether Kcorrections are applied
        if Kcorr == True:
            if zcut is None:
                self.zcut = 0.5
            self.color_limit = galaxy_catalog.color_limit
        else:
            if zcut is None:
                self.zcut = self.zmax
            self.color_limit = [-np.inf,np.inf]
        
        #TODO make this changeable from command line?
        self.nfine = 10000
        self.ncoarse = 10

        self.hi_res_nside = nside
        pixra, pixdec = ra_dec_from_ipix(self.hi_res_nside, np.arange(hp.pixelfunc.nside2npix(self.hi_res_nside)), nest=skymap.nested)
        ipix = ipix_from_ra_dec(galaxy_catalog.nside, pixra, pixdec, nest=skymap.nested)
        self.sub_pixel_indices = np.arange(hp.pixelfunc.nside2npix(self.hi_res_nside))[np.where(ipix == pixel_index)[0]]
        print('Dividing this pixel into {} equal pieces'.format(len(self.sub_pixel_indices)))
        
        # pick out galaxies in the big pixel and divide into smaller pixels
        if pixel_index in galaxy_catalog.gal_indices_per_pixel_idx:
            ind = galaxy_catalog.gal_indices_per_pixel_idx[pixel_index]
            self.galz = galaxy_catalog.z[ind]
            self.galra = galaxy_catalog.ra[ind]
            self.galdec = galaxy_catalog.dec[ind]
            self.galm = galaxy_catalog.m[ind]
            self.galsigmaz = galaxy_catalog.sigmaz[ind]
            self.galcolor = galaxy_catalog.color[ind]

            self.sub_galaxy_indices = skymap.pixel_split(self.galra, self.galdec, self.hi_res_nside)

            self.mth_map = np.ones(len(self.sub_pixel_indices))*(np.inf)
            for i, index in enumerate(self.sub_pixel_indices):
                if index in self.sub_galaxy_indices.keys():
                    m = self.galm[self.sub_galaxy_indices[index]]               
                    temp_mth = np.median(m)
                    # if less than 10 galaxies, consider pixel empty
                    if len(m) < 10: 
                        temp_mth = np.inf
                    self.mth_map[i] = temp_mth
            if mth != None:
                self.mth_map = np.ones(len(self.sub_pixel_indices))*mth
        else:
            print('There are no galaxies in this pixel')
            self.mth_map = np.ones(len(self.sub_pixel_indices))*mth

        if zuncert == False:
            self.nfine = 1
            self.ncoarse = 1
            self.galsigmaz = np.zeros(len(self.galz))
        
        self.pxG = 0.
        self.pDG = 1.
        self.pG = 1.
        self.pxB = 0.
        self.pDB = 1.
        self.pB = 0.
        self.pxO = 0.
        self.pDO = 1.
        self.pO = 0.
        
        self.pixel_area_low_res = 1./hp.nside2npix(galaxy_catalog.nside)
        self.pixel_area_hi_res = 1./hp.nside2npix(self.hi_res_nside)
        
        hi_res_skyprob = hp.pixelfunc.ud_grade(skymap.prob, self.hi_res_nside, order_in='NESTED', order_out='NESTED')
        self.hi_res_skyprob = hi_res_skyprob/np.sum(hi_res_skyprob) #renormalise


    def pxD_GH0_multi(self,H0, z, sigmaz, m, ra, dec, color, Lambda=0.):
        """
        Evaluate p(x|G,H0) and p(D|G,H0).

        Parameters
        ----------
        H0 : array of floats
            Hubble constant value(s) in kms-1Mpc-1
        Lambda : float, optional
            redshift evolution parameter (default=0)

        Returns
        -------
        arrays
            p(x|G,H0), p(D|G,H0)
        """
        
        nGal = len(z)
        galindex_sep = {}
        if self.luminosity_weights.luminosity_weights == True:
            # TODO: find better selection criteria for sampling
            mlim = np.percentile(np.sort(m),0.01) # more draws for galaxies in brightest 0.01 percent
            samp_res = {'fine': self.nfine, 'coarse': self.ncoarse}
            galindex = {'fine': np.where(m <= mlim)[0], 'coarse': np.where(mlim < m)[0]}
            
            # for arrays with more than 1million entries, break into sub arrays
            no_chunks_coarse = int(np.ceil(len(galindex['coarse'])/1000000))
            chunks_coarse = np.array_split(galindex['coarse'],no_chunks_coarse)
            galindex_sep['coarse'] = {i+1 : chunks_coarse[i] for i in range(no_chunks_coarse)} 
            galindex_sep['fine'] = {i : galindex['fine'] for i in range(1)} 
        else:
            samp_res = {'coarse': self.ncoarse}
            galindex = {'coarse': np.arange(len(z))}
            
            # for arrays with more than 1million entries, break into sub arrays
            no_chunks_coarse = int(np.ceil(len(galindex['coarse'])/1000000))
            chunks_coarse = np.array_split(galindex['coarse'],no_chunks_coarse)
            galindex_sep['coarse'] = {i : chunks_coarse[i] for i in range(no_chunks_coarse)} 
        
        K = sum(len(v) for v in galindex.values()) # total number of sub arrays
        tempnum = np.zeros([K,len(H0)])
        tempden = np.zeros([K,len(H0)])
        
        # loop over sub arrays of galaxies
        for i,key in enumerate(samp_res):
            print('{} galaxies are getting sampled {}ly'.format(len(galindex[key]),key))
            for n, key2 in enumerate(galindex_sep[key]):
                zs = z[galindex_sep[key][key2]]
                sigmazs = sigmaz[galindex_sep[key][key2]]
                ms = m[galindex_sep[key][key2]]
                ras = ra[galindex_sep[key][key2]]
                decs = dec[galindex_sep[key][key2]]
                colors = color[galindex_sep[key][key2]]
                
                sampz, sampm, sampra, sampdec, sampcolor, count = gal_nsmear(zs, sigmazs, ms, ras, decs, colors, samp_res[key], zcut=self.zcut)
                    
                tempnum[key2,:],tempden[key2,:] = self.pxD_GH0(H0, sampz, sampm, sampra, sampdec, sampcolor, count, Lambda=Lambda)
                
        num = np.sum(tempnum,axis=0)/nGal
        den = np.sum(tempden,axis=0)/nGal

        return num,den  
        
    def full_pixel(self, H0, z, sigmaz, m, ra, dec, color, mth, px_Omega=1., pOmega=1., Lambda=0.):
        """
        Compute the full likelihood.
        
        Parameters
        ----------
        H0 : array of floats
            Hubble constant values in kms-1Mpc-1
        Lambda : float, optional
            Redshift evolution parameter (default=0)
        complete_catalog : bool, optional
            Assume that the galaxy catalogue is complete? (default=False)

        Returns
        -------
        float
            Returns likelihood, pxG, pDG, pG, pxB, pDB, pB, pxO, pDO, pO
            where likelihood = (pxG / pDG) * pG + (pxB / pDB) * pB + (pxO / pDO) * pO
        """
        pG = np.ones(len(H0))
        pxB = np.zeros(len(H0))
        pDB = np.ones(len(H0))
        pB = np.zeros(len(H0))
        
        num = np.zeros(len(H0))
        den = np.zeros(len(H0))
        
        print('Computing the in-catalogue part')
        pxG, pDG = self.pxD_GH0_multi(H0, z, sigmaz, m, ra, dec, color, Lambda=Lambda)

        if not self.complete_catalog:
            print('Computing the beyond catalogue part')   
            for i,h in enumerate(H0):
                pG[i], pB[i], num[i], den[i] = self.pGB_DH0(h, mth, pOmega, Lambda=Lambda, zcut=self.zcut)
                pxB[i] = self.px_BH0(h, mth, px_Omega, Lambda=Lambda, zcut=self.zcut)
            if self.zcut == self.zmax:
                pDB = (den - num) * pOmega
            else:
                print('Computing all integrals explicitly as zcut < zmax: this will take a little longer')
                for i,h in enumerate(H0):
                    pDB[i] = self.pD_BH0(h, mth, pOmega, Lambda=Lambda, zcut=self.zcut)

        likelihood = (pxG / pDG) * pG + (pxB / pDB) * pB
        return pxG, pDG, pG, pxB, pDB, pB
        
    def likelihood(self,H0,Lambda=0.):
        
        self.pxG = np.zeros([len(H0),len(self.sub_pixel_indices)])
        self.pDG = np.zeros([len(H0),len(self.sub_pixel_indices)])
        self.pG = np.zeros([len(H0),len(self.sub_pixel_indices)])
        
        self.pxB = np.zeros([len(H0),len(self.sub_pixel_indices)])
        self.pDB = np.zeros([len(H0),len(self.sub_pixel_indices)])
        self.pB = np.zeros([len(H0),len(self.sub_pixel_indices)])
        
        self.pxO = np.zeros([len(H0),len(self.sub_pixel_indices)])
        self.pDO = np.zeros([len(H0),len(self.sub_pixel_indices)])
        self.pO = np.zeros(len(self.sub_pixel_indices))
        
        if np.inf in self.mth_map:
            temp_pxO = np.zeros(len(H0))
            temp_pDO = np.zeros(len(H0))
            print('Not all of this pixel has catalogue support. Computing the out of catalogue contribution')
            for i,h in enumerate(H0):
                temp_pxO[i] = self.px_OH0(h, skyprob=1., Lambda=Lambda)
                temp_pDO[i] = self.pD_OH0(h, skyprob=1., Lambda=Lambda)
        
        # loop over sub-pixels
        for i, idx in enumerate(self.sub_pixel_indices):
            px_Omega = self.hi_res_skyprob[idx]
            if self.mth_map[i] == np.inf:
                self.pxO[:,i] = temp_pxO*px_Omega
                self.pDO[:,i] = temp_pDO*self.pixel_area_hi_res
                self.pO[i] = self.pixel_area_hi_res
                
                self.pxG[:,i] = np.zeros(len(H0))
                self.pDG[:,i] = np.ones(len(H0))
                self.pG[:,i] = np.zeros(len(H0))
                self.pxB[:,i] = np.zeros(len(H0))
                self.pDB[:,i] = np.ones(len(H0))
                self.pB[:,i] = np.zeros(len(H0))
            else:
                print('mth in this sub-pixel: {}'.format(self.mth_map[i]))
                z = self.galz[self.sub_galaxy_indices[idx]]
                ra = self.galra[self.sub_galaxy_indices[idx]]
                dec = self.galdec[self.sub_galaxy_indices[idx]]
                m = self.galm[self.sub_galaxy_indices[idx]]
                sigmaz = self.galsigmaz[self.sub_galaxy_indices[idx]]
                color = self.galcolor[self.sub_galaxy_indices[idx]]
                
                #find galaxies below redshift cut, and with right colour information
                ind = np.where(((z-3*sigmaz) <= self.zcut) & (m <= self.mth_map[i]) & \
                              (self.color_limit[0] <= color) & (color <= self.color_limit[1]))[0]   
                z = z[ind]
                ra = ra[ind]
                dec = dec[ind]
                m = m[ind]
                sigmaz = sigmaz[ind]
                color = color[ind]
                
                self.pxG[:,i], self.pDG[:,i], self.pG[:,i], self.pxB[:,i], self.pDB[:,i], self.pB[:,i] = self.full_pixel(H0, z, sigmaz, m, ra, dec, color, self.mth_map[i], px_Omega=px_Omega, pOmega=self.pixel_area_hi_res, Lambda=Lambda)
                self.pxO[:,i] = np.zeros(len(H0))
                self.pDO[:,i] = np.ones(len(H0))
                self.pO[i] = 0.

        sub_likelihood = np.zeros([len(H0),len(self.sub_pixel_indices)])
        for i in range(len(self.sub_pixel_indices)):
            sub_likelihood[:,i] = (self.pxG[:,i] / self.pDG[:,i]) * self.pG[:,i] + (self.pxB[:,i] / self.pDB[:,i]) * self.pB[:,i] + (self.pxO[:,i] / self.pDO[:,i]) * self.pO[i]
        likelihood = np.sum(sub_likelihood,axis=1)
        return likelihood
        
    def return_components(self):
        return self.pxG, self.pDG, self.pG, self.pxB, self.pDB, self.pB, self.pxO, self.pDO, self.pO
        
    def __call__(self, H0, Lambda=0.):
        return self.likelihood(H0, Lambda=Lambda)


class SimplePixelatedGalaxyCatalogLikelihood(GalaxyCatalogLikelihood):
    """
    Calculate the likelihood of H0 from one GW event, using the galaxy 
    catalogue method.
    
    Parameters
    ----------
    base_functions : gwcosmo.gwcosmo.BaseFunctions object
        p(x|z,H0)*p(z)*p(s|z)*p(M)*p(s|M) and p(D|z,H0)*p(z)*p(s|z)*p(M)*p(s|M)
    skymap : gwcosmo.likelihood.skymap.skymap object
        provides p(x|Omega) and skymap properties
    galaxy_catalog : gwcosmo.prior.catalog.galaxyCatalog object
        The galaxy catalogue
    fast_cosmology : gwcosmo.utilities.standard_cosmology.fast_cosmology object
        Cosmological model
    Kcorr : bool, optional
        Should K corrections be applied to the analysis? (default=False)
        Will raise an error if used in conjunction with a galaxy catalogue 
        without sufficient color information.
    mth : float, optional
        Specify an apparent magnitude threshold for the galaxy catalogue
        (default=None). If none, mth is estimated from the galaxy catalogue.
    zcut : float, optional
        An artificial redshift cut to the galaxy catalogue (default=None)
    zmax : float, optional
        The upper redshift limit for integrals (default=10.). Should be well 
        beyond the highest redshift reachable by GW data or selection effects.
    zuncert : bool, optional
        Should redshift uncertainties be marginalised over? (Default=True).
    
    """
    def __init__(self, galaxy_catalog, skymap, observation_band, fast_cosmology, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights, Kcorr=False, mth=None, zcut=None, zmax=10.,zuncert=True, complete_catalog=False, nside=None):
        super().__init__(skymap, observation_band, fast_cosmology, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights, Kcorr=Kcorr, zmax=zmax)
        
        self.mth = mth
        self.zcut = zcut
        self.complete_catalog = complete_catalog
        
        # Set redshift and colour limits based on whether Kcorrections are applied
        if Kcorr == True:
            if zcut is None:
                self.zcut = 0.5
            self.color_limit = galaxy_catalog.color_limit
        else:
            if zcut is None:
                self.zcut = self.zmax
            self.color_limit = [-np.inf,np.inf]
        
        if mth is None:
            self.mth = galaxy_catalog.mth()
        print('Catalogue apparent magnitude threshold: {}'.format(self.mth))
        
        #TODO make this changeable from command line?
        self.nfine = 10000
        self.ncoarse = 10


        #find galaxies below redshift cut, and with right colour information
        ind = np.where(((galaxy_catalog.z-3*galaxy_catalog.sigmaz) <= self.zcut) & \
                      (galaxy_catalog.m <= self.mth) & \
                      (self.color_limit[0] <= galaxy_catalog.color) & \
                      (galaxy_catalog.color <= self.color_limit[1]))[0]

        self.galz = galaxy_catalog.z[ind]
        self.galra = galaxy_catalog.ra[ind]
        self.galdec = galaxy_catalog.dec[ind]
        self.galm = galaxy_catalog.m[ind]
        self.galsigmaz = galaxy_catalog.sigmaz[ind]
        self.galcolor = galaxy_catalog.color[ind]
        self.nGal = len(self.galz)
        
        if zuncert == False:
            self.nfine = 1
            self.ncoarse = 1
            self.galsigmaz = np.zeros(len(self.galz))

        self.OmegaG = galaxy_catalog.OmegaG
        self.px_OmegaG = galaxy_catalog.px_OmegaG
        self.OmegaO = 1. - self.OmegaG
        self.px_OmegaO = 1. - self.px_OmegaG
        
        self.pxG = None
        self.pDG = None
        self.pG = 1.
        self.pxB = 0.
        self.pDB = 1.
        self.pB = 0.
        self.pxO = 0.
        self.pDO = 1.
        self.pO = 0.
        
        if nside is None:
            self.nside = galaxy_catalog.nside
        else:
            self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.sky_indices,self.sky_prob = skymap.above_percentile(0.9999,self.nside)
        
    def likelihood(self,H0,Lambda=0.):
        """
        Compute the full likelihood.
        
        Parameters
        ----------
        H0 : array of floats
            Hubble constant values in kms-1Mpc-1
        Lambda : float, optional
            Redshift evolution parameter (default=0)
        complete_catalog : bool, optional
            Assume that the galaxy catalogue is complete? (default=False)

        Returns
        -------
        float
            Returns likelihood, pxG, pDG, pG, pxB, pDB, pB, pxO, pDO, pO
            where likelihood = (pxG / pDG) * pG + (pxB / pDB) * pB + (pxO / pDO) * pO
        """
        self.pG = np.ones(len(H0))
        self.pxB = np.zeros(len(H0))
        self.pDB = np.ones(len(H0))
        self.pB = np.zeros(len(H0))
        self.pxO = np.zeros(len(H0))
        self.pDO = np.ones(len(H0))
        self.pO = 0.
        
        num = np.zeros(len(H0))
        den = np.zeros(len(H0))

        if not self.complete_catalog:
            print('Computing the beyond catalogue part')   
            for i,h in enumerate(H0):
                self.pG[i], self.pB[i], num[i], den[i] = self.pGB_DH0(h, self.mth, 1., Lambda=Lambda, zcut=self.zcut)
                self.pxB[i] = self.px_BH0(h, self.mth, 1., Lambda=Lambda, zcut=self.zcut)
            if self.zcut == self.zmax:
                self.pDB = (den - num) #* 1./self.npix
            else:
                print('Computing all integrals explicitly as zcut < zmax: this will take a little longer')
                for i,h in enumerate(H0):
                    self.pDB[i] = self.pD_BH0(h, self.mth, 1., Lambda=Lambda, zcut=self.zcut)
            print("{}% of this event's sky area appears to have galaxy catalogue support".format(self.px_OmegaG*100))
            if self.px_OmegaG < 0.999:
                self.pO = 1.
                #self.pDO = den * self.OmegaO ### alternative to calculating pDO directly below, but requires both px_OH0 and pD_OH0 to use dblquad (not quad) ###
                print('Computing the contribution outside the catalogue footprint')
                for i,h in enumerate(H0):
                    self.pxO[i] = self.px_OH0(h, skyprob=1., Lambda=Lambda)
                    self.pDO[i] = self.pD_OH0(h, skyprob=1., Lambda=Lambda)
        
        res,index_G = self.skymap.pixel_split(self.galra,self.galdec,self.nside) 
        
        pixels_G = np.zeros([len(H0),len(index_G)])
        print('Computing the in catalogue part')
        for i,idx in enumerate(index_G):
            z = self.galz[res[i]]
            ra = self.galra[res[i]]
            dec = self.galdec[res[i]]
            m = self.galm[res[i]]
            sigmaz = self.galsigmaz[res[i]]
            color = self.galcolor[res[i]]
            
            sampz, sampm, sampra, sampdec, sampcolor, count = gal_nsmear(z, sigmaz, m, ra, dec, color, 10, zcut=self.zcut)
            pxG, pDG = self.pxD_GH0(H0, sampz, sampm, sampra, sampdec, sampcolor, count, Lambda=Lambda)
            
            pixels_G[:,i] = (pxG / pDG) * self.pG*(1./self.npix) + (self.pxB*self.sky_prob[idx] / (self.pDB*(1./self.npix))) * self.pB*(1./self.npix)
            
        sum_pixels_G = np.sum(pixels_G,axis=1)
        index = np.arange(self.npix)
        index_O = np.delete(index,index_G)
        sum_pixels_O = np.sum(self.sky_prob[index_O])*self.pxO/(self.pDO*(1./self.npix))*(1./self.npix)
        likelihood = sum_pixels_G + sum_pixels_O
        return likelihood, sum_pixels_G, sum_pixels_O



class WholeSkyGalaxyCatalogLikelihood(GalaxyCatalogLikelihood):
    """
    Calculate the likelihood of H0 from one GW event, using the galaxy 
    catalogue method.
    
    Parameters
    ----------
    base_functions : gwcosmo.gwcosmo.BaseFunctions object
        p(x|z,H0)*p(z)*p(s|z)*p(M)*p(s|M) and p(D|z,H0)*p(z)*p(s|z)*p(M)*p(s|M)
    skymap : gwcosmo.likelihood.skymap.skymap object
        provides p(x|Omega) and skymap properties
    galaxy_catalog : gwcosmo.prior.catalog.galaxyCatalog object
        The galaxy catalogue
    fast_cosmology : gwcosmo.utilities.standard_cosmology.fast_cosmology object
        Cosmological model
    Kcorr : bool, optional
        Should K corrections be applied to the analysis? (default=False)
        Will raise an error if used in conjunction with a galaxy catalogue 
        without sufficient color information.
    mth : float, optional
        Specify an apparent magnitude threshold for the galaxy catalogue
        (default=None). If none, mth is estimated from the galaxy catalogue.
    zcut : float, optional
        An artificial redshift cut to the galaxy catalogue (default=None)
    zmax : float, optional
        The upper redshift limit for integrals (default=10.). Should be well 
        beyond the highest redshift reachable by GW data or selection effects.
    zuncert : bool, optional
        Should redshift uncertainties be marginalised over? (Default=True).
    
    """
    def __init__(self, galaxy_catalog, skymap, observation_band, fast_cosmology, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights, Kcorr=False, mth=None, zcut=None, zmax=10.,zuncert=True, complete_catalog=False):
        super().__init__(skymap, observation_band, fast_cosmology, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights, Kcorr=Kcorr, zmax=zmax)

        self.mth = mth
        self.zcut = zcut
        self.complete_catalog = complete_catalog
        
        # Set redshift and colour limits based on whether Kcorrections are applied
        if Kcorr == True:
            if zcut is None:
                self.zcut = 0.5
            self.color_limit = galaxy_catalog.color_limit
        else:
            if zcut is None:
                self.zcut = self.zmax
            self.color_limit = [-np.inf,np.inf]
        
        if mth is None:
            self.mth = galaxy_catalog.mth()
        print('Catalogue apparent magnitude threshold: {}'.format(self.mth))
        
        #TODO make this changeable from command line?
        self.nfine = 10000
        self.ncoarse = 10


        #find galaxies below redshift cut, and with right colour information
        ind = np.where(((galaxy_catalog.z-3*galaxy_catalog.sigmaz) <= self.zcut) & \
                      (galaxy_catalog.m <= self.mth) & \
                      (self.color_limit[0] <= galaxy_catalog.color) & \
                      (galaxy_catalog.color <= self.color_limit[1]))[0]

        self.galz = galaxy_catalog.z[ind]
        self.galra = galaxy_catalog.ra[ind]
        self.galdec = galaxy_catalog.dec[ind]
        self.galm = galaxy_catalog.m[ind]
        self.galsigmaz = galaxy_catalog.sigmaz[ind]
        self.galcolor = galaxy_catalog.color[ind]
        self.nGal = len(self.galz)
        
        if zuncert == False:
            self.nfine = 1
            self.ncoarse = 1
            self.galsigmaz = np.zeros(len(self.galz))

        self.OmegaG = galaxy_catalog.OmegaG
        self.px_OmegaG = galaxy_catalog.px_OmegaG
        self.OmegaO = 1. - self.OmegaG
        self.px_OmegaO = 1. - self.px_OmegaG
        
        self.pxG = None
        self.pDG = None
        self.pG = 1.
        self.pxB = 0.
        self.pDB = 1.
        self.pB = 0.
        self.pxO = 0.
        self.pDO = 1.
        self.pO = 0.


    def pxD_GH0_multi(self,H0,Lambda=0.):
        """
        Evaluate p(x|G,H0) and p(D|G,H0).

        Parameters
        ----------
        H0 : array of floats
            Hubble constant value(s) in kms-1Mpc-1
        Lambda : float, optional
            redshift evolution parameter (default=0)

        Returns
        -------
        arrays
            p(x|G,H0), p(D|G,H0)
        """
        
        galindex_sep = {}
        if self.luminosity_weights.luminosity_weights == True:
            # TODO: find better selection criteria for sampling
            mlim = np.percentile(np.sort(self.galm),0.01) # more draws for galaxies in brightest 0.01 percent
            samp_res = {'fine': self.nfine, 'coarse': self.ncoarse}
            galindex = {'fine': np.where(self.galm <= mlim)[0], 'coarse': np.where(mlim < self.galm)[0]}
            
            # for arrays with more than 1million entries, break into sub arrays
            no_chunks_coarse = int(np.ceil(len(galindex['coarse'])/1000000))
            chunks_coarse = np.array_split(galindex['coarse'],no_chunks_coarse)
            galindex_sep['coarse'] = {i+1 : chunks_coarse[i] for i in range(no_chunks_coarse)} 
            galindex_sep['fine'] = {i : galindex['fine'] for i in range(1)} 
        else:
            samp_res = {'coarse': self.ncoarse}
            galindex = {'coarse': np.arange(self.nGal)}
            
            # for arrays with more than 1million entries, break into sub arrays
            no_chunks_coarse = int(np.ceil(len(galindex['coarse'])/1000000))
            chunks_coarse = np.array_split(galindex['coarse'],no_chunks_coarse)
            galindex_sep['coarse'] = {i : chunks_coarse[i] for i in range(no_chunks_coarse)} 
        
        K = sum(len(v) for v in galindex.values()) # total number of sub arrays
        tempnum = np.zeros([K,len(H0)])
        tempden = np.zeros([K,len(H0)])
        
        # loop over sub arrays of galaxies
        for i,key in enumerate(samp_res):
            print('{} galaxies are getting sampled {}ly'.format(len(galindex[key]),key))
            for n, key2 in enumerate(galindex_sep[key]):
                zs = self.galz[galindex_sep[key][key2]]
                sigmazs = self.galsigmaz[galindex_sep[key][key2]]
                ms = self.galm[galindex_sep[key][key2]]
                ras = self.galra[galindex_sep[key][key2]]
                decs = self.galdec[galindex_sep[key][key2]]
                colors = self.galcolor[galindex_sep[key][key2]]
                
                sampz, sampm, sampra, sampdec, sampcolor, count = gal_nsmear(zs, sigmazs, ms, ras, decs, colors, samp_res[key], zcut=self.zcut)
                    
                tempnum[key2,:],tempden[key2,:] = self.pxD_GH0(H0, sampz, sampm, sampra, sampdec, sampcolor, count, Lambda=Lambda)
                
        num = np.sum(tempnum,axis=0)/self.nGal
        den = np.sum(tempden,axis=0)/self.nGal

        return num,den        

    def likelihood(self,H0,Lambda=0.):
        """
        Compute the full likelihood.
        
        Parameters
        ----------
        H0 : array of floats
            Hubble constant values in kms-1Mpc-1
        Lambda : float, optional
            Redshift evolution parameter (default=0)
        complete_catalog : bool, optional
            Assume that the galaxy catalogue is complete? (default=False)

        Returns
        -------
        float
            Returns likelihood, pxG, pDG, pG, pxB, pDB, pB, pxO, pDO, pO
            where likelihood = (pxG / pDG) * pG + (pxB / pDB) * pB + (pxO / pDO) * pO
        """
        self.pG = np.ones(len(H0))
        self.pxB = np.zeros(len(H0))
        self.pDB = np.ones(len(H0))
        self.pB = np.zeros(len(H0))
        self.pxO = np.zeros(len(H0))
        self.pDO = np.ones(len(H0))
        self.pO = 0.
        
        num = np.zeros(len(H0))
        den = np.zeros(len(H0))
        
        print('Computing the in-catalogue part')
        self.pxG, self.pDG = self.pxD_GH0_multi(H0,Lambda=Lambda)

        if not self.complete_catalog:
            print('Computing the beyond catalogue part')   
            for i,h in enumerate(H0):
                self.pG[i], self.pB[i], num[i], den[i] = self.pGB_DH0(h, self.mth, self.OmegaG, Lambda=Lambda, zcut=self.zcut)
                self.pxB[i] = self.px_BH0(h, self.mth, self.px_OmegaG, Lambda=Lambda, zcut=self.zcut)
            if self.zcut == self.zmax:
                self.pDB = (den - num) * self.OmegaG
            else:
                print('Computing all integrals explicitly as zcut < zmax: this will take a little longer')
                for i,h in enumerate(H0):
                    self.pDB[i] = self.pD_BH0(h, self.mth, self.OmegaG, Lambda=Lambda, zcut=self.zcut)
            print("{}% of this event's sky area appears to have galaxy catalogue support".format(self.px_OmegaG*100))
            if self.px_OmegaG < 0.999:
                self.pO = self.OmegaO
                #self.pDO = den * self.OmegaO ### alternative to calculating pDO directly below, but requires both px_OH0 and pD_OH0 to use dblquad (not quad) ###
                print('Computing the contribution outside the catalogue footprint')
                for i,h in enumerate(H0):
                    self.pxO[i] = self.px_OH0(h, skyprob=self.px_OmegaO, Lambda=Lambda)
                    self.pDO[i] = self.pD_OH0(h, skyprob=self.OmegaO, Lambda=Lambda)

        likelihood = (self.pxG / self.pDG) * self.pG + (self.pxB / self.pDB) * self.pB + (self.pxO / self.pDO) * self.pO
        return likelihood

    def return_components(self):
        return self.pxG, self.pDG, self.pG, self.pxB, self.pDB, self.pB, self.pxO, self.pDO, self.pO
        
    def __call__(self, H0, Lambda=0.):
        return self.likelihood(H0, Lambda=Lambda)
        


class DirectCounterpartLikelihood(gwcosmoLikelihood):
    """
    Calculate the likelihood of H0 from one GW event, using the counterpart 
    method.
    
    This method is fast relative to the catalog methods, as it does not 
    require an integral over either sky or absolute magnitude, only redshift.
    
    Parameters
    ----------
    base_functions : gwcosmo.gwcosmo.BaseFunctions object
        p(x|z,H0) and p(D|z,H0)*p(z)*p(s|z)
        
    """
    def __init__(self, counterpart_z,counterpart_sigmaz, px_zH0, pD_zH0, zprior, zrates, zmax=10.):
        self.counterpart_z = counterpart_z
        self.counterpart_sigmaz = counterpart_sigmaz
        super().__init__(px_zH0, pD_zH0, zprior, zrates, zmax=zmax)
        
        self.px = None
        self.pD = None
        
    def px_H0(self,H0):
        """
        Returns p(x|H0,counterpart)
        The likelihood of the GW data given H0 and direct counterpart.

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(x|H0,counterpart)
        """
        zsmear =  z_nsmear(self.counterpart_z, self.counterpart_sigmaz, 10000)
        num = np.zeros(len(H0))
        for k,h in enumerate(H0):
            num[k] = np.sum(self.px_zH0(zsmear,h)) 
            # TODO should this include p(s|z)? Would come into play
            # for host galaxies with large redshift uncertainty.
        return num
        
    def likelihood(self,H0,Lambda=0.):
        px = self.px_H0(H0)
        pD = np.zeros(len(H0))
        for i,h in enumerate(H0):
            pD[i] = self.pD_OH0(h, skyprob=1., Lambda=Lambda)
        likelihood = px/pD
        self.px = px
        self.pD = pD
        return likelihood
        
    def return_components(self):
        return self.px, self.pD, 1., 0., 1., 0., 0., 1., 0.
        
    def __call__(self, H0, Lambda=0.):
        return self.likelihood(H0, Lambda=Lambda)

        
        
class EmptyCatalogLikelihood(gwcosmoLikelihood):
    """
    Calculate the likelihood of H0 from one GW event, using the empty catalogue 
    method.
    
    Calculations assuming no EM data (either counterpart or catalog).
    All information comes from the distance distribution of GW events
    or population assumptions which have not yet been marginalized over.
    
    This method is fast relative to the catalog methods, as it does not 
    require an integral over either sky or absolute magnitude, only redshift.
    
    Parameters
    ----------
    base_functions : gwcosmo.gwcosmo.BaseFunctions object
        p(x|z,H0)*p(z)*p(s|z) and p(D|z,H0)*p(z)*p(s|z)
    """
    def __init__(self, px_zH0, pD_zH0, zprior, zrates, zmax=10.):
        super().__init__(px_zH0, pD_zH0, zprior, zrates, zmax=zmax)
        
        self.px = None
        self.pD = None
        
    def likelihood(self,H0,Lambda=0.):
        px = np.zeros(len(H0))
        pD = np.zeros(len(H0))
        for i,h in enumerate(H0):
            px[i] = self.px_OH0(h, skyprob=1., Lambda=Lambda)
            pD[i] = self.pD_OH0(h, skyprob=1., Lambda=Lambda)
        likelihood = px/pD
        self.px = px
        self.pD = pD
        return likelihood
        
    def return_components(self):
        return 0., 1., 0., 0., 1., 0., self.px, self.pD, 1.
        
    def __call__(self, H0, Lambda=0.):
        return self.likelihood(H0, Lambda=Lambda)



################################################################################
################################ ADDITIONAL CLASSES ############################
################################################################################

class LuminosityWeighting(object):
    """
    Host galaxy probability relation to luminosity
    """
    
    def __init__(self):
        self.luminosity_weights = True
        
    def weighted_call(self, M):
        return L_M(M)
        
    def __call__(self, M):
        return self.weighted_call(M)
        
class UniformWeighting(object):
    """
    Host galaxy probability relation to luminosity
    """
    
    def __init__(self):
        self.luminosity_weights = False
        
    def unweighted_call(self, M):
        return 1.
        
    def __call__(self, M):
        return self.unweighted_call(M)
        

class RedshiftEvolution():
    """
    Merger rate relation to redshift
    
    TODO: consider how Lambda might need to be marginalised over in future
    """
    
    def __init__(self):
        self.redshift_evolution = True
        
    def evolving(self, z, Lambda=0.):
        return (1+z)**Lambda

    def __call__(self, z, Lambda=0.):
            return self.evolving(z,Lambda=Lambda)
            
class RedshiftNonEvolution():
    """
    Merger rate relation to redshift
    """
    
    def __init__(self):
        self.redshift_evolution = False
        
    def constant(self, z, Lambda=0.):
        return 1.
        
    def __call__(self, z, Lambda=0.):
        return self.constant(z,Lambda=Lambda)


################################################################################
################################ INTERNAL FUNCTIONS ############################
################################################################################


def z_nsmear(z, sigmaz, nsmear, zcut=10.):
    """
    Draw redshift samples from a galaxy. Ensure no samples fall below z=0
    Remove samples above the redshift cut. zcut cannot be used as an upper limit
    for the draw, as this will cause an overdensity of support.
    """
    a = (0.0 - z) / sigmaz
    zsmear = truncnorm.rvs(a, 5, loc=z, scale=sigmaz, size=nsmear)
    zsmear = zsmear[np.where(zsmear<zcut)[0]].flatten()
    return zsmear #TODO order these before returning them?
    
    
def gal_nsmear(z, sigmaz, m, ra, dec, color, nsmear, zcut=10.):
    """
    Draw redshift samples from a galaxy. Ensure no samples fall below z=0
    Remove samples above the redshift cut. zcut cannot be used as an upper limit
    for the draw, as this will cause an overdensity of support.
    """
    
    # get redshift samples, carefully not going below zero
    a = (0.0 - z) / sigmaz
    sampz = truncnorm.rvs(a, 5, loc=z, scale=sigmaz, size=[nsmear,len(z)]).flatten('F')

    # repeat arrays for other gal parameters to give each sample full info
    sampcolor = np.repeat(color,nsmear)
    sampm = np.repeat(m,nsmear)
    sampra = np.repeat(ra,nsmear)
    sampdec = np.repeat(dec,nsmear)
    count = np.ones(len(sampz))*nsmear

    # remove samples above the redshift cut
    ind = np.where(sampz < zcut)[0]
    sampz = sampz[ind]
    sampcolor = sampcolor[ind]
    sampm = sampm[ind]
    sampra = sampra[ind]
    sampdec = sampdec[ind]
    count = count[ind]
    
    # sort array in ascending order so that px_zH0 and pD_zH0 don't freak out
    ind = np.argsort(sampz)
    sampz = sampz[ind]
    sampcolor = sampcolor[ind]
    sampm = sampm[ind]
    sampra = sampra[ind]
    sampdec = sampdec[ind]
    count = count[ind]
    
    return sampz, sampm, sampra, sampdec, sampcolor, count
    

