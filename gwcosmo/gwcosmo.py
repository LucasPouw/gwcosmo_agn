"""
gwcosmoLikelihood Module
Rachel Gray, Archisman Ghosh, Ignacio Magana, John Veitch, Ankan Sur

In general:
p(x|z,H0,\Omega) is written as p(x|dl(z,H0))*p(x|\Omega)
p(x|dL(z,H0)): self.norms[H0]*self.px_dl(dl(z,H0))
p(x|\Omega): self.skymap.skyprob(ra,dec) or self.skymap.prob[idx]
p(D|z,H0): pdet.pD_zH0_eval(z,H0)
p(s|M(H0)): L_M(M) or L_mdl(m,dl(z,H0))
p(z): zprior(z)
p(M|H0): SchechterMagFunction(H0)(M)
p(\Omega): this term comes out the front and cancels in most cases,
and so does not appear explicitly.
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

import time
import progressbar


################################################################################
################################# THE MAIN CLASSES #############################
################################################################################


class GalaxyCatalogLikelihood(object):
    """
    Calculations assuming no EM data (either counterpart or catalog).
    All information comes from the distance distribution of GW events
    or population assumptions which have not yet been marginalized over.
    
    This method is fast relative to the catalog methods, as it does not 
    require an integral over either sky or absolute magnitude, only redshift.
    """
    def __init__(self, base_functions, skymap, galaxy_catalog, fast_cosmology, Kcorr=False, mth=None, zcut=None, zmax=10.):
        self.base = base_functions
        #self.galaxy_catalog = galaxy_catalog
        self.cosmo = fast_cosmology
        self.mth = mth
        self.zmax = zmax
        self.zcut = zcut
        self.skymap = skymap
        self.Kcorr = Kcorr
        
        # read in Schechter function parameters corresponding to galaxy catalogue band
        self.band = galaxy_catalog.band
        sp = SchechterParams(self.band)
        self.alpha = sp.alpha
        self.Mstar_obs = sp.Mstar
        self.Mmin_obs = sp.Mmin
        self.Mmax_obs = sp.Mmax
        
        # Set redshift and colour limits based on whether Kcorrections are applied
        if Kcorr == True:
            if zcut is None:
                self.zcut = 0.5
            self.color_name = galaxy_catalog.color_name
            self.color_limit = galaxy_catalog.color_limit
        else:
            if zcut is None:
                self.zcut = self.zmax
            self.color_limit = [-np.inf,np.inf]
        
        
        if mth is None:
            self.mth = galaxy_catalog.mth()
        
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

        self.OmegaG = galaxy_catalog.OmegaG
        self.px_OmegaG = galaxy_catalog.px_OmegaG
        
        
    def pxD_GH0(self,H0,Lambda=0.):
        """
        The "in catalog" part of the new skypatch method
        using a catalog which follows the GW event's sky patch contour
        p(x|D,G,H0)

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        arrays
            numerator and denominator
        """      
        if self.base.luminosity_weights.luminosity_weights == True:
            # TODO: find better selection criteria for sampling
            mlim = np.percentile(np.sort(self.galm),0.01) # more draws for galaxies in brightest 0.01 percent
            nsamps = {'fine': self.nfine, 'coarse': self.ncoarse}
            galindex = {'fine': np.where(self.galm <= mlim)[0], 'coarse': np.where(mlim < self.galm)[0]}
        else:
            nsamps = {'coarse': self.ncoarse}
            galindex = {'coarse': np.arange(self.nGal)}
        
        tempnum = np.zeros([len(nsamps),len(H0)])
        tempden = np.zeros([len(nsamps),len(H0)])
        for i,key in enumerate(nsamps):
            print('{} galaxies are getting sampled {}ly'.format(len(galindex[key]),key))
            zs = self.galz[galindex[key]]
            sigmazs = self.galsigmaz[galindex[key]]
            ms = self.galm[galindex[key]]
            ras = self.galra[galindex[key]]
            decs = self.galdec[galindex[key]]
            colors = self.galcolor[galindex[key]]  
            
            sampz, sampm, sampra, sampdec, sampcolor, count = gal_nsmear(zs, sigmazs, ms, ras, decs, colors, nsamps[key], zcut=self.zcut)
            
            if self.Kcorr == True:
                Kcorr = calc_kcor(self.band,sampz,self.color_name,colour_value=sampcolor)
            else:
                Kcorr = 0.
                
            tempsky = self.skymap.skyprob(sampra, sampdec)*self.skymap.npix
            
            zweights = self.base.zrates(sampz)
            
            for k,h in enumerate(H0):
                numinner = self.base.px_zH0(sampz,h)
                deninner = self.base.pD_zH0(sampz,h)
                if self.base.luminosity_weights.luminosity_weights == True:
                    sampAbsM = M_mdl(sampm, self.cosmo.dl_zH0(sampz, h), Kcorr=Kcorr)
                else:
                    sampAbsM = 1.0 # value is irrelevant as weights will evaluate to 1 for all samples
                    
                Lweights = self.base.luminosity_weights(sampAbsM)
                normsamp = 1./count
                
                tempnum[i,k] = np.sum(numinner*tempsky*Lweights*zweights*normsamp)
                tempden[i,k] = np.sum(deninner*Lweights*zweights*normsamp)
                
        num = np.sum(tempnum,axis=0)/self.nGal
        den = np.sum(tempden,axis=0)/self.nGal

        return num,den
        
    def pGB_DH0(self,H0,mth,Lambda=0.,zcut=10.,zmax=10.):
        """
        The probability that the host galaxy of a detected GW event is inside
        the galaxy catalogue.
        """
        Mmin = M_Mobs(H0,self.Mmin_obs)
        Mmax = M_Mobs(H0,self.Mmax_obs)

        num = dblquad(self.base.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,zcut,lambda x: Mmin,lambda x: min(max(M_mdl(mth,self.cosmo.dl_zH0(x,H0)),Mmin),Mmax),args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]

        den = dblquad(self.base.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,zmax,lambda x: Mmin,lambda x: Mmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]

        integral = num/den
        
        pG = integral*self.OmegaG
        pB = self.OmegaG*(1.-integral)
        return pG, pB, num, den
        
    def px_BH0(self,H0,mth,Lambda=0.,zcut=10.,zmax=10.):
        """
        Evaluate the likelihood of the GW data beyond the galaxy catalogue
        If zcut >= zmax then a single integral is performed.
        If zcut < zmax then an additional integral is performed
        """
        Mmin = M_Mobs(H0,self.Mmin_obs)
        Mmax = M_Mobs(H0,self.Mmax_obs)
            
        below_zcut_integral = dblquad(self.base.px_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,zcut,lambda x: min(max(M_mdl(mth,self.cosmo.dl_zH0(x,H0)),Mmin),Mmax), lambda x: Mmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        
        above_zcut_integral = 0.
        if self.zcut < zmax:
            above_zcut_integral = dblquad(self.base.px_zH0_times_pz_times_ps_z_times_pM_times_ps_M,zcut,zmax,lambda x: Mmin, lambda x: Mmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        
        integral = below_zcut_integral + above_zcut_integral

        return integral * self.px_OmegaG
        
    def pD_BH0(self,H0,mth,Lambda=0.,zcut=10.,zmax=10.):
        """
        Evaluate the likelihood of the GW data beyond the galaxy catalogue
        If zcut >= zmax then a single integral is performed.
        If zcut < zmax then an additional integral is performed
        """
        Mmin = M_Mobs(H0,self.Mmin_obs)
        Mmax = M_Mobs(H0,self.Mmax_obs)
            
        below_zcut_integral = dblquad(self.base.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,zcut,lambda x: min(max(M_mdl(mth,self.cosmo.dl_zH0(x,H0)),Mmin),Mmax), lambda x: Mmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        
        above_zcut_integral = 0.
        if zcut < zmax:
            above_zcut_integral = dblquad(self.base.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,zcut,zmax,lambda x: Mmin, lambda x: Mmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]

        integral = below_zcut_integral + above_zcut_integral

        return integral * self.OmegaG 
        
    def px_OH0(self,H0,Lambda=0.,zmax=10.):
        Mmin = M_Mobs(H0,self.Mmin_obs)
        Mmax = M_Mobs(H0,self.Mmax_obs)
        
        integral = dblquad(self.base.px_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,zmax,lambda x: Mmin,lambda x: Mmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        return integral * (1.-self.px_OmegaG)
        
    def pD_OH0(self,H0,Lambda=0.,zmax=10.):
        Mmin = M_Mobs(H0,self.Mmin_obs)
        Mmax = M_Mobs(H0,self.Mmax_obs)
        
        integral = dblquad(self.base.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,zmax,lambda x: Mmin,lambda x: Mmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        return integral * (1.-self.OmegaG)
        
    def pO_DH0(self):
        """
        The probability that the host galaxy of a detected GW event lies outside
        the sky area of the galaxy catalogue.
        """
        return 1.0 - self.OmegaG
        
    def likelihood(self,H0,Lambda=0.,complete_catalog=False):
        """
        Complete the full likelihood.
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
        self.pxG, self.pDG = self.pxD_GH0(H0,Lambda=Lambda)

        if not complete_catalog:
            print('Computing the beyond catalogue part')   
            for i,h in enumerate(H0):
                self.pG[i], self.pB[i], num[i], den[i] = self.pGB_DH0(h,self.mth,Lambda=Lambda,zcut=self.zcut,zmax=self.zmax)
                self.pxB[i] = self.px_BH0(h,self.mth,Lambda=Lambda,zcut=self.zcut,zmax=self.zmax)
            if self.zcut == self.zmax:
                self.pDB = (den - num) * self.OmegaG
            else:
                print('Computing all integrals explicitly as zcut < zmax: this will take a little longer')
                for i,h in enumerate(H0):
                    self.pDB[i] = self.pD_BH0(h,self.mth,Lambda=Lambda,zcut=self.zcut,zmax=self.zmax)
            if self.px_OmegaG < 0.999:
                self.pO = self.pO_DH0()
                self.pDO = den * (1.-self.OmegaG)
                print('Computing the contribution outside the catalogue footprint: this will take a little longer')
                for i,h in enumerate(H0):
                    self.pxO[i] = self.px_OH0(h,Lambda=Lambda,zmax=self.zmax)
        
        likelihood = (self.pxG / self.pDG) * self.pG + (self.pxB / self.pDB) * self.pB + (self.pxO / self.pDO) * self.pO
        return likelihood, self.pxG, self.pDG, self.pG, self.pxB, self.pDB, self.pB, self.pxO, self.pDO, self.pO


class DirectCounterpartLikelihood(object):
    """    
    This method is fast relative to the catalog methods, as it does not 
    require an integral over either sky or absolute magnitude, only redshift.
    """
    def __init__(self, base_functions):
        self.base = base_functions
        
    def px_H0(self,H0,counterpart_z,counterpart_sigmaz):
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
        zsmear =  z_nsmear(counterpart_z, counterpart_sigmaz, 10000)
        num = np.zeros(len(H0))
        for k,H in enumerate(H0):
            num[k] = np.sum(self.base.px_zH0(zsmear,H)) 
            # TODO should this include p(s|z)? Would come into play
            # for host galaxies with large redshift uncertainty.
        return num
        
    def pD_H0(self,H0,Lambda=0.,zmax=10.):
        return quad(self.base.pD_zH0_times_pz_times_ps_z,0.,zmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        
    def likelihood(self,H0,counterpart_z,counterpart_sigmaz,Lambda=0.,zmax=10.):
        px = self.px_H0(H0,counterpart_z,counterpart_sigmaz)
        pD = np.zeros(len(H0))
        for i,h in enumerate(H0):
            pD[i] = self.pD_H0(h,Lambda=Lambda,zmax=zmax)
        likelihood = px/pD
        return likelihood, px, pD
        
        
class EmptyCatalogLikelihood(object):
    """
    Calculations assuming no EM data (either counterpart or catalog).
    All information comes from the distance distribution of GW events
    or population assumptions which have not yet been marginalized over.
    
    This method is fast relative to the catalog methods, as it does not 
    require an integral over either sky or absolute magnitude, only redshift.
    """
    def __init__(self, base_functions):
        self.base = base_functions
        
    def px_H0(self,H0,Lambda=0.,zmax=10.):
        return quad(self.base.px_zH0_times_pz_times_ps_z,0.,zmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        
    def pD_H0(self,H0,Lambda=0.,zmax=10.):
        return quad(self.base.pD_zH0_times_pz_times_ps_z,0.,zmax,args=(H0,Lambda),epsabs=0,epsrel=1.49e-4)[0]
        
    def likelihood(self,H0,Lambda=0.,zmax=10.):
        px = np.zeros(len(H0))
        pD = np.zeros(len(H0))
        for i,h in enumerate(H0):
            px[i] = self.px_H0(h,Lambda=Lambda,zmax=zmax)
            pD[i] = self.pD_H0(h,Lambda=Lambda,zmax=zmax)
        likelihood = px/pD
        return likelihood, px, pD

################################################################################
################################ ADDITIONAL CLASSES ############################
################################################################################


class LuminosityWeighting():
    """
    Host galaxy probability relation to luminosity
    """
    def __init__(self, luminosity_weights=False):
        self.luminosity_weights = luminosity_weights
        
    def weighted_call(self, M):
        return L_M(M)
        
    def unweighted_call(self, M):
        return 1.
        
    def __call__(self, M):
        if self.luminosity_weights:
            return self.weighted_call(M)
        else:
            return self.unweighted_call(M)
        

class RedshiftEvolution():
    """
    Merger rate relation to redshift
    
    TODO: consider how Lambda might need to be marginalised over in future
    """
    def __init__(self, redshift_evolution=False):
        self.redshift_evolution = redshift_evolution
        
    def evolving(self, z, Lambda=0.):
        return (1+z)**Lambda
        
    def constant(self, z, Lambda=0.):
        return 1.
        
    def __call__(self, z, Lambda=0.):
        if self.redshift_evolution:
            return self.evolving(z,Lambda=Lambda)
        else:
            return self.constant(z,Lambda=Lambda)

    
class BaseFunctions():
    """
    Parameters
    ----------
    pxorpD_zH0tion : function
        p(x|z,H0) or p(D|z,H0)
    zprior : gwcosmo.utilities.standard_cosmology.redshift_prior object
        The redshift prior, p(z)
    zrates : gwcosmo.gwcosmo.RedshiftEvolution object
        Merger rate evolution, p(s|z)
    """
    def __init__(self, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights):
        self.px_zH0 = px_zH0
        self.pD_zH0 = pD_zH0
        self.zprior=zprior
        self.zrates=zrates
        self.luminosity_prior=luminosity_prior
        self.luminosity_weights=luminosity_weights
        
    def px_zH0_times_pz_times_ps_z(self, z, H0, Lambda=0.):
        return self.px_zH0(z,H0)*self.zprior(z)*self.zrates(z,Lambda=Lambda)
        
    def pD_zH0_times_pz_times_ps_z(self, z, H0, Lambda=0.):
        return self.pD_zH0(z,H0)*self.zprior(z)*self.zrates(z,Lambda=Lambda)
        
    def px_zH0_times_pz_times_ps_z_times_pM_times_ps_M(self, M, z, H0, Lambda=0.):
        return self.px_zH0(z,H0)*self.zprior(z)*self.zrates(z,Lambda=Lambda) \
                *self.luminosity_prior(M,H0)*self.luminosity_weights(M)
        
    def pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M(self, M, z, H0, Lambda=0.):
        return self.pD_zH0(z,H0)*self.zprior(z)*self.zrates(z,Lambda=Lambda) \
                *self.luminosity_prior(M,H0)*self.luminosity_weights(M)
        

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
    


################################################################################
#################### OLD MODULE (TO BE REMOVED ONCE REDUNDANT) #################
################################################################################

class gwcosmoLikelihood(object):
    """
    A class to hold all the individual components of the posterior for H0,
    and methods to stitch them together in the right way.

    Parameters
    ----------
    GW_data : gwcosmo.likelihood.posterior_samples.posterior_samples object
        Gravitational wave event samples
    skymap : gwcosmo.likelihood.skymap.skymap object
        Gravitational wave event skymap
    galaxy_catalog : gwcosmo.prior.catalog.galaxyCatalog object
        The relevant galaxy catalog
    EM_counterpart : gwcosmo.prior.catalog.galaxyCatalog object, optional
        EM_counterpart data (default=None)
        If not None, will default to using this over the galaxy_catalog
    Omega_m : float, optional
        The matter fraction of the universe (default=0.3)
    linear : bool, optional
        Use linear cosmology (default=False)
    basic : bool, optional
        If True, uses pdet suitable for MDC analysis (default=False)
    uncertainty : bool, optional
        If true, redshift uncertainty will be assumed and corrected
        for (default=False)
    rate : str, optional
        specifies rate evolution model, 'const' or 'evolving'
    Kcorr : bool, optional
        If true, will attempt to apply K corrections (default=False)
    """

    def __init__(self, H0, GW_data, skymap, galaxy_catalog, pdet, reweight=False, EM_counterpart=None,
                 Omega_m=0.308, linear=False, weighted=False, basic=False, uncertainty=False,
                 rate='constant', population_params=None, area=0.999, Kcorr=False):

        self.H0 = H0
        self.pdet = pdet
        self.Omega_m = Omega_m
        self.linear = linear
        self.weighted = weighted
        self.basic = basic
        self.uncertainty = uncertainty
        self.skymap = skymap
        self.area = area
        self.Kcorr = Kcorr
        self.reweight = reweight

        if population_params is None:
            self.mass_distribution = pdet.mass_distribution
            self.alpha = 1.6
            self.mmin = 5
            self.mmax = 100
            self.Lambda = 0
        else:
            self.mass_distribution = population_params['mass_distribution']
            self.alpha = population_params['alpha']
            self.mmin = population_params['mmin']
            self.mmax = population_params['mmax']
            self.Lambda = population_params['Lambda']

        try:
            self.band = galaxy_catalog.band
        except:
            self.band = 'B' # hack so that population analysis works from command line.

        sp = SchechterParams(self.band)
        self.alpha_sp = sp.alpha
        self.Mstar_obs = sp.Mstar
        self.Mobs_min = sp.Mmin
        self.Mobs_max = sp.Mmax

        if galaxy_catalog == None:
            self.galaxy_catalog = None
            self.mth = None
            self.EM_counterpart = EM_counterpart
            self.whole_cat = True
        else:
            self.galaxy_catalog = galaxy_catalog
            self.mth = galaxy_catalog.mth()
            self.EM_counterpart = None


        if GW_data is not None:
            temps = []
            norms = []
            if reweight == True:
                print("Reweighting samples")
            seed = np.random.randint(10000)
            bar = progressbar.ProgressBar()
            z_max = []
            for H0 in bar(self.H0):
                if reweight == True:
                    zkernel, norm = GW_data.marginalized_redshift_reweight(H0, self.mass_distribution, self.alpha, self.mmin, self.mmax)
                else:
                    zkernel, norm = GW_data.marginalized_redshift(H0)

                zmin = np.min(zkernel.dataset)
                zmax = np.max(zkernel.dataset)
                z_max.append(3*zmax)
                z_array = np.linspace(zmin, zmax, 500)
                vals = zkernel(z_array)
                temps.append(interp1d(z_array, vals,bounds_error=False,fill_value=0))
                norms.append(norm)
            self.zmax_GW=z_max
            self.temps = np.array(temps)
            self.norms = np.array(norms)

        if (GW_data is None and self.EM_counterpart is None):
            dl_array, vals = self.skymap.marginalized_distance()
            self.temp = splrep(dl_array,vals)

        if (GW_data is None and self.EM_counterpart is not None):
            counterpart = self.EM_counterpart
            dl_array, vals = self.skymap.lineofsight_distance(counterpart.ra, counterpart.dec)
            self.temp = splrep(dl_array,vals)

        # TODO: calculate mth for the patch of catalog being used, if whole_cat=False


        if (self.EM_counterpart is None and self.galaxy_catalog is not None):
            self.radec_lim = self.galaxy_catalog.radec_lim[0]
            if self.radec_lim == 0:
                self.whole_cat = True
            else:
                self.whole_cat = False
            self.ra_min = self.galaxy_catalog.radec_lim[1]
            self.ra_max = self.galaxy_catalog.radec_lim[2]
            self.dec_min = self.galaxy_catalog.radec_lim[3]
            self.dec_max = self.galaxy_catalog.radec_lim[4]

            if self.Kcorr == True:
                self.zcut = 0.5
                self.color_name = self.galaxy_catalog.color_name
                self.color_limit = self.galaxy_catalog.color_limit
            else:
                self.zcut = 10.
                self.color_limit = [-np.inf,np.inf]

            if self.whole_cat == False:
                def skynorm(dec,ra):
                    return np.cos(dec)
                self.catalog_fraction = dblquad(skynorm,self.ra_min,self.ra_max,
                                                lambda x: self.dec_min,
                                                lambda x: self.dec_max,
                                                epsabs=0,epsrel=1.49e-4)[0]/(4.*np.pi)
                self.rest_fraction = 1-self.catalog_fraction
                print('This catalog covers {}% of the full sky'.format(self.catalog_fraction*100))


            #find galaxies within the bounds of the galaxy catalog
            sel = np.argwhere((self.ra_min <= self.galaxy_catalog.ra) & \
                              (self.galaxy_catalog.ra <= self.ra_max) & \
                              (self.dec_min <= self.galaxy_catalog.dec) & \
                              (self.galaxy_catalog.dec <= self.dec_max) & \
                              ((self.galaxy_catalog.z-3*self.galaxy_catalog.sigmaz) <= self.zcut) & \
                              (self.color_limit[0] <= galaxy_catalog.color) & \
                              (galaxy_catalog.color <= self.color_limit[1]))

            self.allz = self.galaxy_catalog.z[sel].flatten()
            self.allra = self.galaxy_catalog.ra[sel].flatten()
            self.alldec = self.galaxy_catalog.dec[sel].flatten()
            self.allm = self.galaxy_catalog.m[sel].flatten()
            self.allsigmaz = self.galaxy_catalog.sigmaz[sel].flatten()
            self.allcolor = self.galaxy_catalog.color[sel].flatten()
            self.mth = self.galaxy_catalog.mth()
            self.nGal = len(self.allz)

            if self.uncertainty == False:
                self.nsmear_fine = 1
                self.nsmear_coarse = 1
                self.allsigmaz = np.zeros(len(self.allz))
            else:
                self.nsmear_fine = 10000
                self.nsmear_coarse = 20

        self.pDG = None
        self.pGD = None
        self.pnGD = None
        self.pDnG = None

        # Note that zmax is an artificial limit that
        # should be well above any redshift value that could
        # impact the results for the considered H0 values.
        self.zmax = 10.

        self.zprior = redshift_prior(Omega_m=self.Omega_m, linear=self.linear)
        self.cosmo = fast_cosmology(Omega_m=self.Omega_m, linear=self.linear)
        self.rate = rate

    def ps_z(self, z):
        if self.rate == 'constant':
            return 1.0
        if self.rate == 'evolving':
            return (1.0+z)**self.Lambda

    def px_dl(self, dl, temp):
        """
        Returns a probability for a given distance dl
        from the interpolated function.
        """
        if self.reweight==True:
            return splev(dl, temp, ext=3)
        else:
            return splev(dl, temp, ext=3)/dl**2

    def pz_xH0(self,z,temp):
        """
        Returns p(z|x,H0)
        """
        return temp(z)

    def px_H0G(self, H0):
        """
        Returns p(x|H0,G) for given values of H0.
        This corresponds to the numerator of Eq 12 in the method doc.
        The likelihood of the GW data given H0 and conditioned on
        the source being inside the galaxy catalog

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(x|H0,G)
        """
        num = np.zeros(len(H0))

        prob_sorted = np.sort(self.skymap.prob)[::-1]
        prob_sorted_cum = np.cumsum(prob_sorted)
        # find index of array which bounds the self.area confidence interval
        idx = np.searchsorted(prob_sorted_cum, self.area)
        minskypdf = prob_sorted[idx]*self.skymap.npix

        count = 0

        # TODO: expand this case to look at a skypatch
        # around the counterpart ('pencilbeam')
        if self.EM_counterpart is not None:
            nGalEM = self.EM_counterpart.nGal()
            for i in range(nGalEM):
                counterpart = self.EM_counterpart.get_galaxy(i)
                tempsky = self.skymap.skyprob(counterpart.ra, counterpart.dec)*self.skymap.npix
                tempdist = np.zeros(len(H0))
                for k in range(len(H0)):
                    tempdist[k] = self.norms[k]*self.pz_xH0(z,self.temps[k])
                numnorm += tempdist*tempsky

        else:
            #find galaxies within the bounds of the GW event
            tempsky = self.skymap.skyprob(self.allra, self.alldec)*self.skymap.npix
            ind = np.argwhere(tempsky >= minskypdf)
            tempsky = tempsky[ind].flatten()
            zs = self.allz[ind].flatten()
            ras = self.allra[ind].flatten()
            decs = self.alldec[ind].flatten()
            ms = self.allm[ind].flatten()
            sigzs = self.allsigmaz[ind].flatten()
            colors = self.allcolor[ind].flatten()

            if self.weighted:
                mlim = np.percentile(np.sort(ms),0.01) # more draws for galaxies in brightest 0.01 percent
            else:
                mlim = 1.0

            bar = progressbar.ProgressBar()
            print("Calculating p(x|H0,G)")
            # loop over galaxies
            for i in bar(range(len(zs))):
                if ms[i] <= mlim: #do more loops over brightest galaxies
                    nsmear = self.nsmear_fine
                else:
                    nsmear = self.nsmear_coarse
                numinner=np.zeros(len(H0))
                a = (0.0 - zs[i]) / sigzs[i]
                zsmear = truncnorm.rvs(a, 5, loc=zs[i], scale=sigzs[i], size=nsmear)
                zsmear = zsmear[np.argwhere(zsmear<self.zcut)].flatten() # remove support above the catalogue hard redshift cut
                tempdist = np.zeros([len(H0),len(zsmear)])
                if len(zsmear)>0:
                    for k in range(len(H0)):
                        tempdist[k,:] = self.norms[k]*self.pz_xH0(zsmear,self.temps[k])*self.ps_z(zsmear)
                    for n in range(len(zsmear)):
                        if self.weighted:
                            if self.Kcorr == True:
                                Kcorr = calc_kcor(self.band,zsmear[n],self.color_name,colour_value=colors[i])
                            else:
                                Kcorr = 0.
                            weight = L_mdl(ms[i], self.cosmo.dl_zH0(zsmear[n], H0), Kcorr=Kcorr)
                        else:
                            weight = 1.0
                        numinner += tempdist[:,n]*tempsky[i]*weight

                normnuminner = numinner/nsmear
                num += normnuminner

            print("{} galaxies from this catalog lie in the event's {}% confidence interval".format(len(zs),self.area*100))
            numnorm = num/self.nGal

        return numnorm


    def pD_H0G(self,H0):
        """
        Returns p(D|H0,G) (the normalising factor for px_H0G).
        This corresponds to the denominator of Eq 12 in the methods doc.
        The probability of detection as a function of H0, conditioned on the source being inside the galaxy catalog

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(D|H0,G)
        """
        den = np.zeros(len(H0))

        if self.weighted:
            mlim = np.percentile(np.sort(self.allm),0.01) # more draws for galaxies in brightest 0.01 percent
        else:
            mlim = 1.0

        bar = progressbar.ProgressBar()
        print("Calculating p(D|H0,G)")
        # loop over galaxies
        for i in bar(range(len(self.allz))):
            if self.allm[i] <= mlim: #do more loops over brightest galaxies
                nsmear = self.nsmear_fine
            else:
                nsmear = self.nsmear_coarse
            deninner=np.zeros(len(H0))
            a = (0.0 - self.allz[i]) / self.allsigmaz[i]
            zsmear = truncnorm.rvs(a, 5, loc=self.allz[i], scale=self.allsigmaz[i], size=nsmear)
            zsmear = zsmear[np.argwhere(zsmear<self.zcut)] # remove support above the catalogue hard redshift cut
            if len(zsmear)>0:
            # loop over random draws from galaxies
                for n in range(len(zsmear)):
                    if self.weighted:
                        if self.Kcorr == True:
                            Kcorr = calc_kcor(self.band,zsmear[n],self.color_name,colour_value=self.allcolor[i])
                        else:
                            Kcorr = 0.
                        weight = L_mdl(self.allm[i], self.cosmo.dl_zH0(zsmear[n], H0), Kcorr=Kcorr)
                    else:
                        weight = 1.0
                    if self.basic:
                        prob = self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(zsmear[n],H0)).flatten()
                    else:
                        prob = self.pdet.pD_zH0_eval(zsmear[n],H0).flatten()
                    deninner += prob*weight*self.ps_z(zsmear[n])
            normdeninner = deninner/nsmear
            den += normdeninner

        self.pDG = den/self.nGal

        return self.pDG


    def pG_H0D(self,H0):
        """
        Returns p(G|H0,D)
        This corresponds to Eq 16 in the doc.
        The probability that the host galaxy is in the catalogue given detection and H0.

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(G|H0,D)
        """
        # Warning - this integral misbehaves for small values of H0 (<25 kms-1Mpc-1).  TODO: fix this.
        num = np.zeros(len(H0))
        den = np.zeros(len(H0))

        # TODO: vectorize this if possible
        bar = progressbar.ProgressBar()
        print("Calculating p(G|H0,D)")
        for i in bar(range(len(H0))):
            def I(M,z):
                if self.basic:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha_sp)(M)*self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)*self.ps_z(z)
                else:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha_sp)(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)*self.ps_z(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            # Mmin and Mmax currently corresponding to 10L* and 0.001L* respectively, to correspond with MDC
            # Will want to change in future.
            # TODO: test how sensitive this result is to changing Mmin and Mmax.
            Mmin = M_Mobs(H0[i],self.Mobs_min)
            Mmax = M_Mobs(H0[i],self.Mobs_max)

            num[i] = dblquad(I,0,self.zcut,lambda x: Mmin,lambda x: min(max(M_mdl(self.mth,self.cosmo.dl_zH0(x,H0[i])),Mmin),Mmax),epsabs=0,epsrel=1.49e-4)[0]
            den[i] = dblquad(I,0,self.zmax,lambda x: Mmin,lambda x: Mmax,epsabs=0,epsrel=1.49e-4)[0]

        self.pGD = num/den
        return self.pGD


    def pnG_H0D(self,H0):
        """
        Returns 1.0 - pG_H0D(H0).
        This corresponds to Eq 17 in the doc.
        The probability that a galaxy is not in the catalogue given detection and H0

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(bar{G}|H0,D)
        """
        if all(self.pGD)==None:
            self.pGD = self.pG_H0D(H0)
        self.pnGD = 1.0 - self.pGD
        return self.pnGD


    def px_H0nG(self,H0,allsky=True):
        """
        Returns p(x|H0,bar{G}).
        This corresponds to the numerator of Eq 19 in the doc
        The likelihood of the GW data given H0, conditioned on the source being outside the galaxy catalog for an
        all sky or patchy galaxy catalog.
        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(x|H0,bar{G})
        """
        distnum = np.zeros(len(H0))

        bar = progressbar.ProgressBar()
        print("Calculating p(x|H0,bar{G})")
        for i in bar(range(len(H0))):

            def Inum(M,z):
                temp = self.norms[i]*self.pz_xH0(z,self.temps[i])*self.zprior(z) \
            *SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha_sp)(M)*self.ps_z(z)

                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(H0[i],self.Mobs_min)
            Mmax = M_Mobs(H0[i],self.Mobs_max)
            if allsky == True:
                distnum[i] = dblquad(Inum,0.0,self.zcut, lambda x: min(max(M_mdl(self.mth,self.cosmo.dl_zH0(x,H0[i])),Mmin),Mmax), lambda x: Mmax,epsabs=0,epsrel=1.49e-4)[0] \
                            + dblquad(Inum,self.zcut,self.zmax_GW[i], lambda x: Mmin, lambda x: Mmax,epsabs=0,epsrel=1.49e-4)[0]
            else:
                distnum[i] = dblquad(Inum,0.0,self.zmax_GW[i],lambda x: Mmin,lambda x: Mmax,epsabs=0,epsrel=1.49e-4)[0]

        # TODO: expand this case to look at a skypatch around the counterpart ('pencilbeam')
        if self.EM_counterpart != None:
            nGalEM = self.EM_counterpart.nGal()
            for i in range(nGalEM):
                counterpart = self.EM_counterpart.get_galaxy(i)
                tempsky = self.skymap.skyprob(counterpart.ra,counterpart.dec)*self.skymap.npix
                num += distnum*tempsky

        else:
            pixind = range(self.skymap.npix)
            theta,rapix = hp.pix2ang(self.skymap.nside,pixind,nest=True)
            decpix = np.pi/2.0 - theta
            idx = (self.ra_min <= rapix) & (rapix <= self.ra_max) & (self.dec_min <= decpix) & (decpix <= self.dec_max)
            if allsky == True:
                skynum = self.skymap.prob[idx].sum()
            else:
                skynum = 1.0 - self.skymap.prob[idx].sum()
            print("{}% of the event's sky probability is contained within the patch covered by the catalog".format(skynum*100))
            num = distnum*skynum
        return num


    def pD_H0nG(self,H0,allsky=True):
        """
        Returns p(D|H0,bar{G})
        This corresponds to the denominator of Eq 19 in the doc.
        The probability of detection as a function of H0, conditioned on the source being outside the galaxy catalog for an
        all sky or patchy galaxy catalog.

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(D|H0,bar{G})
        """
        # TODO: same fixes as for pG_H0D
        den = np.zeros(len(H0))

        def skynorm(dec,ra):
            return np.cos(dec)

        norm = dblquad(skynorm,self.ra_min,self.ra_max,lambda x: self.dec_min,lambda x: self.dec_max,epsabs=0,epsrel=1.49e-4)[0]/(4.*np.pi)

        bar = progressbar.ProgressBar()
        print("Calculating p(D|H0,bar{G})")
        for i in bar(range(len(H0))):

            def I(M,z):
                if self.basic:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha_sp)(M)*self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)*self.ps_z(z)
                else:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha_sp)(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)*self.ps_z(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(H0[i],self.Mobs_min)
            Mmax = M_Mobs(H0[i],self.Mobs_max)
            if allsky == True:
                den[i] = dblquad(I,0.0,self.zcut, lambda x: min(max(M_mdl(self.mth,self.cosmo.dl_zH0(x,H0[i])),Mmin),Mmax), lambda x: Mmax,epsabs=0,epsrel=1.49e-4)[0] \
                        + dblquad(I,self.zcut,self.zmax, lambda x: Mmin, lambda x: Mmax,epsabs=0,epsrel=1.49e-4)[0]
            else:
                den[i] = dblquad(I,0.0,self.zmax,lambda x: Mmin,lambda x: Mmax,epsabs=0,epsrel=1.49e-4)[0]
        if allsky == True:
            pDnG = den*norm
        else:
            pDnG = den*(1.-norm)

        return pDnG

    def px_H0_counterpart(self,H0):
        """
        Returns p(x|H0,counterpart)
        This corresponds to the numerator or Eq 6 in the doc.
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
        z = self.EM_counterpart.z
        sigma = self.EM_counterpart.sigmaz
        a = (0.0 - z) / sigma # boundary so samples don't go below 0
        zsmear = truncnorm.rvs(a, 5, loc=z, scale=sigma, size=10000)

        num = np.zeros(len(H0))
        for k in range(len(H0)):
            num[k] = np.sum(self.norms[k]*self.pz_xH0(zsmear,self.temps[k]))

        return num


    def pD_H0(self,H0):
        """
        Returns p(D|H0).
        This corresponds to the denominator of Eq 6 in the doc.
        The probability of detection as a function of H0, marginalised over redshift, and absolute magnitude

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(D|H0)
        """
        den = np.zeros(len(H0))

        bar = progressbar.ProgressBar()
        print("Calculating p(D|H0)")
        for i in bar(range(len(H0))):

            def I(z,M):
                if self.basic:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha_sp)(M)*self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)
                else:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha_sp)(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(H0[i],self.Mobs_min)
            Mmax = M_Mobs(H0[i],self.Mobs_max)

            den[i] = dblquad(I,Mmin,Mmax,lambda x: 0.0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        self.pDnG = den
        return self.pDnG


    def px_H0_empty(self,H0):
        """
        Returns the numerator of the empty catalog case
        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(x|H0,bar{G})
        """
        distnum = np.zeros(len(H0))
        for i in range(len(H0)):
            def Inum(z):
                temp = self.norms[i]*self.pz_xH0(z,self.temps[i])*self.zprior(z)*self.ps_z(z)
                return temp
            distnum[i] = quad(Inum,0.0,self.zmax_GW[i],epsabs=0,epsrel=1.49e-4)[0]
        skynum = 1.
        num = distnum*skynum
        return num


    def pD_H0_empty(self,H0):
        """
        Returns the denominator of the empty catalog case

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(D|H0,bar{G})
        """
        den = np.zeros(len(H0))
        for i in range(len(H0)):
            def I(z):
                temp = self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)*self.ps_z(z)
                return temp
            den[i] = quad(I,0.0,self.zmax,epsabs=0,epsrel=1.49e-4)[0]
        return den


    def likelihood(self,H0,complete=False,counterpart_case='direct',new_skypatch=False,population=False):
        """
        The likelihood for a single event
        This corresponds to Eq 3 (statistical) or Eq 6 (counterpart) in the doc, depending on parameter choices.

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1
        complete : bool, optional
            Is the galaxy catalog complete to all relevant distances/redshifts? (default=False)
        counterpart_case : str, optional
            Choice of counterpart analysis (default='direct')
            if 'direct', will assume the counterpart is correct with certainty
            if 'pencilbeam', will assume the host galaxy is along the counterpart's line of sight, but may be beyond it

        Returns
        -------
        float or array_like
            p(x|H0,D)
        """
        if self.EM_counterpart != None:

            if counterpart_case == 'direct':
                pxG = self.px_H0_counterpart(H0)
                self.pDG = self.pD_H0(H0)
                likelihood = pxG/self.pDG # Eq 6

            # The pencilbeam case is currently coded up along the line of sight of the counterpart
            # For GW170817 the likelihood produced is identical to the 'direct' counterpart case
            # TODO: allow this to cover a small patch of sky
            elif counterpart_case == 'pencilbeam':
                pxG = self.px_H0G(H0)
                if all(self.pDG)==None:
                    self.pDG = self.pD_H0G(H0)
                if all(self.pGD)==None:
                    self.pGD = self.pG_H0D(H0)
                if all(self.pnGD)==None:
                    self.pnGD = self.pnG_H0D(H0)
                if all(self.pDnG)==None:
                    self.pDnG = self.pD_H0nG(H0)
                pxnG = self.px_H0nG(H0)

                likelihood = self.pGD*(pxG/self.pDG) + self.pnGD*(pxnG/self.pDnG) # Eq 3 along a single line of sight
            else:
                print("Please specify counterpart_case ('direct' or 'pencilbeam').")

        elif new_skypatch==True:
            likelihood,pxG,self.pDG,self.pGD,self.pnGD,pxnG,self.pDnG = self.likelihood_skypatch(H0,complete=complete)

        elif population==True:
            pxG = self.px_H0_empty(H0)
            self.pDG = self.pD_H0_empty(H0)
            likelihood = pxG/self.pDG

        else:
            pxG = self.px_H0G(H0)
            if all(self.pDG)==None:
                self.pDG = self.pD_H0G(H0)

            if complete==True:
                likelihood = pxG/self.pDG # Eq 3 with p(G|H0,D)=1 and p(bar{G}|H0,D)=0

            else:
                if all(self.pGD)==None:
                    self.pGD = self.pG_H0D(H0)
                if all(self.pnGD)==None:
                    self.pnGD = self.pnG_H0D(H0)
                if all(self.pDnG)==None:
                    self.pDnG = self.pD_H0nG(H0)

                pxnG = self.px_H0nG(H0)

                likelihood = self.pGD*(pxG/self.pDG) + self.pnGD*(pxnG/self.pDnG) # Eq 3


            if self.whole_cat == False:
                pDnG_rest_of_sky = self.pD_H0nG(H0,allsky=False)
                pxnG_rest_of_sky = self.px_H0nG(H0,allsky=False)

                likelihood = likelihood*self.catalog_fraction + (pxnG_rest_of_sky/pDnG_rest_of_sky)*self.rest_fraction # Eq 4


        if (complete==True) or (self.EM_counterpart != None) or (population==True):
            self.pGD = np.ones(len(H0))
            self.pnGD = np.zeros(len(H0))
            pxnG = np.zeros(len(H0))
            self.pDnG = np.ones(len(H0))

        if (self.whole_cat==True) or (self.EM_counterpart != None) or (population==True):
            pDnG_rest_of_sky = np.ones(len(H0))
            pxnG_rest_of_sky = np.zeros(len(H0))
            self.rest_fraction = 0
            self.catalog_fraction = 1


        return likelihood,pxG,self.pDG,self.pGD,self.catalog_fraction, pxnG,self.pDnG,self.pnGD, pxnG_rest_of_sky,pDnG_rest_of_sky,self.rest_fraction


    def px_DGH0_skypatch(self,H0):
        """
        The "in catalog" part of the new skypatch method
        using a catalog which follows the GW event's sky patch contour
        p(x|D,G,H0)

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        arrays
            numerator and denominator
        """
        num = np.zeros(len(H0))
        den = np.zeros(len(H0))
        print('whole catalog apparent magnitude threshold: {}'.format(self.mth))

        tempsky = self.skymap.skyprob(self.allra, self.alldec)*self.skymap.npix
        ind = np.argwhere(tempsky > 0.)
        tempsky = tempsky[ind].flatten()
        zs = self.allz[ind].flatten()
        ras = self.allra[ind].flatten()
        decs = self.alldec[ind].flatten()
        ms = self.allm[ind].flatten()
        sigzs = self.allsigmaz[ind].flatten()
        colors = self.allcolor[ind].flatten()

        max_mth = np.amax(ms)
        N = len(zs)

        if self.weighted:
            mlim = np.percentile(np.sort(ms),0.01) # more draws for galaxies in brightest 0.01 percent
        else:
            mlim = 1.0

        bar = progressbar.ProgressBar()
        print("Calculating p(x|D,H0,G) for this event's skyarea")
        # loop over galaxies
        for i in bar(range(N)):
            numinner=np.zeros(len(H0))
            deninner=np.zeros(len(H0))

            if ms[i] <= mlim: #do more loops over brightest galaxies
                nsmear = self.nsmear_fine
            else:
                nsmear = self.nsmear_coarse

            a = (0.0 - zs[i]) / sigzs[i]
            zsmear = truncnorm.rvs(a, 5, loc=zs[i], scale=sigzs[i], size=nsmear)
            zsmear = zsmear[np.argwhere(zsmear<self.zcut)].flatten() # remove support above the catalogue hard redshift cut
            tempdist = np.zeros([len(H0),len(zsmear)])
            if len(zsmear)>0:
                for k in range(len(H0)):
                    tempdist[k,:] = self.norms[k]*self.pz_xH0(zsmear,self.temps[k])*self.ps_z(zsmear)
            # loop over random draws from galaxies
                for n in range(len(zsmear)):
                    if self.weighted:
                        if self.Kcorr == True:
                            Kcorr = calc_kcor(self.band,zsmear[n],self.color_name,colour_value=colors[i])
                        else:
                            Kcorr = 0.
                        weight = L_mdl(ms[i], self.cosmo.dl_zH0(zsmear[n], H0), Kcorr=Kcorr)
                    else:
                        weight = 1.0

                    numinner += tempdist[:,n]*tempsky[i]*weight

                    if self.basic:
                        prob = self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(zsmear[n],H0)).flatten()
                    else:
                        prob = self.pdet.pD_zH0_eval(zsmear[n],H0).flatten()
                    deninner += prob*weight*self.ps_z(zsmear[n])
            normnuminner = numinner/nsmear
            num += normnuminner
            normdeninner = deninner/nsmear
            den += normdeninner
        print("{} galaxies from this catalog lie in the event's {}% confidence interval".format(len(zs),self.area*100))
        numnorm = num/self.nGal

        if N >= 500:
            self.mth = np.median(ms)
        else:
            self.mth = max_mth #update mth to reflect the area within the event's sky localisation (max m within patch)

        print('event patch apparent magnitude threshold: {}'.format(self.mth))
        print("{} galaxies (out of a total possible {}) are supported by this event's skymap".format(N,self.nGal))
        return num,den

    def px_DnGH0_skypatch(self,H0):
        """
        The "beyond catalog" part of the new skypatch method
        using a catalog which follows the GW event's sky patch contour
        p(x|D,Gbar,H0)

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        arrays
            numerator and denominator
        """
        distnum = np.zeros(len(H0))
        distden = np.zeros(len(H0))

        bar = progressbar.ProgressBar()
        print("Calculating p(x|D,H0,bar{G}) for this event's skyarea")
        for i in bar(range(len(H0))):

            Mmin = M_Mobs(H0[i],self.Mobs_min)
            Mmax = M_Mobs(H0[i],self.Mobs_max)

            def Inum(z,M):
                temp = self.norms[i]*self.pz_xH0(z,self.temps[i])*self.zprior(z) \
            *SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha_sp)(M)*self.ps_z(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp
            distnum[i] = dblquad(Inum,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear),lambda x: self.zmax_GW[i],epsabs=0,epsrel=1.49e-4)[0]

            def Iden(z,M):
                if self.basic:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha_sp)(M)*self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)*self.ps_z(z)
                else:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha_sp)(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)*self.ps_z(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp
            distden[i] = dblquad(Iden,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear),lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        skynum = 1.0
        num = distnum*skynum
        a = len(np.asarray(np.where(self.skymap.prob!=0)).flatten()) # find number of pixels with any GW event support
        skyden = a/self.skymap.npix
        den = distden*skyden

        return num,den


    def likelihood_skypatch(self,H0,complete=False):
        """
        The event likelihood using the new skypatch method
        p(x|D,H0)

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        array
            the unnormalised likelihood
        """
        pxDG_num,pxDG_den = self.px_DGH0_skypatch(H0)


        if complete==True:
            likelihood = pxDG_num/pxDG_den
            pGD = np.ones(len(H0))
            pnGD = np.zeros(len(H0))
            pxDnG_num = np.zeros(len(H0))
            pxDnG_den = np.ones(len(H0))

        else:
            pGD = self.pG_H0D(H0)
            pnGD = self.pnG_H0D(H0)

            pxDnG_num,pxDnG_den = self.px_DnGH0_skypatch(H0)
            pxDnG = pxDnG_num/pxDnG_den

            likelihood = pGD*(pxDG_num/pxDG_den) + pnGD*pxDnG

        return likelihood,pxDG_num,pxDG_den,pGD,pnGD,pxDnG_num,pxDnG_den
