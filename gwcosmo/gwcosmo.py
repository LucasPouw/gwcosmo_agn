"""
gwcosmoLikelihood Module
Rachel Gray, Archisman Ghosh, Ignacio Magana, John Veitch, Ankan Sur

In general:
p(x|z,H0,\Omega) is written as p(x|dl(z,H0))*p(x|\Omega)
p(x|dL(z,H0)): self.px_dl(dl(z,H0))
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
from scipy.stats import ncx2, norm
from scipy.interpolate import splev, splrep
from astropy import constants as const
from astropy import units as u
from ligo.skymap.moc import rasterize
from ligo.skymap.core import uniq2ang

import gwcosmo

from .utilities.standard_cosmology import *
from .utilities.schechter_function import *

import time
import progressbar


class gwcosmoLikelihood(object):
    """
    A class to hold all the individual components of the posterior for H0,
    and methods to stitch them together in the right way.

    Parameters
    ----------
    event_type : str
        Type of gravitational wave event (either 'BNS', 'BNS-uniform' or 'BBH')
    GW_data : gwcosmo.likelihood.posterior_samples.posterior_samples object 
        Gravitational wave event samples
    skymap : gwcosmo.likelihood.skymap.skymap object
        Gravitational wave event skymap
    galaxy_catalog : gwcosmo.prior.catalog.galaxyCatalog object
        The relevant galaxy catalog
    psd : str, optional
        Select between 'O1' and 'O2' PSDs, by default we use aLIGO at
        design sensitivity (default=None).
    EM_counterpart : gwcosmo.prior.catalog.galaxyCatalog object, optional
        EM_counterpart data (default=None)
        If not None, will default to using this over the galaxy_catalog 
    Omega_m : float, optional
        The matter fraction of the universe (default=0.3)
    linear : bool, optional
        Use linear cosmology (default=False)
    weighted : bool, optional
        Use luminosity weighting (default=False)
    weights : str, optional
        Specifies type of luminosity weighting to use: 'schechter' or 'trivial'
        (default='schechter') 'trivial' is only for testing purposes and
        should not be used in analysis
    whole_cat : bool, optional
        Does the galaxy catalog provided cover the whole sky? (default=True)
    radec_lim : array_like, optional
        Needed if whole_cat=False. RA and Dec limits on the catalog in the
        format np.array([ramin,ramax,decmin,decmax]) in radians
    basic : bool, optional
        If True, uses pdet suitable for MDC analysis (default=False)
    uncertainty : bool, optional
        If true, redshift uncertainty will be assumed and corrected
        for (default=False)
    rate : str, optional
        specifies rate evolution model, 'const' or 'evolving'
    """

    def __init__(self, GW_data, skymap, galaxy_catalog, pdet, EM_counterpart=None,
                 Omega_m=0.308, linear=False, weighted=False, whole_cat=True, radec_lim=None,
                 basic=False, uncertainty=False, rate='constant', Lambda=3.0, area=0.999,
                 Mstar_obs=-20.457,alpha=-1.07,Mmin_obs=-22.96,Mmax_obs=-12.96, mth=None):
        self.pdet = pdet
        self.event_type = pdet.mass_distribution
        self.psd = pdet.psd
        self.Omega_m = Omega_m
        self.linear = linear
        self.weighted = weighted
        self.whole_cat = whole_cat
        self.radec_lim = radec_lim
        self.basic = basic
        self.uncertainty = uncertainty
        self.skymap = skymap
        self.area = area
        self.Mstar_obs = Mstar_obs
        self.alpha = alpha
        self.Mmin_obs = Mmin_obs
        self.Mmax_obs = Mmax_obs

        if self.uncertainty == False:
            self.galaxy_catalog = galaxy_catalog
            self.EM_counterpart = None
            if EM_counterpart is not None:
                self.EM_counterpart = EM_counterpart
        else:
            if galaxy_catalog is not None:
                self.galaxy_catalog = galaxy_catalog.redshiftUncertainty()
                self.mth = galaxy_catalog.mth()

            self.EM_counterpart = None
            if EM_counterpart is not None:
                self.EM_counterpart = EM_counterpart.redshiftUncertainty(peculiarVelocityCorr=True)
        
        if GW_data is not None:
            distkernel = GW_data.lineofsight_distance()
            distmin = 0.5*np.amin(GW_data.distance)
            distmax = 2.0*np.amax(GW_data.distance)
            dl_array = np.linspace(distmin, distmax, 500)
            vals = distkernel(dl_array)
            
        if (GW_data is None and self.EM_counterpart is None):
            dl_array, vals = self.skymap.marginalized_distance()
            
        if (GW_data is None and self.EM_counterpart is not None):
            counterpart = self.EM_counterpart.get_galaxy(0)
            dl_array, vals = self.skymap.lineofsight_distance(counterpart.ra, counterpart.dec)

        self.temp = splrep(dl_array,vals)
        # TODO: calculate mth for the patch of catalog being used, if whole_cat=False
        if self.whole_cat == False:
            if all(radec_lim) == None:
                print('must include ra and dec limits for a catalog which only covers part of the sky')
            else:
                self.ra_min = radec_lim[0]
                self.ra_max = radec_lim[1]
                self.dec_min = radec_lim[2]
                self.dec_max = radec_lim[3]
        else:
            self.ra_min = 0.0
            self.ra_max = np.pi*2.0
            self.dec_min = -np.pi/2.0
            self.dec_max = np.pi/2.0        
        self.pDG = None
        self.pGD = None
        self.pnGD = None
        self.pDnG = None

        # Note that zmax is an artificial limit that
        # should be well above any redshift value that could
        # impact the results for the considered H0 values.
        if self.event_type == 'BNS-gaussian':
            self.zmax = 0.5
        elif self.event_type == 'BNS-uniform':
            self.zmax = 0.5
        elif self.event_type == 'BBH-powerlaw':
            self.zmax = 4.0
        
        self.zprior = redshift_prior(Omega_m=self.Omega_m, linear=self.linear)
        self.cosmo = fast_cosmology(Omega_m=self.Omega_m, linear=self.linear)
        self.rate = rate
        self.Lambda = Lambda

    def ps_z(self, z):
        if self.rate == 'constant':
            return 1.0
        if self.rate == 'evolving':
            return (1.0+z)**self.Lambda

    def px_dl(self, dl):
        """
        Returns a probability for a given distance dl
        from the interpolated function.
        """
        return splev(dl, self.temp, ext=3)

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
        nGal = self.galaxy_catalog.nGal()
        num = np.zeros(len(H0))

        prob_sorted = np.sort(self.skymap.prob)[::-1]
        prob_sorted_cum = np.cumsum(prob_sorted)
        # find index of array which bounds the self.area confidence interval
        idx = np.searchsorted(prob_sorted_cum, self.area)
        minskypdf = prob_sorted[idx]*self.skymap.npix

        count = 0
        nGal_patch = 0

        # TODO: expand this case to look at a skypatch
        # around the counterpart ('pencilbeam')
        if self.EM_counterpart is not None:
            nGalEM = self.EM_counterpart.nGal()
            for i in range(nGalEM):
                counterpart = self.EM_counterpart.get_galaxy(i)
                tempsky = self.skymap.skyprob(counterpart.ra, counterpart.dec)*self.skymap.npix
                tempdist = self.px_dl(self.cosmo.dl_zH0(counterpart.z, H0))/self.cosmo.dl_zH0(counterpart.z, H0)**2 # remove dl^2 prior from samples
                numnorm += tempdist*tempsky
        else:
            galIndex = []
            # loop over all possible galaxies
            bar = progressbar.ProgressBar()
            print("Calculating p(x|H0,G)")
            for i in bar(range(nGal)):
                gal = self.galaxy_catalog.get_galaxy(i)
                if (self.ra_min <= gal.ra <= self.ra_max and self.dec_min <= gal.dec <= self.dec_max):
                    nGal_patch += 1.0
                    tempsky = self.skymap.skyprob(gal.ra, gal.dec)*self.skymap.npix
                    if tempsky >= minskypdf:
                        count += 1
                        galIndex.append(i)
                        if self.weighted:
                            weight = L_mdl(gal.m, self.cosmo.dl_zH0(gal.z, H0))
                        else:
                            weight = 1.0
                        if gal.z == 0:
                            tempdist = 0.0
                        else:
                            tempdist = self.px_dl(self.cosmo.dl_zH0(gal.z, H0))/self.cosmo.dl_zH0(gal.z, H0)**2 # remove dl^2 prior from samples
                        num += tempdist*tempsky*weight*self.ps_z(gal.z)
                    else:
                        continue
                else:
                    continue
            print("{} galaxies from this catalog lie in the event's 90% confidence interval".format(int(count)))

            if self.whole_cat == True:
                numnorm = num/nGal
            else:
                numnorm = num/nGal_patch
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
        nGal = self.galaxy_catalog.nGal()   
        nGal_patch = 0.0     

        den = np.zeros(len(H0))
        
        bar = progressbar.ProgressBar()
        print("Calculating p(D|H0,G)")
        for i in bar(range(nGal)):
            gal = self.galaxy_catalog.get_galaxy(i)
            if (self.ra_min <= gal.ra <= self.ra_max and self.dec_min <= gal.dec <= self.dec_max):
                nGal_patch += 1.0
                if self.weighted:
                    weight = L_mdl(gal.m,self.cosmo.dl_zH0(gal.z,H0))
                else:
                    weight = 1.0
                if self.basic:
                    prob = self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(gal.z,H0))
                else:
                    prob = self.pdet.pD_zH0_eval(gal.z,H0)
                den += np.reshape(prob,len(H0))*weight*self.ps_z(gal.z)
            else:
                continue

        if self.whole_cat == True:
            self.pDG = den/nGal
        else:
            self.pDG = den/nGal_patch
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
            def I(z,M):
                if self.basic:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)*self.ps_z(z)
                else:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)*self.ps_z(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp
            
            # Mmin and Mmax currently corresponding to 10L* and 0.001L* respectively, to correspond with MDC
            # Will want to change in future.
            # TODO: test how sensitive this result is to changing Mmin and Mmax.
            Mmin = M_Mobs(H0[i],self.Mmin_obs)
            Mmax = M_Mobs(H0[i],self.Mmax_obs)
            
            num[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear),epsabs=0,epsrel=1.49e-4)[0]
            den[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

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

            def Inum(z,M):
                temp = px_dl(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z) \
            *SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.ps_z(z)/self.cosmo.dl_zH0(z,H0[i])**2 # remove dl^2 prior from samples
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(H0[i],self.Mmin_obs)
            Mmax = M_Mobs(H0[i],self.Mmax_obs)
            if allsky == True:
                distnum[i] = dblquad(Inum,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear),lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]
            else:
                distnum[i] = dblquad(Inum,Mmin,Mmax,lambda x: 0.0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

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

            def I(z,M):
                if self.basic:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)*self.ps_z(z)
                else:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)*self.ps_z(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(H0[i],self.Mmin_obs)
            Mmax = M_Mobs(H0[i],self.Mmax_obs)
            if allsky == True:
                den[i] = dblquad(I,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear),lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]
                self.pDnG = den*norm
            else:
                den[i] = dblquad(I,Mmin,Mmax,lambda x: 0.0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]
                self.pDnG = den*(1.-norm)
                
        return self.pDnG

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
        numnorm = np.zeros(len(H0))
        nGalEM = self.EM_counterpart.nGal()
        for i in range(nGalEM):
            counterpart = self.EM_counterpart.get_galaxy(i)
            tempsky = self.skymap.skyprob(counterpart.ra,counterpart.dec)*self.skymap.npix
            tempdist = self.px_dl(self.cosmo.dl_zH0(counterpart.z,H0))/self.cosmo.dl_zH0(counterpart.z,H0)**2 # remove dl^2 prior from samples
            numnorm += tempdist*tempsky
        return numnorm


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
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)*self.ps_z(z)
                else:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)*self.ps_z(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(H0[i],self.Mmin_obs)
            Mmax = M_Mobs(H0[i],self.Mmax_obs)

            den[i] = dblquad(I,Mmin,Mmax,lambda x: 0.0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        self.pDnG = den   
        return self.pDnG

    def likelihood(self,H0,complete=False,counterpart_case='direct'):
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
                pD_H0 = self.pD_H0(H0)
                likelihood = pxG/pD_H0 # Eq 6
                
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

                likelihood = likelihood + (pxnG_rest_of_sky/pDnG_rest_of_sky) # Eq 4

        return likelihood
