"""
Master Equation
Rachel Gray, John Veitch, Ignacio Magana
"""
from __future__ import absolute_import

import lal
import numpy as np
import sys
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import healpy as hp

from scipy.integrate import quad, dblquad
from scipy.stats import ncx2, norm
from scipy.interpolate import splev,splrep
from astropy import constants as const
from astropy import units as u

from ligo.skymap.moc import nest2uniq,uniq2nest,uniq2ang
from ligo.skymap.bayestar import rasterize

import gwcosmo

from .utilities.standard_cosmology import *
from .utilities.schechter_function import *

class MasterEquation(object):
    """
    A class to hold all the individual components of the posterior for H0,
    and methods to stitch them together in the right way.
    
    Parameters
    ----------
    event_type : str
        Type of gravitational wave event (either 'BNS' or 'BBH')
    galaxy_catalog : gwcosmo.prior.catalog.galaxyCatalog object
        The relevant galaxy catalog
    Omega_m : float, optional
        The matter fraction of the universe (default=0.3)
    linear : bool, optional
        Use linear cosmology (default=False)
    weighted : bool, optional
        Use luminosity weighting (default=False)
    whole_cat : bool, optional
        Does the galaxy catalog provided cover the whole sky? (default=True)
    radec_lim : array_like, optional
        Needed if whole_cat=False. RA and Dec limits on the catalog in the format np.array([ramin,ramax,decmin,decmax]) in radians
    """

    def __init__(self,event_type,galaxy_catalog,Omega_m=0.3,linear=False,weighted=False,whole_cat=True,radec_lim=None,basic=False):
        self.event_type = event_type
        self.galaxy_catalog = galaxy_catalog
        self.pdet = gwcosmo.detection_probability.DetectionProbability(self.event_type,Nsamps=5000)
        self.mth = galaxy_catalog.mth() # TODO: calculate mth for the patch of catalog being used, if whole_cat=False
        self.Omega_m = Omega_m
        self.linear = linear
        self.weighted = weighted
        self.whole_cat = whole_cat
        self.radec_lim = radec_lim
        self.basic = basic
        if self.whole_cat == False:
            if all(radec_lim)==None:
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
        self.pDnG_rest_of_sky = None
        
        # Note that zmax is an artificial limit that should be well above any redshift value that could impact the results for the considered H0 values.
        if event_type == 'BNS':
            self.zmax = 0.4
        elif event_type == 'BBH':
            self.zmax = 4.0
        self.zprior = redshift_prior(Omega_m=self.Omega_m,linear=self.linear)
        self.cosmo = fast_cosmology(Omega_m=self.Omega_m,linear=self.linear)

    def px_H0G(self,H0,GW_data,EM_counterpart=None,skymap2d=None):
        """
        Returns p(x|H0,G) for given values of H0.
        The likelihood of the GW data given H0 and conditioned on the source being inside the galaxy catalog
        
        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1
        GW_data : gwcosmo.likelihood.posterior_samples.posterior_samples object
            Gravitational wave event samples
        EM_counterpart : gwcosmo.prior.catalog.galaxyCatalog object, optional
            EM_counterpart data (default=None)
            If not None, will default to using this over the galaxy_catalog 
        skymap2d : gwcosmo.likelihood.skymap.skymap object, optional
            Gravitational wave event skymap (default=None)
            If not None, will default to using this over the GW_data
            
        Returns
        -------
        float or array_like
            p(x|H0,G)
        """
        nGal = self.galaxy_catalog.nGal()
        num = np.zeros(len(H0))
        
        if skymap2d is not None:
            prob_sorted = np.sort(skymap2d.prob)[::-1]
            prob_sorted_cum = np.cumsum(prob_sorted)
            idx = np.searchsorted(prob_sorted_cum,0.999) # find index of array which bounds the 99.9% confidence interval
            minskypdf = prob_sorted[idx]*skymap2d.npix
        
        else:    
            skykernel = GW_data.compute_2d_kde()
            skypdf = skykernel.evaluate([GW_data.longitude,GW_data.latitude])
            skypdf.sort()
            sampno = int(0.001*np.size(skypdf)) # find the position of the sample in the list which bounds the 99.9% confidence interval
            minskypdf = skypdf[sampno]

        count = 0
        nGal_patch = 0
        
        distkernel = GW_data.lineofsight_distance()
        distmax = 2.0*np.amax(GW_data.distance)
        distmin = 0.5*np.amin(GW_data.distance)
        dl_array = np.linspace(distmin,distmax,500)
        vals = distkernel(dl_array)
        temp = splrep(dl_array,vals)
        
        def px_dl(dl):
            """
            Returns a probability for a given distance dl from the interpolated function.
            """
            return splev(dl,temp,ext=3)
        
        # TODO: expand this case to look at a skypatch around the counterpart ('pencilbeam')
        if EM_counterpart != None:
            counterpart = EM_counterpart.get_galaxy(0)
            
            if skymap2d is not None:
                tempsky = skymap2d.skyprob(counterpart.ra,counterpart.dec)*skymap2d.npix
            else:
                tempsky = skykernel.evaluate([counterpart.ra,counterpart.dec])*4.0*np.pi/np.cos(counterpart.dec) # remove uniform sky prior from samples
                
            tempdist = px_dl(self.cosmo.dl_zH0(counterpart.z,H0))/self.cosmo.dl_zH0(counterpart.z,H0)**2 # remove dl^2 prior from samples
            numnorm = tempdist*tempsky

        else:
            # loop over all possible galaxies
            for i in range(nGal):
                gal = self.galaxy_catalog.get_galaxy(i)
                if (self.ra_min <= gal.ra <= self.ra_max and self.dec_min <= gal.dec <= self.dec_max):
                    nGal_patch += 1.0
                    if skymap2d is not None:
                        tempsky = skymap2d.skyprob(gal.ra,gal.dec)*skymap2d.npix
                    else:
                        tempsky = skykernel.evaluate([gal.ra,gal.dec])*4.0*np.pi/np.cos(gal.dec) # remove uniform sky prior from samples
    
                    if tempsky >= minskypdf:
                        count += 1
    
                        if self.weighted:
                            weight = L_mdl(gal.m,self.cosmo.dl_zH0(gal.z,H0))
                        else:
                            weight = 1.0
                    
                        if gal.z == 0:
                            tempdist = 0.0
                        else:
                            tempdist = px_dl(self.cosmo.dl_zH0(gal.z,H0))/self.cosmo.dl_zH0(gal.z,H0)**2 # remove dl^2 prior from samples
                        num += tempdist*tempsky*weight
                    else:
                        continue
                else:
                    continue
            print("{} galaxies from this catalog lie in the event's 99.9% confidence interval".format(count))
            
            if self.whole_cat == True:
                numnorm = num/nGal
            else:
                numnorm = num/nGal_patch
        return numnorm


    def pD_H0G(self,H0):
        """
        Returns p(D|H0,G) (the normalising factor for px_H0G).
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
        for i in range(nGal):
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
                den += np.reshape(prob,len(H0))*weight
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
        for i in range(len(H0)):
            
            def I(z,M):
                if self.basic:
                    temp = SchechterMagFunction(H0=H0[i])(M)*self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)
                else:
                    temp = SchechterMagFunction(H0=H0[i])(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp
            
            # Mmin and Mmax currently corresponding to 10L* and 0.001L* respectively, to correspond with MDC
            # Will want to change in future.
            # TODO: test how sensitive this result is to changing Mmin and Mmax.
            Mmin = M_Mobs(H0[i],-22.96)
            Mmax = M_Mobs(H0[i],-12.96)
            
            num[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear),epsabs=0,epsrel=1.49e-4)[0]
            den[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        self.pGD = num/den
        return self.pGD    


    def pnG_H0D(self,H0):
        """
        Returns 1.0 - pG_H0D(H0).
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
       
        
    def px_H0nG(self,H0,GW_data,EM_counterpart=None,skymap2d=None):
        """
        Returns p(x|H0,bar{G}).
        The likelihood of the GW data given H0, conditioned on the source being outside the galaxy catalog.
        
        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1
        GW_data : gwcosmo.likelihood.posterior_samples.posterior_samples object
            Gravitational wave event samples
        EM_counterpart : gwcosmo.prior.catalog.galaxyCatalog object, optional
            EM_counterpart data (default=None)
            If not None, will default to using this over the galaxy_catalog 
        skymap2d : gwcosmo.likelihood.skymap.skymap object, optional
            Gravitational wave event skymap (default=None)
            If not None, will default to using this over the GW_data
            
        Returns
        -------
        float or array_like
            p(x|H0,bar{G})
        """
        distnum = np.zeros(len(H0))
        
        distkernel = GW_data.lineofsight_distance()
        
        distmax = 2.0*np.amax(GW_data.distance)
        distmin = 0.5*np.amin(GW_data.distance)
        dl_array = np.linspace(distmin,distmax,500)
        vals = distkernel(dl_array)
        temp = splrep(dl_array,vals)
        
        def px_dl(dl):
            """
            Returns a probability for a given distance dl from the interpolated function.
            """
            return splev(dl,temp,ext=3)

        for i in range(len(H0)):

            def Inum(z,M):
                temp = px_dl(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z) \
            *SchechterMagFunction(H0=H0[i])(M)/self.cosmo.dl_zH0(z,H0[i])**2 # remove dl^2 prior from samples
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(H0[i],-22.96)
            Mmax = M_Mobs(H0[i],-12.96)

            distnum[i] = dblquad(Inum,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear),lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]
        
        # TODO: expand this case to look at a skypatch around the counterpart ('pencilbeam')    
        if EM_counterpart != None:
            counterpart = EM_counterpart.get_galaxy(0)
            
            if skymap2d is not None:
                tempsky = skymap2d.skyprob(counterpart.ra,counterpart.dec)*skymap2d.npix
            else:
                tempsky = skykernel.evaluate([counterpart.ra,counterpart.dec])*4.0*np.pi/np.cos(counterpart.dec) # remove uniform sky prior from samples
                
            num = distnum*tempsky
        
        else:
            pixind = range(skymap2d.npix)
            theta,rapix = hp.pix2ang(skymap2d.nside,pixind,nest=True)
            decpix = np.pi/2.0 - theta
            idx = (self.ra_min <= rapix) & (rapix <= self.ra_max) & (self.dec_min <= decpix) & (decpix <= self.dec_max)
            skynum = skymap2d.prob[idx].sum()
            print("{}% of the event's sky probability is contained within the patch covered by the catalog".format(skynum*100))
            num = distnum*skynum
        
        return num


    def pD_H0nG(self,H0):
        """
        Returns p(D|H0,bar{G})
        The probability of detection as a function of H0, conditioned on the source being outside the galaxy catalog
        
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
        distden = np.zeros(len(H0))
        
        def skynorm(dec,ra):
            return np.cos(dec)
                
        norm = dblquad(skynorm,self.ra_min,self.ra_max,lambda x: self.dec_min,lambda x: self.dec_max,epsabs=0,epsrel=1.49e-4)[0]/(4.*np.pi)
        
        for i in range(len(H0)):

            def I(z,M):
                if self.basic:
                    temp = SchechterMagFunction(H0=H0[i])(M)*self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)
                else:
                    temp = SchechterMagFunction(H0=H0[i])(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(H0[i],-22.96)
            Mmax = M_Mobs(H0[i],-12.96)

            distden[i] = dblquad(I,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear),lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]
        
        den = distden*norm

        self.pDnG = den   
        return self.pDnG


    def px_H0nG_rest_of_sky(self,H0,GW_data,skymap2d=None):
        """
        FOR THE AREA OF SKY OUTSIDE THE RA AND DEC LIMITS 
        Returns p(x|H0,bar{G}).
        The likelihood of the GW data given H0, conditioned on the source being outside the galaxy catalog.
        
        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1
        GW_data : gwcosmo.likelihood.posterior_samples.posterior_samples object
            Gravitational wave event samples
        skymap2d : gwcosmo.likelihood.skymap.skymap object, optional
            Gravitational wave event skymap (default=None)
            If not None, will default to using this over the GW_data
            
        Returns
        -------
        float or array_like
            p(x|H0,bar{G})
        """
        distnum = np.zeros(len(H0))
        
        distkernel = GW_data.lineofsight_distance()
        
        distmax = 2.0*np.amax(GW_data.distance)
        distmin = 0.5*np.amin(GW_data.distance)
        dl_array = np.linspace(distmin,distmax,500)
        vals = distkernel(dl_array)
        temp = splrep(dl_array,vals)
        
        def px_dl(dl):
            """
            Returns a probability for a given distance dl from the interpolated function.
            """
            return splev(dl,temp,ext=3)
            
        pixind = range(skymap2d.npix)
        theta,rapix = hp.pix2ang(skymap2d.nside,pixind,nest=True)
        decpix = np.pi/2.0 - theta
        idx = (self.ra_min <= rapix) & (rapix <= self.ra_max) & (self.dec_min <= decpix) & (decpix <= self.dec_max)
        skynum = 1.0 - skymap2d.prob[idx].sum()
        print("{}% of the event's sky probability is not inside the patch covered by the catalog".format(skynum*100))

        for i in range(len(H0)):

            def Icat(z,M):
                temp = px_dl(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z) \
            *SchechterMagFunction(H0=H0[i])(M)/self.cosmo.dl_zH0(z,H0[i])**2 # remove dl^2 prior from samples
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(H0[i],-22.96)
            Mmax = M_Mobs(H0[i],-12.96)

            distnum[i] = dblquad(Icat,Mmin,Mmax,lambda x: 0.0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]
            
        num = distnum*skynum
        return num


    def pD_H0nG_rest_of_sky(self,H0):
        """
        FOR THE AREA OF SKY OUTSIDE THE RA AND DEC LIMITS
        Returns p(D|H0,bar{G})
        The probability of detection as a function of H0, conditioned on the source being outside the galaxy catalog
        
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
                
        norm = 1.0 - dblquad(skynorm,self.ra_min,self.ra_max,lambda x: self.dec_min,lambda x: self.dec_max,epsabs=0,epsrel=1.49e-4)[0]/(4.*np.pi)

        for i in range(len(H0)):

            def I(z,M):
                if self.basic:
                    temp = SchechterMagFunction(H0=H0[i])(M)*self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)
                else:
                    temp = SchechterMagFunction(H0=H0[i])(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(H0[i],-22.96)
            Mmax = M_Mobs(H0[i],-12.96)

            den[i] = dblquad(I,Mmin,Mmax,lambda x: 0.0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]
            
        self.pDnG_rest_of_sky = den*norm  
        return self.pDnG_rest_of_sky


    def px_H0_counterpart(self,H0,GW_data,EM_counterpart,skymap2d):
        """
        Returns p(x|H0,counterpart)
        The likelihood of the GW data given H0 and direct counterpart.
        
        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1
        GW_data : gwcosmo.likelihood.posterior_samples.posterior_samples object
            Gravitational wave event samples
        EM_counterpart : gwcosmo.prior.catalog.galaxyCatalog object, optional
            EM_counterpart data (default=None)
            If not None, will default to using this over the galaxy_catalog 
        skymap2d : gwcosmo.likelihood.skymap.skymap object, optional
            Gravitational wave event skymap (default=None)
            If not None, will default to using this over the GW_data
            
        Returns
        -------
        float or array_like
            p(x|H0,counterpart)
        """
        num = np.zeros(len(H0))
        
        distkernel = GW_data.lineofsight_distance()
        distmax = 2.0*np.amax(GW_data.distance)
        distmin = 0.5*np.amin(GW_data.distance)
        dl_array = np.linspace(distmin,distmax,500)
        vals = distkernel(dl_array)
        temp = splrep(dl_array,vals)
        
        def px_dl(dl):
            """
            Returns a probability for a given distance dl from the interpolated function.
            """
            return splev(dl,temp,ext=3)
        
        counterpart = EM_counterpart.get_galaxy(0)
            
        if skymap2d is not None:
            tempsky = skymap2d.skyprob(counterpart.ra,counterpart.dec)*skymap2d.npix
        else:
            tempsky = skykernel.evaluate([counterpart.ra,counterpart.dec])*4.0*np.pi/np.cos(counterpart.dec) # remove uniform sky prior from samples
                
        tempdist = px_dl(self.cosmo.dl_zH0(counterpart.z,H0))/self.cosmo.dl_zH0(counterpart.z,H0)**2 # remove dl^2 prior from samples
        numnorm = tempdist*tempsky
        return numnorm


    def pD_H0(self,H0):
        """
        Returns p(D|H0).
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

        for i in range(len(H0)):

            def I(z,M):
                if self.basic:
                    temp = SchechterMagFunction(H0=H0[i])(M)*self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)
                else:
                    temp = SchechterMagFunction(H0=H0[i])(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(H0[i],-22.96)
            Mmax = M_Mobs(H0[i],-12.96)

            den[i] = dblquad(I,Mmin,Mmax,lambda x: 0.0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        self.pDnG = den   
        return self.pDnG

            
    def pH0(self,H0,prior='log'):
        """
        Returns p(H0)
        The prior probability of H0
        
        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1
        prior : str, optional
            The choice of prior (default='log')
            if 'log' uses uniform in log prior
            if 'uniform' uses uniform prior
            
        Returns
        -------
        float or array_like
            p(H0)
        """
        if prior == 'uniform':
            return np.ones(len(H0))
        if prior == 'log':
            return 1./H0


    def likelihood(self,H0,GW_data,EM_counterpart=None,complete=False,skymap2d=None,counterpart_case='direct'):
        """
        The likelihood for a single event
        
        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1
        GW_data : gwcosmo.likelihood.posterior_samples.posterior_samples object
            Gravitational wave event samples
        EM_counterpart : gwcosmo.prior.catalog.galaxyCatalog object, optional
            EM_counterpart data (default=None)
            If not None, will default to doing a counterpart analysis over statistical analysis
        complete : bool, optional
            Is the galaxy catalog complete to all relevant distances/redshifts? (default=False)
        skymap2d : gwcosmo.likelihood.skymap.skymap object, optional
            Gravitational wave event skymap (default=None)
            If not None, will default to using this over the GW_data
        counterpart_case : str, optional
            Choice of counterpart analysis (default='direct')
            if 'direct', will assume the counterpart is correct with certainty
            if 'pencilbeam', will assume the host galaxy is along the counterpart's line of sight, but may be beyond it
            
        Returns
        -------
        float or array_like
            p(x|H0,D)
        """    
        dH0 = H0[1]-H0[0]
        
        if EM_counterpart != None:
            
            if counterpart_case == 'direct':
                pxG = self.px_H0_counterpart(H0,GW_data,EM_counterpart,skymap2d)
                pD_H0 = self.pD_H0(H0)
                likelihood = pxG/pD_H0
                
            # The pencilbeam case is currently coded up along the line of sight of the counterpart
            # For GW170817 the likelihood produced is identical to the 'direct' counterpart case
            # TODO: allow this to cover a small patch of sky
            elif counterpart_case == 'pencilbeam':
                pxG = self.px_H0G(H0,GW_data,EM_counterpart,skymap2d)
                if all(self.pDG)==None:
                    self.pDG = self.pD_H0G(H0)
                if all(self.pGD)==None:
                    self.pGD = self.pG_H0D(H0)
                if all(self.pnGD)==None:
                    self.pnGD = self.pnG_H0D(H0)
                if all(self.pDnG)==None:
                    self.pDnG = self.pD_H0nG(H0)
                pxnG = self.px_H0nG(H0,GW_data,EM_counterpart,skymap2d)
                
                likelihood = self.pGD*(pxG/self.pDG) + self.pnGD*(pxnG/self.pDnG)            
            else:
                print("Please specify counterpart_case ('direct' or 'pencilbeam').")


        else:
            pxG = self.px_H0G(H0,GW_data,EM_counterpart,skymap2d)
            if all(self.pDG)==None:
                self.pDG = self.pD_H0G(H0)
        
            if complete==True:
                likelihood = pxG/self.pDG
        
            else:
                if all(self.pGD)==None:
                    self.pGD = self.pG_H0D(H0)
                if all(self.pnGD)==None:
                    self.pnGD = self.pnG_H0D(H0)
                if all(self.pDnG)==None:
                    self.pDnG = self.pD_H0nG(H0)
                    
                pxnG = self.px_H0nG(H0,GW_data,EM_counterpart,skymap2d)
    
                likelihood = self.pGD*(pxG/self.pDG) + self.pnGD*(pxnG/self.pDnG)
                
            if self.whole_cat == False:
                if all(self.pDnG_rest_of_sky)==None:
                    self.pDnG_rest_of_sky = self.pD_H0nG_rest_of_sky(H0)
                pxnG_rest_of_sky = self.px_H0nG_rest_of_sky(H0,GW_data,skymap2d)

                likelihood = likelihood + (pxnG_rest_of_sky/self.pDnG_rest_of_sky)
            
        return likelihood/np.sum(likelihood)/dH0




class PixelBasedLikelihood(MasterEquation):
    """
    Likelihood for a single event, evaluated using a pixel based sky map
    p(x|D, H0, I)
    Parameters
    ----
    H0 : np.ndarray
        Vector of H0 values
    catalogue : gwcosmo.prior.galaxyCatalog
        Galaxy catalogue
    skymap3d  : ligo.skymap.kde.SkyKDE
        Skymap for this event
    pdet      : gwcosmo.likelihood.detection_probability.DetectionProbability
        Detection probability class
    GMST      : float
        Greenwich Mean Sidereal Time for this event
    Optional Arguments
    ----
    linear : boolean
        Use linear cosmology?
    weighted : boolean
        Use luminosity weighting?
    counterparts : boolean
        Event has a counterpart?
    """
    def __init__(self, H0, catalog, skymap3d, GMST, pdet, linear=False, weighted=False, counterparts=False):
        super(PixelBasedLikelihood,self).__init__(H0,catalog,pdet,linear=linear,weighted=weighted,counterparts=counterparts)
        self.moc_map = skymap3d.as_healpix()
        self.pixelmap = rasterize(self.moc_map)
        self.skymap = skymap3d
        self.npix = len(self.pixelmap)
        self.nside = hp.npix2nside(self.npix)
        self.gmst = GMST
        
        # For each pixel in the sky map, build a list of galaxies and the effective magnitude threshold
        self.pixel_cats = catalog.pixelCatalogs(skymap3d)# dict of list of galaxies, indexed by NUNIQ
        self.mths = {x:18 for x in self.moc_map['UNIQ']} # just for testing
        
    def likelihood(self, H0, cum_prob=1.0):
        """
        Return likelihood of this event, given H0.
        
        Parameters
        ----
        H0 : float or ndarray
            Value of H0
            
        cum_prob : float (default = 1.0)
            Only use pixels within the cum_prob credible interval
        """
        from ligo.skymap.moc import uniq2pixarea

        moc_areas = uniq2pixarea(self.moc_map['UNIQ'])
        moc_probs = moc_areas * self.moc_map['PROBDENSITY']
        
        if cum_prob==1.0:
            moc_map = self.moc_map
        else:


            densort = np.argsort(moc_probs)[::-1]
            selection = np.cumsum(moc_probs[densort])<cum_prob
            moc_map = self.moc_map[densort[selection]]
            print('Integrating over {0} pixels containing {1} probability and {2} galaxies, covering {3} sq. deg.'\
                  .format(len(moc_map),np.sum(moc_probs[densort[selection]]),
                          np.sum([len(self.pixel_cats[x]) for x in moc_map['UNIQ'] ]),
                          np.sum([uniq2pixarea(x) for x in moc_map['UNIQ']])*(180/np.pi)**2
                          )
                  )

        result = 0.0
        # Iterate over the UNIQ pixels and add up their contributions
        from tqdm import tqdm
        with tqdm(total=len(moc_map),unit='pix',position=1) as pbar: # For displaying progress info
            for NUNIQ in moc_map['UNIQ']:
                result += self.pixel_likelihood(H0, NUNIQ) * uniq2pixarea(NUNIQ) \
                    * moc_map[moc_map['UNIQ']==NUNIQ]['PROBDENSITY'] * self.pixel_pD(H0,NUNIQ)
                pbar.update()
        return result
        
    def pixel_likelihood(self, H0, pixel):
        """
        Compute the likelihood for a given pixel
        Parameters
        ----
        H0 : np.ndarray
            0 values to compute for
        pixel : np.uint64
            UNIQ pixel index
        """
        pG = self.pixel_pG_D(H0,pixel)
        pnG = self.pixel_pnG_D(H0,pixel,pG=pG)
        return self.pixel_pD(H0,pixel)*(self.pixel_likelihood_G(H0,pixel)*pG \
                + self.pixel_likelihood_notG(H0,pixel)*pnG)      
        #return self.pixel_likelihood_G(H0,pixel)*self.pixel_pG_D(H0,pixel) \
        #        + self.pixel_likelihood_notG(H0,pixel)*self.pixel_pnG_D(H0,pixel)
        
    def pixel_likelihood_G(self,H0,pixel):
        """
        p(x | H0, D, G, pix)
        
        Parameters
        ----
        H0 : np.ndarray
            0 values to compute for
        pixel : np.uint64
            UNIQ pixel index
        """
        theta, ra = uniq2ang(pixel)
        dec = np.pi/2.0 - theta
        spl = self.pdet.pDdl_radec(ra,dec,self.gmst)
        weight,distmu,distsigma,distnorm = self.skymap(np.array([[ra,dec]]),distances=True)
        
        val = np.zeros(len(H0))
        for i,h0 in enumerate(H0):
            for gal in self.pixel_cats[pixel]:
                dl = self.cosmo.dl_zH0(gal.z,h0)
                #galprob = self.skymap.posterior_spherical(np.array([[gal.ra,gal.dec,dl]])) #TODO: figure out if using this version is okay normalisation-wise
                galprob = weight*norm.pdf(dl,distmu,distsigma)/distnorm
                detprob = self.pdet.pD_dl_eval(dl)
                val[i] += galprob/detprob
        return val
        
    
    def pixel_likelihood_notG(self,H0,pixel):
        """
        p(x | H0, D, notG, pix)
        
        Parameters
        ----
        H0 : np.ndarray
            0 values to compute for
        pixel : np.uint64
            UNIQ pixel index
        """
        theta, ra = uniq2ang(pixel)
        dec = np.pi/2.0 - theta
        spl = self.pdet.pDdl_radec(ra,dec,self.gmst)
        weight,distmu,distsigma,distnorm = self.skymap(np.array([[ra,dec]]),distances=True)
        
        #FIXME: Get correct mth
        mth = self.mths[pixel]
        
        num = np.zeros(len(H0))
        den = np.zeros(len(H0))
        
        for i,h0 in enumerate(H0):
            Schechter=SchechterMagFunction(H0=h0)
            def Inum(z,M):
                #temp = self.zprior(z)*SchechterMagFunction(H0=h0)(M)*weight*norm.pdf(self.cosmo.dl_zH0(z,h0),distmu,distsigma)/distnorm
                temp = self.zprior(z)*Schechter(M)*weight*((self.cosmo.dl_zH0(z,h0)-distmu)/distsigma)**2
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            def Iden(z,M):
                temp = Schechter(M)*self.pdet.pD_dl_eval(self.cosmo.dl_zH0(z,h0))*self.zprior(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp
        
        #def dentemp(z,M):
        #    return SchechterMagFunction(H0=H0)(M)*self.pdet.pD_dl_eval(self.cosmo.dl_zH0(z,H0))*self.zprior(z)
        #if self.weighted:
        #    def Iden(z,M):
        #        return dentemp(z,M)*L_M(M)
        #else:
        #    def Iden(z,M):
        #        return dentemp(z,M)
               
            Mmin = M_Mobs(h0,-22.96)
            Mmax = M_Mobs(h0,-12.96)
        
            num[i] = dblquad(Inum,Mmin,Mmax,lambda x: z_dlH0(dl_mM(mth,x),h0,linear=self.linear),lambda x: self.zmax,epsabs=1.49e-9,epsrel=1.49e-9)[0]/distnorm/np.sqrt(2*np.pi)/distsigma
            den[i] = dblquad(Iden,Mmin,Mmax,lambda x: z_dlH0(dl_mM(mth,x),h0,linear=self.linear),lambda x: self.zmax,epsabs=1.49e-9,epsrel=1.49e-9)[0]
            #num[i] = dblquad(Inum,Mmin,Mmax,lambda x: z_dlH0(dl_mM(mth,x),h0,linear=self.linear),lambda x: self.zmax,epsabs=0,epsrel=1.49e-3)[0]
            #den[i] = dblquad(Iden,Mmin,Mmax,lambda x: z_dlH0(dl_mM(mth,x),h0,linear=self.linear),lambda x: self.zmax,epsabs=0,epsrel=1.49e-3)[0]        
        return num/den

    def pixel_pG_D(self,H0,pixel):
        """
        p(G|D, pix)
        
        Parameters
        ----
        H0 : np.ndarray
            0 values to compute for
        pixel : np.uint64
            UNIQ pixel index
        """
        theta, ra = uniq2ang(pixel)
        dec = np.pi/2.0 - theta
        spl = self.pdet.pDdl_radec(ra,dec,self.gmst)
        mth = self.mths[pixel]
        num = np.zeros(len(H0))
        den = np.zeros(len(H0))
        
        for i,h0 in enumerate(H0):
            Schechter=SchechterMagFunction(H0=h0)
            if self.weighted:
                def I(z,M):
                    return L_M(M)*Schechter(M)*self.pdet.pD_dl_eval(self.cosmo.dl_zH0(z,h0))*self.zprior(z)
            else:
                def I(z,M):
                    return Schechter(M)*self.pdet.pD_dl_eval(self.cosmo.dl_zH0(z,h0))*self.zprior(z)
        
        # Mmin and Mmax currently corresponding to 10L* and 0.001L* respectively, to correspond with MDC
        # Will want to change in future.
        # TODO: test how sensitive this result is to changing Mmin and Mmax.
        
            Mmin = M_Mobs(h0,-22.96)
            Mmax = M_Mobs(h0,-12.96)
            num[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: z_dlH0(dl_mM(mth,x),h0,linear=self.linear),epsabs=0,epsrel=1.49e-4)[0]
            # TODO: Factorise into 2 1D integrals
            den[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        return num/den
    
    def pixel_pnG_D(self, H0, pixel,pG=None):
        """
        p(notG | D, pix)
        
        Parameters
        ----
        H0 : np.ndarray
            0 values to compute for
        pixel : np.uint64
            UNIQ pixel index
        """
        if all(pG) != None:
            return 1.0 - pG
        else:
            return 1.0 - self.pixel_pG_D(H0,pixel)
            
    def pixel_pD(self,H0,pixel):
        """
        p(D|H0,pix)
        
        Parameters
        ----
        H0 : np.ndarray
            0 values to compute for
        pixel : np.uint64
            UNIQ pixel index
        """
        theta, ra = uniq2ang(pixel)
        dec = np.pi/2.0 - theta
        spl = self.pdet.pDdl_radec(ra,dec,self.gmst)
        num = np.zeros(len(H0))
        
        for i,h0 in enumerate(H0):
            Schechter=SchechterMagFunction(H0=h0)
            if self.weighted:
                def I(z,M):
                    return L_M(M)*Schechter(M)*self.pdet.pD_dl_eval(self.cosmo.dl_zH0(z,h0))*self.zprior(z)
            else:
                def I(z,M):
                    return Schechter(M)*self.pdet.pD_dl_eval(self.cosmo.dl_zH0(z,h0))*self.zprior(z)
        
        # Mmin and Mmax currently corresponding to 10L* and 0.001L* respectively, to correspond with MDC
        # Will want to change in future.
        # TODO: test how sensitive this result is to changing Mmin and Mmax.
        
            Mmin = M_Mobs(h0,-22.96)
            Mmax = M_Mobs(h0,-12.96)
            # TODO: Can this be factorised into 2 1D integrals?
            num[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        return num
        
    def pD(self,H0):
        """
        p(D|H0)
        """
        spl = self.pdet.interp_average
        num = np.zeros(len(H0))
        
        for i,h0 in enumerate(H0):
            Schechter=SchechterMagFunction(H0=h0)
            if self.weighted:
                def I(z,M):
                    return L_M(M)*Schechter(M)*self.pdet.pD_dl_eval(self.cosmo.dl_zH0(z,h0))*self.zprior(z)
            else:
                def I(z,M):
                    return Schechter(M)*self.pdet.pD_dl_eval(self.cosmo.dl_zH0(z,h0))*self.zprior(z)
        
        # Mmin and Mmax currently corresponding to 10L* and 0.001L* respectively, to correspond with MDC
        # Will want to change in future.
        # TODO: test how sensitive this result is to changing Mmin and Mmax.
        
            Mmin = M_Mobs(h0,-22.96)
            Mmax = M_Mobs(h0,-12.96)
            # TODO: Factorise into 2 1D integrals
            num[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        return num
    
