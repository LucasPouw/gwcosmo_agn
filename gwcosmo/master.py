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
    """
    def __init__(self,H0,galaxy_catalog,pdet,Omega_m=0.3,linear=False,weighted=False,counterparts=False):
        self.H0 = H0
        self.galaxy_catalog = galaxy_catalog
        self.pdet = pdet
        self.mth = galaxy_catalog.mth()
        self.Omega_m = Omega_m
        self.linear = linear
        self.weighted = weighted
        self.counterparts = counterparts
        
        self.pDG = None
        self.pGD = None
        self.pnGD = None
        self.pDnG = None
        self.pG = None
        self.PnG = None
        
        # Note that zmax is an artificial limit that should be well above any redshift value that could impact the results for the considered H0 values.
        # Also note, when zmax is set too high (ie 6.0), it can cause px_H0nG to incorrectly evaluate to 0 for some values of H0.
        self.distmax = pdet.pD_distmax()
        self.zmax = z_dlH0(self.distmax,H0=max(self.H0),linear=self.linear)
        self.zprior = redshift_prior(Omega_m=self.Omega_m,linear=self.linear)
        self.cosmo = fast_cosmology(Omega_m=self.Omega_m,linear=self.linear)

    def px_H0G(self,event_data,skymap2d=None):
        """
        The likelihood of the GW data given the source is in the catalogue and given H0 (will eventually include luminosity weighting). 
        
        Takes an array of H0 values, a galaxy catalogue, and posterior samples/skymap for 1 event.
        Creates a likelihood using the samples/skymap.
        Evaluates the likelihood at the location of every galaxy in 99% sky area of event.
        Sums over these values.
        Returns an array of values corresponding to different values of H0.
        """
        nGal = self.galaxy_catalog.nGal()
        num = np.zeros(len(self.H0))
        
        if skymap2d is not None:
            prob_sorted = np.sort(skymap2d.prob)[::-1]
            prob_sorted_cum = np.cumsum(prob_sorted)
            idx = np.searchsorted(prob_sorted_cum,0.999) # find index of array which bounds the 99.9% confidence interval
            minskypdf = prob_sorted[idx]*skymap2d.npix
        
        else:    
            skykernel = event_data.compute_2d_kde()
            skypdf = skykernel.evaluate([event_data.longitude,event_data.latitude])
            skypdf.sort()
            sampno = int(0.001*np.size(skypdf)) # find the position of the sample in the list which bounds the 99.9% confidence interval
            minskypdf = skypdf[sampno]

        count = 0
        data = []
        
        distkernel = event_data.lineofsight_distance()
        distmax = 2.0*np.amax(event_data.distance)
        distmin = 0.5*np.amin(event_data.distance)
        dl_array = np.linspace(distmin,distmax,500)
        vals = distkernel(dl_array)
        temp = splrep(dl_array,vals)
        
        def px_dl(dl):
            """
            Returns a probability for a given distance dl from the interpolated function.
            """
            return splev(dl,temp,ext=3)
        
        # loop over all possible galaxies
        for i in range(nGal):
            gal = self.galaxy_catalog.get_galaxy(i)

            # TODO: add possibility of using skymaps/other ways of using gw data
            if skymap2d is not None:
                tempsky = skymap2d.skyprob(gal.ra,gal.dec)*skymap2d.npix # TODO: test fully and integrate into px_H0nG
            else:
                tempsky = skykernel.evaluate([gal.ra,gal.dec])*4.0*np.pi/np.cos(gal.dec) # remove uniform sky prior from samples

            if tempsky >= minskypdf:
                count += 1

                if self.weighted:
                    weight = L_mdl(gal.m,self.cosmo.dl_zH0(gal.z,self.H0)) # TODO: make this compatible with all galaxy catalogs (ie make gal.m universal)
                else:
                    weight = 1.0
                
                if gal.z == 0:
                    tempdist = 0.0
                else:
                    tempdist = px_dl(self.cosmo.dl_zH0(gal.z,self.H0))/self.cosmo.dl_zH0(gal.z,self.H0)**2 # remove dl^2 prior from samples
                num += tempdist*tempsky*weight
            else:
                continue
        print(count)        
        return num


    def pD_H0G(self):
        """
        The normalising factor for px_H0G.
        
        Takes an array of H0 values and a galaxy catalogue.
        Evaluates detection probability at the location of every galaxy in the catalogue.
        Sums over these values.
        Returns an array of values corresponding to different values of H0.
        """  
        nGal = self.galaxy_catalog.nGal()
        
        if self.counterparts == True:
            print('counterparts')
            
            den = np.zeros(len(self.H0))
            catalog = gwcosmo.catalog.galaxyCatalog()
            catalog.load_glade_catalog()
            nGal = catalog.nGal()
            for i in range(nGal):
                gal = catalog.get_galaxy(i)

                if self.weighted:
                    weight = L_mdl(gal.m,self.cosmo.dl_zH0(gal.z,self.H0)) # TODO: make this compatible with all galaxy catalogs (ie make gal.m universal)
                else:
                    weight = 1.0

                den += self.pdet.pD_dl_eval(self.cosmo.dl_zH0(gal.z,self.H0))*weight            

        else:
            den = np.zeros(len(self.H0))       
            for i in range(nGal):
                gal = self.galaxy_catalog.get_galaxy(i)

                if self.weighted:
                    weight = L_mdl(gal.m,self.cosmo.dl_zH0(gal.z,self.H0)) # TODO: make this compatible with all galaxy catalogs (ie make gal.m universal)
                else:
                    weight = 1.0

                den += self.pdet.pD_dl_eval(self.cosmo.dl_zH0(gal.z,self.H0))*weight

        self.pDG = den
        return self.pDG


    def pG_H0D(self):
        """
        The probability that the host galaxy is in the catalogue given detection and H0.
        
        Takes an array of H0 values, and the apparent magnitude threshold of the galaxy catalogue.
        Integrates p(M|H0)*p(z)*p(D|dL(z,H0)) over z and M, incorporating mth into limits.  Should be internally normalised.
        Returns an array of probabilities corresponding to different H0s.
        """  
        # Warning - this integral misbehaves for small values of H0 (<25 kms-1Mpc-1).  TODO: fix this.
        num = np.zeros(len(self.H0)) 
        den = np.zeros(len(self.H0))
        
        # TODO: vectorize this if possible
        for i in range(len(self.H0)):
            
            def I(z,M):
                temp = SchechterMagFunction(H0=self.H0[i])(M)*self.pdet.pD_dl_eval(self.cosmo.dl_zH0(z,self.H0[i]))*self.zprior(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp
            
            # Mmin and Mmax currently corresponding to 10L* and 0.001L* respectively, to correspond with MDC
            # Will want to change in future.
            # TODO: test how sensitive this result is to changing Mmin and Mmax.
            Mmin = M_Mobs(self.H0[i],-22.96)
            Mmax = M_Mobs(self.H0[i],-12.96)
            
            num[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: z_dlH0(dl_mM(self.mth,x),self.H0[i],linear=self.linear),epsabs=0,epsrel=1.49e-4)[0]
            den[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        self.pGD = num/den
        return self.pGD    


    def pnG_H0D(self):
        """
        The probability that a galaxy is not in the catalogue given detection and H0
        
        Returns the complement of pG_H0D.
        """
        if all(self.pGD)==None:
            self.pGD = self.pG_H0D()
        self.pnGD = 1.0 - self.pGD
        return self.pnGD
       
        
    def px_H0nG(self,event_data):
        """
        The likelihood of the GW data given not in the catalogue and H0
        
        Takes an array of H0 values, an apparent magnitude threshold, and posterior samples/skymap for 1 event.
        Creates a likelihood using the samples/skymap: p(x|dL,Omega).
        Integrates p(x|dL(z,H0))*p(z)*p(M|H0) over z and M, incorporating mth into limits.
        Returns an array of values corresponding to different values of H0.
        """
        num = np.zeros(len(self.H0))
        
        distkernel = event_data.lineofsight_distance()
        
        distmax = 2.0*np.amax(event_data.distance)
        distmin = 0.5*np.amin(event_data.distance)
        dl_array = np.linspace(distmin,distmax,500)
        vals = distkernel(dl_array)
        temp = splrep(dl_array,vals)
        
        def px_dl(dl):
            """
            Returns a probability for a given distance dl from the interpolated function.
            """
            return splev(dl,temp,ext=3)

        for i in range(len(self.H0)):

            def Inum(z,M):
                temp = px_dl(self.cosmo.dl_zH0(z,self.H0[i]))*self.zprior(z) \
            *SchechterMagFunction(H0=self.H0[i])(M)/self.cosmo.dl_zH0(z,self.H0[i])**2 # remove dl^2 prior from samples
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(self.H0[i],-22.96)
            Mmax = M_Mobs(self.H0[i],-12.96)

            num[i] = dblquad(Inum,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),self.H0[i],linear=self.linear),lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        return num


    def pD_H0nG(self):
        """
        Normalising factor for px_H0nG
        
        Takes an array of H0 values and an apparent magnitude threshold.
        Integrates p(D|dL(z,H0))*p(z)*p(M|H0) over z and M, incorporating mth into limits.
        Returns an array of values corresponding to different values of H0.
        """  
        # TODO: same fixes as for pG_H0D 
        den = np.zeros(len(self.H0))

        for i in range(len(self.H0)):

            def I(z,M):
                temp = SchechterMagFunction(H0=self.H0[i])(M)*self.pdet.pD_dl_eval(self.cosmo.dl_zH0(z,self.H0[i]))*self.zprior(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp

            Mmin = M_Mobs(self.H0[i],-22.96)
            Mmax = M_Mobs(self.H0[i],-12.96)

            den[i] = dblquad(I,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),self.H0[i],linear=self.linear),lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        self.pDnG = den   
        return self.pDnG


    def pH0_D(self,prior='uniform'):
        """
        The prior probability of H0 given a detection
        
        Takes an array of H0 values and a choice of prior.
        Integrates p(D|dL(z,H0))*p(z) over z
        Returns an array of values corresponding to different values of H0.
        """
        pH0 = np.zeros(len(self.H0))
        for i in range(len(self.H0)):
            def I(z):
                return self.pdet.pD_dl_eval(self.cosmo.dl_zH0(z,self.H0[i]))*self.zprior(z)
            pH0[i] = quad(I,0,self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        if prior == 'jeffreys':
            return pH0/self.H0  
        else:
            return pH0
            
    def pH0(self,prior='log'):
        """
        The prior probability of H0 independent of detection
        
        Takes an array of H0 values and a choice of prior.
        """
        if prior == 'uniform':
            return np.ones(len(self.H0))
        if prior == 'log':
            return 1./self.H0
        
    def likelihood(self,event_data,complete=False,skymap2d=None):
        """
        The likelihood for a single event
        
        Parameters
        ----
        event_data : posterior samples (distance, ra, dec)
        complete : boolean
                    Is catalogue complete?
        skymap2d : KDe for skymap
        """    
        dH0 = self.H0[1]-self.H0[0]
        
        pxG = self.px_H0G(event_data,skymap2d)
        if all(self.pDG)==None:
            self.pDG = self.pD_H0G()
        
        if complete==True:
            likelihood = pxG/self.pDG 
        else:
            if all(self.pGD)==None:
                self.pGD = self.pG_H0D()
            if all(self.pnGD)==None:
                self.pnGD = self.pnG_H0D()
            if all(self.pDnG)==None:
                self.pDnG = self.pD_H0nG()
                
            pxnG = self.px_H0nG(event_data)

            likelihood = self.pGD*(pxG/self.pDG) + self.pnGD*(pxnG/self.pDnG)
            
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
    