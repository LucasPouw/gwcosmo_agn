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

from scipy.integrate import quad, dblquad
from scipy.stats import ncx2, norm
from astropy import constants as const
from astropy import units as u
import gwcosmo

from .utilities.standard_cosmology import *
from .utilities.schechter_function import *
from .utilities.basic import *

class MasterEquation(object):
    """
    A class to hold all the individual components of the posterior for H0,
    and methods to stitch them together in the right way.
    """
    def __init__(self,H0,galaxy_catalog,pdet,linear=False,weighted=False,use_3d_kde=True,counterparts=False):
        self.H0 = H0
        self.galaxy_catalog = galaxy_catalog
        self.pdet = pdet
        self.mth = galaxy_catalog.mth()
        self.linear = linear
        self.weighted = weighted
        self.use_3d_kde = use_3d_kde
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
        
        if self.use_3d_kde == True:
            ra, dec, z, lumB = self.extract_galaxies()
            for k, x in enumerate(self.H0):
                coverh = (const.c.to('km/s') / (x * u.km / u.s / u.Mpc)).value
                num[k] = event_data.compute_3d_probability(ra, dec, z, lumB, coverh, self.distmax) # TODO: lumB does the weighting here in the "trivial" way...

        else: # loop over all possible galaxies
            skykernel = event_data.compute_2d_kde()
            distkernel = event_data.lineofsight_distance()
            for i in range(nGal):
                gal = self.galaxy_catalog.get_galaxy(i)

                if self.weighted:
                    weight = L_mdl(gal.m,dl_zH0(gal.z,self.H0)) # TODO: make this compatible with all galaxy catalogs (ie make gal.m universal)
                else:
                    weight = 1.0

                # TODO: add possibility of using skymaps/other ways of using gw data
                if skymap2d is not None:
                    tempsky = skymap2d.skyprob(gal.ra,gal.dec) # TODO: test fully and integrate into px_H0nG
                else:
                    tempsky = skykernel.evaluate([gal.ra,gal.dec])*4.0*np.pi/np.cos(gal.dec) # remove uniform sky prior from samples

                tempdist = distkernel(dl_zH0(gal.z,self.H0,linear=self.linear))/dl_zH0(gal.z,self.H0,linear=self.linear)**2 # remove dl^2 prior from samples
                
                num += tempdist*tempsky*weight
    
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

        if self.use_3d_kde == True:
            den = np.ones(len(self.H0))
        
        if self.counterparts == True:
            print('counterparts')
            
            den = np.zeros(len(self.H0))
            catalog = gwcosmo.catalog.galaxyCatalog()
            catalog.load_glade_catalog()
            nGal = catalog.nGal()
            for i in range(nGal):
                gal = catalog.get_galaxy(i)

                if self.weighted:
                    weight = L_mdl(gal.m,dl_zH0(gal.z,self.H0)) # TODO: make this compatible with all galaxy catalogs (ie make gal.m universal)
                else:
                    weight = 1.0

                den += self.pdet.pD_dl_eval(dl_zH0(gal.z,self.H0,linear=self.linear))*weight            

        else:
            den = np.zeros(len(self.H0))       
            for i in range(nGal):
                gal = self.galaxy_catalog.get_galaxy(i)

                if self.weighted:
                    weight = L_mdl(gal.m,dl_zH0(gal.z,self.H0)) # TODO: make this compatible with all galaxy catalogs (ie make gal.m universal)
                else:
                    weight = 1.0

                den += self.pdet.pD_dl_eval(dl_zH0(gal.z,self.H0,linear=self.linear))*weight

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
                temp = SchechterMagFunction(H0=self.H0[i])(M)*self.pdet.pD_dl_eval(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z)
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
        
        if self.use_3d_kde == True:
            ra, dec, z, lumB = self.extract_galaxies()
            for k, x in enumerate(self.H0):
                coverh = (const.c.to('km/s') / (x * u.km / u.s / u.Mpc)).value
                completion = self.pd( event_data.distance / coverh, lumB, z*coverh) / ( 4.0 * np.pi )
                epsilon = self.pdet(event_data.distance)
                num[k] = np.mean( (completion ) / ((event_data.distance/coverh)**2) ) 

        else:
            distkernel = event_data.lineofsight_distance()

            for i in range(len(self.H0)):

                def Inum(z,M):
                    temp = distkernel(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z) \
                *SchechterMagFunction(H0=self.H0[i])(M)/dl_zH0(z,self.H0[i],linear=self.linear)**2 # remove dl^2 prior from samples
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
        
        if self.use_3d_kde == True:
            den = np.ones(len(self.H0))
        
        else:
            # TODO: same fixes as for pG_H0D 
            den = np.zeros(len(self.H0))

            for i in range(len(self.H0)):

                def I(z,M):
                    temp = SchechterMagFunction(H0=self.H0[i])(M)*self.pdet.pD_dl_eval(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z)
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
                return self.pdet.pD_dl_eval(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z)
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
    
    def extract_galaxies(self):
        nGal = self.galaxy_catalog.nGal()
        ra = np.zeros(nGal)
        dec = np.zeros(nGal)
        z = np.zeros(nGal)
        lumB = np.zeros(nGal)
        for i in range(nGal):
            gal = self.galaxy_catalog.get_galaxy(i)
            ra[i] = gal.ra
            dec[i] = gal.dec
            z[i] = gal.z
            lumB[i] = gal.lumB
        if all(lumB) == 0: #for mdc1 and mdc2
            lumB = np.ones(nGal)
        return ra, dec, z, lumB

    #Completion specific to catalog #TODO: Figure out place where to put this.   
    def pd(self,x,lumB,dist):
        blue_luminosity_density = np.cumsum(lumB)[np.argmax(dist>73.)]/(4.0*np.pi*0.33333*np.power(73.0,3))
        coverh = (const.c.to('km/s') / (70 * u.km / u.s / u.Mpc)).value
        tmpd = coverh * x
        tmpp = (3.0*coverh*4.0*np.pi*0.33333*blue_luminosity_density*(tmpd-50.0)**2)
        return np.ma.masked_where(tmpd<50.,tmpp).filled(0)