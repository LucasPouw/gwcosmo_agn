"""
Master Equation
Rachel Gray, John Veitch, Ignacio Magana
"""
from __future__ import absolute_import

import lal
import numpy as np
import sys
import matplotlib.pyplot as plt

from scipy.integrate import quad, dblquad
from scipy.stats import ncx2, norm
from astropy import constants as const
from astropy import units as u

from .utilities.standard_cosmology import *
from .utilities.schechter_function import *
from .utilities.basic import *

class MasterEquation(object):
    """
    A class to hold all the individual components of the posterior for H0,
    and methods to stitch them together in the right way.
    """
    def __init__(self,H0,galaxy_catalog,pdet,mth=18.0,linear=False,weighted=False):
        self.H0 = H0
        self.galaxy_catalog = galaxy_catalog
        self.pdet = pdet
        self.mth = mth # TODO: get this from galaxy_catalog, not explicitly
        self.linear = linear
        self.weighted = weighted
        
        self.pDG = None
        self.pGD = None
        self.pnGD = None
        self.pDnG = None
        self.pG = None
        self.PnG = None
        
        # Note that zmax is an artificial limit that should be well above any redshift value that could impact the results for the considered H0 values.
        # Also note, when zmax is set too high (ie 6.0), it can cause px_H0nG to incorrectly evaluate to 0 for some values of H0.
        self.zmax = 1.0 # TODO: change so that this is set by some property of pdet

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
        weightnorm = 0.0 # set up normalising factor for the luminosity weights    
       
        skykernel = event_data.compute_2d_kde()
        distkernel = event_data.lineofsight_distance()

        num = np.zeros(len(self.H0)) 
        # loop over all possible galaxies
        for i in range(nGal):
            gal = self.galaxy_catalog.get_galaxy(i)
            
            if self.weighted:
                weight = L_mdl(gal.m,dl_zH0(gal.z,self.H0)) # TODO: make this compatible with all galaxy catalogs (ie make gal.m universal)
                weightnorm += weight
            else:
                weight = 1.0
                weightnorm = 1.0
            
            # TODO: add possibility of using skymaps/other ways of using gw data
            if skymap2d is not None:
                tempsky = skymap2d.skyprob(gal.ra,gal.dec) # TODO: test fully and integrate into px_H0nG
            else:
                tempsky = skykernel.evaluate([gal.ra,gal.dec])*4.0*np.pi/np.cos(gal.dec) # remove uniform sky prior from samples
            
            tempdist = distkernel(dl_zH0(gal.z,self.H0,linear=self.linear))/dl_zH0(gal.z,self.H0,linear=self.linear)**2 # remove dl^2 prior from samples
            
            num += tempdist*tempsky*weight

        return num/weightnorm


    def pD_H0G(self):
        """
        The normalising factor for px_H0G.
        
        Takes an array of H0 values and a galaxy catalogue.
        Evaluates detection probability at the location of every galaxy in the catalogue.
        Sums over these values.
        Returns an array of values corresponding to different values of H0.
        """  
        nGal = self.galaxy_catalog.nGal()
        weightnorm = 0.0
        
        den = np.zeros(len(self.H0))       
        for i in range(nGal):
            gal = self.galaxy_catalog.get_galaxy(i)
            
            if self.weighted:
                weight = L_mdl(gal.m,dl_zH0(gal.z,self.H0)) # TODO: make this compatible with all galaxy catalogs (ie make gal.m universal)
                weightnorm += weight
            else:
                weight = 1.0
                weightnorm = 1.0
                                
            den += self.pdet.pD_dl_eval(dl_zH0(gal.z,self.H0,linear=self.linear))*weight
        return den/weightnorm


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
            
            if self.weighted:
                def I(z,M):
                    return L_M(M)*SchechterMagFunction(H0=self.H0[i])(M)*self.pdet.pD_dl_eval(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z)
            else:
                def I(z,M):
                    return SchechterMagFunction(H0=self.H0[i])(M)*self.pdet.pD_dl_eval(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z)
            
            # Mmin and Mmax currently corresponding to 10L* and 0.001L* respectively, to correspond with MDC
            # Will want to change in future.
            # TODO: test how sensitive this result is to changing Mmin and Mmax.
            Mmin = M_Mobs(self.H0[i],-22.96)
            Mmax = M_Mobs(self.H0[i],-12.96)
            
            num[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: z_dlH0(dl_mM(self.mth,x),self.H0[i],linear=self.linear),epsabs=0,epsrel=1.49e-4)[0]
            den[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]

        return num/den    


    def pnG_H0D(self):
        """
        The probability that a galaxy is not in the catalogue given detection and H0
        
        Returns the complement of pG_H0D.
        """
        # TODO: set so that if pG_H0D() has been computed since the class was initialised, it uses the computed value
        return 1.0 - self.pG_H0D()
       
        
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

        for i in range(len(self.H0)):

            if self.weighted:
                def Inum(z,M):
                    return distkernel(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z) \
                *L_M(M)*SchechterMagFunction(H0=self.H0[i])(M)/dl_zH0(z,self.H0[i],linear=self.linear)**2 # remove dl^2 prior from samples
            else:
                def Inum(z,M):
                    return distkernel(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z) \
                *SchechterMagFunction(H0=self.H0[i])(M)/dl_zH0(z,self.H0[i],linear=self.linear)**2 # remove dl^2 prior from samples

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

            if self.weighted:
                def I(z,M):
                    return L_M(M)*SchechterMagFunction(H0=self.H0[i])(M)*self.pdet.pD_dl_eval(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z)
            else:
                def I(z,M):
                    return SchechterMagFunction(H0=self.H0[i])(M)*self.pdet.pD_dl_eval(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z)
            
            Mmin = M_Mobs(self.H0[i],-22.96)
            Mmax = M_Mobs(self.H0[i],-12.96)
            
            den[i] = dblquad(I,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),self.H0[i],linear=self.linear),lambda x: self.zmax,epsabs=0,epsrel=1.49e-4)[0]
            
        return den


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
    
    def p_G(self):
        """
        The prior probability that a galaxy is in the catalog.
        """
        L0 = 1.98e-2 # Kopparapu et al
        #def I(z):
        #    return L0*z**2
        I = lambda x: L0*x**2
        
        nGal = self.galaxy_catalog.nGal()
        zgal = np.zeros(nGal)
        for i in range(nGal):
            gal = self.galaxy_catalog.get_galaxy(i)
            zgal[i] = gal.z
        zgal = np.sort(zgal)
        zs = max(zgal)
        
        dl = np.linspace(0.1,400.0,20)
        zH0 = dl*self.H0/const.c.to('km/s').value
    
        zG = np.zeros(len(self.H0))
        znG = np.zeros(len(self.H0))
        for i in range(len(self.H0)):
            if zH0[i] < zs:
                zG[i] = zH0[i]
            else: 
                znG[i] = zH0[i]
        
        LG = L0*pz_nG(zG)/quad(I, 0., zs)[0]
        LnG = L0*pz_nG(znG)/quad(I, zs, self.zmax)[0]
        return LG/(LG + LnG)

    def p_nG(self):
        """
        The prior probability that a galaxy is not in the catalog.
        
        Returns the complement of pG.
        """
        return 1.0 - self.p_G()
    
    def likelihood_PRB(self,event_data):
        """
        The likelihood for a single event
        """ 
        pG = self.p_G()
        pnG = self.p_nG()
        
        dH0 = self.H0[1]-self.H0[0]
        
        #pxG = self.px_H0G(event_data)
        pxG = self.px_H0G3D(event_data)
        pxnG = self.px_H0nG(event_data)

        likelihood_prb = pG*pxG + pnG*pxnG
            
        return likelihood_prb

class pofH0(object):
    """
    Class that contains ingredients necessary to compute P(H0) in a different way.
    """
    def __init__(self,H0,galaxy_catalog,pdet,linear=False,zmax=1.0,dmax=400.0,cfactor=1.0, option='GW170817'):
        self.H0 = H0
        self.galaxy_catalog = galaxy_catalog
        self.pdet = pdet
        self.linear = linear
        self.dmax = dmax
        self.zmax = zmax
        self.cfactor = cfactor
        
        self.post = None
        self.like = None
        self.norm = None
        self.psi = None
        self.prior_ = None
        
        self.prior_type = None
        self.dH0 = self.H0[1] - self.H0[0]
        self.option = option
    
    def prior(self, prior_type='log'):
        self.prior_type = prior_type
        if prior_type == 'log':
            self.prior_ = 1./self.H0
            return self.prior_
        if prior_type == 'uniform':
            self.prior_ = np.ones(len(self.H0))
            return self.prior_
        
    def psiH0(self):
        """
        The infamous H0**3 term.
        """
        pH0 = np.zeros(len(self.H0))
        for i in range(len(self.H0)):
            def I(z):
                return self.pdet.pD_dl_eval(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z)
        pH0[i] = quad(I,0,self.zmax,epsabs=0,epsrel=1.49e-4)[0]
        #self.psi = pH0/(np.sum(pH0)*self.dH0)
        #return pH0/(np.sum(pH0)*self.dH0)
        self.psi = self.H0**3
        return self.psi
    
    def likelihood(self,event_data):
        """
        The likelihood for a single event.
        """
        ra, dec, dist, z, lumB = self.extract_galaxies()
            
        ph = np.zeros(len(self.H0))
        for k, x in enumerate(self.H0):
            coverh = (const.c.to('km/s') / (x * u.km / u.s / u.Mpc)).value
            ph[k] = event_data.compute_3d_probability(ra, dec, dist, z, lumB, coverh)
            completion = self.cfactor * self.pd( event_data.distance / coverh, lumB, dist, self.option ) / ( 4.0 * np.pi )
            if self.option == 'GW170817':
                epsilon = 0.5*(1 - np.tanh(3.3*np.log(event_data.distance/80.)))
            if self.option == 'MDC1':
                epsilon = 0.5*(1 - np.tanh(2.8*np.log(event_data.distance/90.)))
            ph[k] = ( ph[k] + np.mean( (completion ) / ((event_data.distance/coverh)**2) ) )
            print(ph[k])
        self.like = ph
        return self.like
    
    def normalization(self):
        """
        The normalization for a single event.
        """
        ra, dec, dist, z, lumB = self.extract_galaxies()
        
        normalization = np.ones(len(self.H0))
        for k, x in enumerate(self.H0):
            zmax = ( (self.dmax * u.Mpc) * (x * u.km / u.s / u.Mpc) / const.c.to('km/s') ).value
            tmpz = np.linspace(0.00001,zmax,100)
            coverh = (const.c.to('km/s') / ( x * u.km / u.s / u.Mpc )).value
            tmpr = z * coverh
            epsilon = 0.5*(1 - np.tanh(3.3*np.log(tmpr/80.)))
            epLumB = lumB * epsilon
            dz = tmpz[1]-tmpz[0]
            completion = self.cfactor*self.pd(tmpz,lumB,dist,self.option)
            if self.option == 'GW170817':
                epsilon = 0.5*(1 - np.tanh(3.3*np.log(coverh*tmpz/80.)))
            if self.option == 'MDC1':
                epsilon = 0.5*(1 - np.tanh(2.8*np.log(coverh*tmpz/90.)))
            tmpnorm = 0.0
            tmpnorm = np.sum(epLumB) + np.sum(epsilon*completion)*dz
            normalization[k] = tmpnorm
        self.norm = normalization
        return self.norm
    
    def posterior(self, event_data, prior_type='log'):
        """
        The posterior for a single event.
        """
        if self.like is None:
            print("Calculating aofh")
            norm = self.normalization()
            psi = self.psiH0()
            if prior_type == 'log':
                prior = self.prior('log')
            if prior_type == 'uniform':
                prior = self.prior('uniform')
            print("Setting up" + str(prior_type) + "prior")
            print("Calculating likelihood from H0 = " + str(self.H0[0]) + " to " + str(self.H0[-1]) + ", " + str(len(self.H0)) + " bins...")
            like = self.likelihood(event_data)

            self.like = like
            self.norm = norm
            self.psi = psi
            self.prior_ = prior
            self.prior_type = prior_type

        posterior=self.like*self.prior_*self.psi/self.norm
        self.post = posterior/np.sum(posterior*self.dH0)
        return self.post
    
    def plot(self,fname='posterior.pdf'):
        """
        Make plot of P(H0).
        """
        if self.post is None:
            print("Calculate posterior first fool...")
            return 0
        else:
            fig, ax = plt.subplots()
            ax.plot(self.H0,self.post,linewidth=2,color='orange',label='Posterior')
            if self.prior_type == 'log':
                ax.plot(self.H0,self.prior_/np.sum(self.prior_*self.dH0),'g-.',linewidth=2,label='Log Prior')
            if self.prior_type == 'uniform':
                ax.plot(self.H0,self.prior_/np.sum(self.prior_*self.dH0),'g-.',linewidth=2,label='Uniform Prior')
            ax.axvline(70.,0.0, 1,color='r', label='$H_0$ = 70 (km s$^{-1}$ Mpc$^{-1}$)')
            ax.set_xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)',size='large')
            ax.set_ylabel('$p(H_0|data)$ (km$^{-1}$ s Mpc)',size='large')
            legend = ax.legend(loc='upper right', shadow=True, fontsize='medium')
            legend.get_frame().set_facecolor('#FFFFFF')
            fig.savefig(fname,format='pdf')
            plt.show()
            
    #place this somewhere in catalog modules..
    def extract_galaxies(self):
        nGal = self.galaxy_catalog.nGal()
        ra = np.zeros(nGal)
        dec = np.zeros(nGal)
        dist = np.zeros(nGal)
        z = np.zeros(nGal)
        lumB = np.zeros(nGal)
        for i in range(nGal):
            gal = self.galaxy_catalog.get_galaxy(i)
            ra[i] = gal.ra
            dec[i] = gal.dec
            dist[i] = gal.distance
            z[i] = gal.z
            lumB[i] = gal.lumB
        return ra, dec, dist, z, lumB

    #place this somewhere specific to glade... preprocessing?       
    def pd(self,x,lumB,dist,option):
        option = self.option
        if option == 'GW170817':
            blue_luminosity_density = np.cumsum(lumB)[np.argmax(dist>73.)]/(4.0*np.pi*0.33333*np.power(73.0,3))
            coverh = (const.c.to('km/s') / (70 * u.km / u.s / u.Mpc)).value
            tmpd = coverh * x
            tmpp = (3.0*coverh*4.0*np.pi*0.33333*blue_luminosity_density*(tmpd-50.0)**2)
            return np.ma.masked_where(tmpd<50.,tmpp).filled(0)
        if option == 'MDC1':
            blue_luminosity_density = 0.00013501121143
            coverh = (const.c.to('km/s') / (70 * u.km / u.s / u.Mpc)).value
            tmpd = coverh * x
            tmpp = (3.0*coverh*4.0*np.pi*0.33333*blue_luminosity_density*(tmpd-0.0)**2)
            return ma.masked_where(tmpd<435.,tmpp).filled(0)
    
