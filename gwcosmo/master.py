"""
Master Equation
Rachel Gray, John Veitch, Ignacio Magana
"""
import lal
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import ncx2, norm
from gwcosmo.standard_cosmology import *
from gwcosmo.schechter_function import *
from gwcosmo.prior.basic import *

class MasterEquation(object):
    """
    A class to hold all the individual components of the "master equation" (someone please suggest a better name), and stitch them together in the right way
    """    
    def __init__(self,H0,galaxy_catalog,pdet,mth=18.0,linear=False):
        self.H0 = H0
        self.galaxy_catalog = galaxy_catalog
        self.pdet = pdet
        self.mth = mth # TODO: get this from galaxy_catalog, not explicitly
        self.linear = linear
        
        self.pDG = None
        self.pGD = None
        self.pnGD = None
        self.pDnG = None
        self.pG = None
        self.PnG = None
    
    def px_H0G(self,event_data):
        """
        The likelihood of the GW data given the source is in the catalogue and given H0 (will eventually include luminosity weighting). 
        
        Takes an array of H0 values, a galaxy catalogue, and posterior samples/skymap for 1 event.
        Creates a likelihood using the samples/skymap.
        Evaluates the likelihood at the location of every galaxy in 99% sky area of event.
        Sums over these values.
        Returns an array of values corresponding to different values of H0.
        """
        nGal = self.galaxy_catalog.nGal()
        weight = np.ones(nGal)
        
        skykernel = event_data.compute_2d_kde()
        distkernel = event_data.dist_prior_corr()

        num = np.zeros(len(self.H0)) 
        # loop over all possible galaxies
        for i in range(nGal):
            gal = self.galaxy_catalog.get_galaxy(i)
            # TODO: add possibility of using skymaps/other ways of using gw data
            tempsky = skykernel.evaluate([gal.ra,gal.dec])*4.0*np.pi/np.cos(gal.dec) # remove uniform sky prior from samples
            tempdist = distkernel(dl_zH0(gal.z,self.H0,linear=self.linear))
            
            num += tempdist*tempsky*weight[i]

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
        den = np.zeros(len(self.H0))       
        for i in range(nGal):
            gal = self.galaxy_catalog.get_galaxy(i)
            z = gal.z
            den += self.pdet.pD_dl_eval(dl_zH0(z,self.H0,linear=self.linear))
        return den


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
                return SchechterMagFunction(H0=self.H0[i])(M)*self.pdet.pD_dl_eval(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z)
            
            # Mmin and Mmax currently corresponding to 10L* and 0.001L* respectively, to correspond with MDC
            # Will want to change in future.
            # TODO: test how sensitive this result is to changing Mmin and Mmax.
            Mmin = M_Mobs(self.H0[i],-22.96)
            Mmax = M_Mobs(self.H0[i],-12.96)
            
            # TODO: change zmax = 6.0 to a reasonably high limit
            num[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: z_dlH0(dl_mM(self.mth,x),self.H0[i],linear=self.linear),epsabs=0,epsrel=1.49e-4)[0]
            den[i] = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: 6.0,epsabs=0,epsrel=1.49e-4)[0]
            #print("{}: Calculated for H0 {}/{}".format(time.asctime(),i+1,len(self.H0)))
        
        return num/den    


    def pnG_H0D(self,pG_H0D):
        """
        The probability that a galaxy is not in the catalogue given detection and H0
        
        Returns the complement of pG_H0D.
        """
        return 1.0 - pG_H0D
       
        
    def px_H0nG(self,event_data):
        """
        The likelihood of the GW data given not in the catalogue and H0
        
        Takes an array of H0 values, an apparent magnitude threshold, and posterior samples/skymap for 1 event.
        Creates a likelihood using the samples/skymap: p(x|dL,Omega).
        Integrates p(x|dL(z,H0))*p(z)*p(M|H0) over z and M, incorporating mth into limits.
        Returns an array of values corresponding to different values of H0.
        """
        num = np.zeros(len(self.H0))
        
        #skykernel = event_data.compute_2d_kde() 
        distkernel = event_data.dist_prior_corr()

        for i in range(len(self.H0)):
            def Inum(z,M):
                return distkernel(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z)*SchechterMagFunction(H0=self.H0[i])(M)
            Mmin = M_Mobs(self.H0[i],-22.96)
            Mmax = M_Mobs(self.H0[i],-12.96)

            num[i] = dblquad(Inum,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),self.H0[i],linear=self.linear),lambda x: 6.0,epsabs=0,epsrel=1.49e-4)[0]
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
                return SchechterMagFunction(H0=self.H0[i])(M)*self.pdet.pD_dl_eval(dl_zH0(z,self.H0[i],linear=self.linear))*pz_nG(z)
            
            Mmin = M_Mobs(self.H0[i],-22.96)
            Mmax = M_Mobs(self.H0[i],-12.96)
            
            den[i] = dblquad(I,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),self.H0[i],linear=self.linear),lambda x: 6.0,epsabs=0,epsrel=1.49e-4)[0]
            #print("{}: Calculated for H0 {}/{}".format(time.asctime(),i+1,len(self.H0)))
        
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
       
            pH0[i] = quad(I,0,6.0,epsabs=0,epsrel=1.49e-4)[0]
                
        if prior == 'jeffreys':
            return pH0/self.H0  
        else:
            return pH0

    def pG(self):
        """
        The prior probability that a galaxy is in the catalog.
        
        Takes an array of H0 values and a choice of prior.
        Integrates p(D|dL(z,H0))*p(z) over z
        Returns an array of values corresponding to different values of H0.
        """
        pG = np.zeros(len(self.H0))
        return pG

    def pnG(self,pG):
        """
        The prior probability that a galaxy is not in the catalog.
        
        Returns the complement of pG.
        """
        return 1.0 - pG

    def psiH0(self):
        """
        The infamous H0**3 term.
        """
        return self.H0**3        

    def likelihood_PRB(self,event_data):
        """
        The likelihood for a single event
        """    
        dH0 = self.H0[1]-self.H0[0]
        
        pxG = self.px_H0G(event_data)
        pxnG = self.px_H0nG(event_data)

        pG = self.pG()
        pnG = self.pnG(pG)

        likelihood = pG*pxG + pnG*pxnG
            
        return likelihood

    def likelihood(self,event_data,complete=False):
        """
        The likelihood for a single event
        """    
        dH0 = self.H0[1]-self.H0[0]
        
        pxG = self.px_H0G(event_data)
        if all(self.pDG)==None:
            self.pDG = self.pD_H0G()
        
        if complete==True:
            likelihood = pxG/self.pDG
        
        # TODO: check this works in python 3 as well as 2.7
        else:
            if all(self.pGD)==None:
                self.pGD = self.pG_H0D()    
            if all(self.pnGD)==None:
                self.pnGD = self.pnG_H0D(self.pGD)
            if all(self.pDnG)==None:
                self.pDnG = self.pD_H0nG()
            
            pxnG = self.px_H0nG(event_data)
            
            likelihood = self.pG*(pxG/self.pDG) + self.pnG*(pxnG/self.pDnG)
            
        return likelihood/np.sum(likelihood)/dH0
        
