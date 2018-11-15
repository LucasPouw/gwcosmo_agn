"""
PofH0
Ignacio Magana, Rachel Gray
"""
from __future__ import absolute_import

import numpy as np
import sys
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy.integrate import quad
from astropy import constants as const
from astropy import units as u
from astropy.table import Table

import gwcosmo

from .utilities.standard_cosmology import *
from .utilities.schechter_function import *

class pofH0(object):
    """
    Class that contains ingredients necessary to compute P(H0) in a different way.
    """
    def __init__(self,H0,galaxy_catalog,pdet,Omega_m=0.3,linear=False,weighted=True,complete=False):
        self.H0 = H0
        self.galaxy_catalog = galaxy_catalog
        self.pdet = pdet
        self.Omega_m = Omega_m
        self.linear = linear
        self.weighted = weighted
        
        if complete == True:
            self.cfactor = 0
        else:
            self.cfactor = 1.0
        
        self.likeG = None
        self.likenG = None

        self.norm = None
        self.psi = None
        self.prior_ = None
        
        self.prior_type = None
        
        self.dmax = pdet.pD_distmax()
        self.zmax = z_dlH0(self.dmax,H0=max(self.H0),linear=self.linear)
        self.zprior = redshift_prior(Omega_m=self.Omega_m,linear=self.linear)
        self.cosmo = fast_cosmology(Omega_m=self.Omega_m,linear=self.linear)
    
    def prior(self, prior_type='uniform'):
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
        dH0 = self.H0[1] - self.H0[0]
        pH0 = np.zeros(len(self.H0))
        for i in range(len(self.H0)):
            def I(z):
                return self.pdet.pD_dl_eval(self.cosmo.dl_zH0(z,self.H0[i]))*self.zprior(z)
            pH0[i] = quad(I,0,self.zmax,epsabs=0,epsrel=1.49e-4)[0]
        self.psi = pH0/np.sum(pH0*dH0)
        return self.psi
    
    def likelihoodG(self,event_data):
        """
        The likelihood for a single event P(x|G)
        """
        tables = self.extract_galaxies()
            
        coverh_x = np.ones(len(self.H0))
        for k, x in enumerate(self.H0):
            coverh_x[k] = (const.c.to('km/s') / (x * u.km / u.s / u.Mpc)).value
        
        kde_list = event_data.compute_3d_kde(coverh_x)

        ph_list = []
        i0 = 0
        for t in tables:
            i0=i0+1
            print("processing chunk " + str(i0) + " out of " + str(len(tables)))
            ph = np.zeros(len(self.H0))
            for k, x in enumerate(self.H0):
                ph[k] = event_data.compute_3d_probability(t, kde_list[k][0], kde_list[k][1], self.zmax)
            ph_list.append(ph)

        ph = np.zeros(len(self.H0))
        for j in range(0,len(tables)):
            ph += ph_list[j]
            
        self.likeG = ph
        return self.likeG
    
    def likelihoodnG(self,event_data):
        """
        The likelihood for a single event P(x|notG)
        """
        tables = self.extract_galaxies()
            
        coverh_x = np.ones(len(self.H0))
        for k, x in enumerate(self.H0):
            coverh_x[k] = (const.c.to('km/s') / (x * u.km / u.s / u.Mpc)).value
        
        epsilon = self.pdet(event_data.distance)

        ph_list = []
        for t in tables:
            ph = np.zeros(len(self.H0))
            for k, x in enumerate(self.H0):
                completion = self.cfactor * self.pd( event_data.distance / coverh_x[k], t ) / ( 4.0 * np.pi )
                ph[k] = np.mean( (completion ) / ((event_data.distance/coverh_x[k])**2) ) 
            ph_list.append(ph)

        ph = np.zeros(len(self.H0))
        for j in range(0,len(tables)):
            ph += ph_list[j]
            
        self.likenG = ph
        return self.likenG
    
    
    def normalization(self):
        """
        The normalization for a single event.
        """
        tables = self.extract_galaxies()
        
        normalization_list=[]
        for t in tables:
            normalization = np.ones(len(self.H0))
            for k, x in enumerate(self.H0):
                zmax = ( (self.dmax * u.Mpc) * (x * u.km / u.s / u.Mpc) / const.c.to('km/s') ).value
                tmpz = np.linspace(0.00001,zmax,100)
                coverh = (const.c.to('km/s') / ( x * u.km / u.s / u.Mpc )).value
                tmpr = t['z'] * coverh
                epsilon = self.pdet(tmpr)
                epLumB = t['lumB'] * epsilon
                dz = tmpz[1]-tmpz[0]
                completion = self.cfactor*self.pd(tmpz,t)
                epsilon = self.pdet(coverh*tmpz)
                tmpnorm = 0.0
                tmpnorm = np.sum(epLumB) + np.sum(epsilon*completion)*dz
                normalization[k] = tmpnorm
            normalization_list.append(normalization)
            
        normalization = np.ones(len(self.H0))
        for j in range(0,len(tables)):
            normalization += normalization_list[j]
            
        self.norm = normalization
        return self.norm

    #place this somewhere in catalog modules..
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
        if self.weighted == False:
            lumB = np.ones(nGal)
        if nGal == 1:
            tables = [Table([ra, dec, lumB, z], names=('RA','Dec', 'lumB', 'z'))]
            return tables
        else:
            nSplit = 100
            ra = np.array_split(ra, nSplit)
            dec = np.array_split(dec, nSplit)
            z = np.array_split(z, nSplit)
            lumB = np.array_split(lumB, nSplit)

            tables = []
            for k in range(nSplit):
                tables.append(Table([ra[k], dec[k], lumB[k], z[k]], names=('RA','Dec', 'lumB', 'z')))
            return tables

    #place this somewhere specific to glade... preprocessing?       
    def pd(self,x,t):
        coverh = (const.c.to('km/s') / (70 * u.km / u.s / u.Mpc)).value
        lumB = t['lumB']
        dist = t['z']*coverh
        blue_luminosity_density = np.cumsum(lumB)[np.argmax(dist>73.)]/(4.0*np.pi*0.33333*np.power(73.0,3))
        tmpd = coverh * x
        tmpp = (3.0*coverh*4.0*np.pi*0.33333*blue_luminosity_density*(tmpd-50.0)**2)
        return np.ma.masked_where(tmpd<50.,tmpp).filled(0)