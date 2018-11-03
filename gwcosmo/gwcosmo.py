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

from astropy import constants as const
from astropy import units as u
from astropy.table import Table

import gwcosmo

from .utilities.standard_cosmology import *
from .utilities.schechter_function import *
from .utilities.basic import *

class pofH0(object):
    """
    Class that contains ingredients necessary to compute P(H0) in a different way.
    """
    def __init__(self,H0,galaxy_catalog,pdet,linear=False,complete=False):
        self.H0 = H0
        self.galaxy_catalog = galaxy_catalog
        self.pdet = pdet
        self.linear = linear
        self.dmax = pdet.pD_distmax()
        self.zmax = z_dlH0(self.dmax,H0=max(self.H0),linear=self.linear) 
        
        if complete == True:
            self.cfactor = 0
        else:
            self.cfactor = 1.0
        
        self.post = None
        self.like = None
        self.norm = None
        self.psi = None
        self.prior_ = None
        
        self.prior_type = None
        self.dH0 = self.H0[1] - self.H0[0]
    
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
        self.psi = self.H0**3
        return self.psi
    
    def likelihood(self,event_data):
        """
        The likelihood for a single event.
        """
        print('gwcosmo')
        ra, dec, dist, z, lumB = self.extract_galaxies()
        nSplit = 100
        ra = np.array_split(ra, nSplit)
        dec = np.array_split(dec, nSplit)
        dist = np.array_split(dist, nSplit)
        z = np.array_split(z, nSplit)
        lumB = np.array_split(lumB, nSplit)
        
        tables = []
        for k in range(nSplit):
            tables.append(Table([ra[k],dec[k],lumB[k],z[k],dist[k]],names=('RA','Dec', 'lumB', 'z', 'dist')))
            
        coverh_x = np.ones(len(self.H0))
        for k, x in enumerate(self.H0):
            coverh_x[k] = (const.c.to('km/s') / (x * u.km / u.s / u.Mpc)).value
        
        kde_list = event_data.compute_3d_kde(coverh_x)
        epsilon = self.pdet(event_data.distance)

        ph_list = []
        for j in range(0,nSplit):
            print('Processing chunk ' + str(j) + ' out of ' + str(nSplit))
            ph = np.zeros(len(self.H0))
            for k, x in enumerate(self.H0):
                ph[k] = event_data.compute_3d_probability(tables[j], kde_list[k][0], kde_list[k][1], self.zmax)
                completion = self.cfactor * self.pd( event_data.distance / coverh_x[k], tables[j] ) / ( 4.0 * np.pi )
                ph[k] = ( ph[k] + np.mean( (completion ) / ((event_data.distance/coverh_x[k])**2) ) )
            ph_list.append(ph)

        ph = np.zeros(len(self.H0))
        for j in range(0,nSplit):
            ph = ph + ph_list[j]
            
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
            epsilon = self.pdet(tmpr)
            epLumB = lumB * epsilon
            dz = tmpz[1]-tmpz[0]
            completion = self.cfactor*self.pd(tmpz,lumB,dist)
            epsilon = self.pdet(coverh*tmpz)
            tmpnorm = 0.0
            tmpnorm = np.sum(epLumB) + np.sum(epsilon*completion)*dz
            normalization[k] = tmpnorm
        self.norm = normalization
        return self.norm
    
    def posterior(self, event_data, prior_type='uniform'):
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
    def pd(self,x,t):
        lumB = t['lumB']
        dist = t['dist']
        blue_luminosity_density = np.cumsum(lumB)[np.argmax(dist>73.)]/(4.0*np.pi*0.33333*np.power(73.0,3))
        coverh = (const.c.to('km/s') / (70 * u.km / u.s / u.Mpc)).value
        tmpd = coverh * x
        tmpp = (3.0*coverh*4.0*np.pi*0.33333*blue_luminosity_density*(tmpd-50.0)**2)
        return np.ma.masked_where(tmpd<50.,tmpp).filled(0)