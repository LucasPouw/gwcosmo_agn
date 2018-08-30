# define all of the functions that will be used in hubble_estimate (in an attempt to clear things up)

import numpy as np
from scipy.stats import ncx2, norm
from scipy.integrate import quad, dblquad, nquad
from scipy.interpolate import splev, splrep, interp1d
import pickle
import healpy as hp
from pkg_resources import resource_stream
import astropy.io
from astropy.io import fits

##DLmax = 400.0 
##DL = np.linspace(1.0,DLmax,50)
pofd_zH0_new = pickle.load(resource_stream(__name__,'pofd_zH0_new_pickle.p')) #add encoding='latin1' to make compatible with python 3
##survival_func = interp1d(DL,pofd_zH0_new,bounds_error=False,fill_value=1e-10)

pofd_zH0_new2 = pickle.load(resource_stream(__name__,'pofd_zH0_N8.p'))
##survival_func2 = interp1d(DL,pofd_zH0_new2,bounds_error=False,fill_value=1e-10)

mdc1_den = pickle.load(resource_stream(__name__,'MDCv1_den.p'))
pxG_den_18 = pickle.load(resource_stream(__name__,'pxG_den_18.p'))
#pxG_den_18 = pickle.load(resource_stream(__name__,'pxG_den_18_withschech.p'))
pxG_den_16 = pickle.load(resource_stream(__name__,'pxG_den_16.p'))
pxG_den_195 = pickle.load(resource_stream(__name__,'pxG_den_19-5.p'))
pxnG_den3D_18 = pickle.load(resource_stream(__name__,'pxnG_den3D_m18.p')) # use with 3D KDE for mth = 18
pxnG_den21D_18 = pickle.load(resource_stream(__name__,'pxnG_den21D_m18.p')) # use with 2+1D KDE for mth = 18
pxnG_den21D_195 = pickle.load(resource_stream(__name__,'pxnG_den21D_m19-5.p')) # use with 2+1D KDE for mth = 19.5
pxnG_den21D_16 = pickle.load(resource_stream(__name__,'pxnG_den21D_m16.p')) # use with 2+1D KDE for mth = 16
pxnG_den21D_14_Lweights = pickle.load(resource_stream(__name__,'pxnG_den21D_m14_Lweights.p')) # use with 2+1D KDE for mth = 14
pxnG_den21D_14_noLweights = pickle.load(resource_stream(__name__,'pxnG_den21D_m14_noLweights.p')) # use with 2+1D KDE for mth = 14
pxG_den_14_Lweights = pickle.load(resource_stream(__name__,'pxG_den_14_Lweights.p')) # use for mth = 16

## no weights
pGtemp18 = pickle.load(resource_stream(__name__,'pGnew_m18pickle.p')) # for use with mth = 18
pGtemp195 = pickle.load(resource_stream(__name__,'pGnew_m19-5pickle.p')) # for use with mth = 19.5
pGtemp16 = pickle.load(resource_stream(__name__,'pGnew_m16pickle.p')) # for use with mth = 16

## weights
pGtemp14_weight = pickle.load(resource_stream(__name__,'pG_m14_Lweights.p')) # for use with mth = 14, luminosity weights applied

from ._gwcosmo import *

#mth = 18
detmax = 6.0*dofsnr_opt(8.0)
zmax = zofd(detmax,1.5)

import scipy.stats


class FITSkernel(object):
    """
    Read a FITS file and return interpolation kernels
    """
    def __init__(self, filename):
        hdulist = fits.open(filename)
        hdu = hdulist[1]
        tbdata = hdu.data
        self.nested = True
        self.prob = tbdata['PROB']
        self.mean = tbdata['DISTMU']
        self.sigma = tbdata['DISTSIGMA']
        self.distnorm = tbdata['DISTNORM']
        self.npix = len(self.prob)
        self.nside = hp.npix2nside(self.npix)
        colat, self.ra = hp.pix2ang(self.nside, range(len(self.prob)), nest=self.nested)
        self.dec = np.pi/2.0 - colat
        #self.dist_pdfs = [scipy.stats.norm(loc = self.mean[i], scale = self.sigma[i]) for i in range(self.npix)]
        
    def probability(self, ra, dec, dist):
        """
        returns probability density at given ra, dec, dist
        p(ra,dec) * p(dist | ra,dec )
        """
        theta = np.pi/2.0 - dec
        # Step 1: find 4 nearest pixels
        (pixnums, weights) = \
            hp.get_interp_weights(self.nside, theta, ra, nest=self.nested, lonlat = False)
        
        dist_pdfs = [scipy.stats.norm(loc = self.mean[i], scale = self.sigma[i]) for i in pixnums]
        # Step 2: compute p(ra,dec)
        # p(ra, dec) = sum_i weight_i p(pixel_i)
        probvals = np.array([self.distnorm[i] * dist_pdfs[i].pdf(dist) for i,pixel in enumerate(pixnums)])
        skyprob = self.prob[pixnums]      
        p_ra_dec = np.sum( weights * probvals * skyprob )
        
        return(p_ra_dec)
    
    def __call__(self, ra, dec, dist):
        return self.probability(ra, dec, dist)
        

def zdsamp(n,xmax):
    """
    Generates samples from the z (or distance) distribution
    """
    return xmax*np.random.uniform(0,1,n)**(1.0/3.0)

def mincred(x,y,c):
    """
    returns the minimal credible region of pdf y defined on x
    """
    xfine = np.linspace(x[0],x[-1],1000)
    newdx = xfine[1]-xfine[0]
    temp = splrep(x,y)
    new = splev(xfine,temp)
    new = new/np.sum(new)/newdx
    idx = np.argsort(new)[::-1]
    s = 0
    minidx = np.inf
    maxidx = -np.inf
    cnt = 0
    while s<c:
        i = idx[cnt]
        s += newdx*new[i]
        if i<minidx:
            minidx = i
        if i>maxidx:
            maxidx = i
        cnt += 1    
    return float(xfine[minidx]),float(xfine[maxidx])


