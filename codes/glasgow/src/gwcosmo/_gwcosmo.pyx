cimport numpy as np
import numpy as np
import scipy
from scipy.stats import ncx2, norm
from scipy.integrate import quad, dblquad, nquad
from scipy.interpolate import splev, splrep, interp1d, griddata, interpn
import pickle
import healpy as hp
from pkg_resources import resource_stream
from cython_gsl cimport *
from cpython cimport array

from cython cimport view

cpdef double DLmax = 400.0 


detmax = 6.0*dofsnr_opt(8.0)
zmax = zofd(detmax,1.5)
eps = 1.49e-12

from .gwcosmo import pofd_zH0_new, pofd_zH0_new2

DL = np.linspace(1.0,DLmax,50)


class vector_interp(object):
    def __init__(self, xvals, yvals):
        self.dim = yvals.shape[1]
        self.interps = [my_gsl_interpolator(xvals, yvals[:,i]) for i in range(self.dim)]
    def __call__(self,x):
        return np.array( [interp(x) for interp in self.interps] )


cdef class my_gsl_interpolator:
    cdef gsl_spline * spline
    cdef gsl_interp_accel *acc 
    cdef view.array xvals,yvals
    cdef double xmax
    cdef double xmin
    cdef double fill_value
    def __cinit__(self, np.ndarray[double] xin ,np.ndarray[double] yin, fill_value=1e-10, bounds_error=False):
        if len(xin)!=len(yin):
           print('ERROR: x and y must be same length')
        npoints = len(xin)
        
        self.xvals = view.array(shape=(1,npoints), itemsize=sizeof(double), format='d', mode='c')
        self.yvals = view.array(shape=(1,npoints), itemsize=sizeof(double), format='d', mode='c')
        self.xvals[:] = np.ascontiguousarray(xin)[:]
        self.yvals[:] = np.ascontiguousarray(yin)[:]
        self.fill_value = fill_value
        self.xmax = np.max(xin)
        self.xmin = np.min(xin)
        self.spline = gsl_spline_alloc(gsl_interp_linear,npoints)
        self.acc = gsl_interp_accel_alloc()
        gsl_spline_init(self.spline,<double *> self.xvals.data,<double *> self.yvals.data,npoints)
    def __dealloc__(self):
        gsl_spline_free(self.spline)
    def __call__(self,x):
        if isinstance(x,np.ndarray):
            return self.call_array(x)
        
        if x > self.xmax:
            return self.fill_value
        elif x < self.xmin:
            return 1.0
        else:
            return gsl_spline_eval(self.spline,x,self.acc)
    def call_array(self, x):
        """
        Same as call but for arrays
        """
        result = np.zeros(x.shape)
        for i in range(x.shape[-1]):
            if x[i] < self.xmin: result[i]=1
            elif x[i]>self.xmax: result[i]=self.fill_value
            else: result[i]=self.__call__(x[i])
        return result
        

survival_func = my_gsl_interpolator(DL,pofd_zH0_new)

cpdef survival_func_1D(dist):
    """
    produces same results as survival_func and can be called from outside this module
    however it is slower than survival_func so in general should not be used except for illustrative purposes
    """
    return my_gsl_interpolator(DL,pofd_zH0_new)(dist)


survival_func2 = vector_interp(DL,pofd_zH0_new2.T)





def pH0_det(H0,prior='uniform'):
    """
    The prior probability of H0 given a detection
    """
    def I(z):
        return z*z*survival_func(dofz(z,H0))
   
    num,err1 = quad(I,0,zmax,epsabs=eps,epsrel=eps)
            
    if prior == 'jeffreys':
        return num/H0
        
    else:
        return num


####################################
####################################
# 2+1D KDE functions below. Speed not required, hence def, not cpdef


def px_H0G_num(catz,rhosq,H0,gmstrad,distkernel=None,skykernel=None,catm=None,catra=None,catdec=None,distpost=False):
    """
    the likelihood of the GW data given the source is in the catalogue and detected and given H0
    
    We allow for the use of GW posterior distributions on the distance, or a simulated SNR measurement.
    We allow absolute magnitude weighting and options for ignoring GW selection effects.
    We also allow the use of the likelihood or the posterior on distance
    
    """
    # setup luminosity weighting
    if catm is not None:
        weight = luminosity(catm,dofz(catz,H0)) ##### TO DO: fix normalisation for this case #####
        #weight /= np.sum(weight)
    else:
        weight = np.ones(catz.size)
    
    num = 0.0
    # loop over all possible galaxies
    for i in xrange(catz.size):
        # if using real data samples
        if distkernel is not None:
            tempsky = skykernel.evaluate([catra[i],catdec[i]])*4.0*np.pi/np.cos(catdec[i]) # remove uniform sky prior from samples
            tempdist = distkernel.evaluate(dofz(catz[i],H0))
            if distpost==False:
                # we remove distance squared prior from the samples
                tempdist /= dofz(catz[i],H0)**2
        else:
            tempdist = ncx2.pdf(rhosq,2,snr(dofz(catz[i],H0))**2)
            tempsky = 1.0
        
        num += tempdist*tempsky*weight[i]

    return num


def px_H0G_den(catz,H0,catm=None):
    """
    the denominator for the likelihood of the GW data given the source is in the catalogue and detected and given H0
    """
    den = 0.0
    if catm is not None:
        for i in xrange(catz.size):
            sfunc = survival_func(dofz(catz[i],H0))
            lum = luminosity(catm[i],dofz(catz[i],H0))
            den += sfunc*lum
            
    else:
        for i in xrange(catz.size):
            sfunc = survival_func(dofz(catz[i],H0))
            den += sfunc
    
    return den



def pcat_detH0(H0,alpha,mth,weight=False):
    """
    the probability that the (luminosity weighted) host galaxy is in the catalogue given detection and H0
    """
    Mmin = Mlower(H0)
    Mmax = Mupper(H0)
    
    if weight==True:
        def I(z,M):
            return gal_weight(M,alpha,H0)*z*z*survival_func(dofz(z,H0))   

    else:
        def I(z,M):
            return pM_H0_schech(M,alpha,H0)*z*z*survival_func(dofz(z,H0))
    
    num,err1 = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: zofmMH0(mth,x,H0),epsabs=eps,epsrel=eps)
    den,err2 = dblquad(I,Mmin,Mmax,lambda x: 0,lambda x: zmax, epsabs=eps,epsrel=eps)
    
    return num/den



def pnocat_detH0(H0,pG=None):
    """
    the probability that a galaxy is not in the catalogue given detection and H0
    """
    if pG is not None:
        return 1.0 - pG
    else:
        return 1.0 - pcat_detH0(H0)


   
    
def px_H0nG_num(rhosq,H0,alpha,mth,distkernel=None,weight=False):
    """
    the likelihood of an snr value given not in the catalogue and detected and H0
    """
    Mmin = Mlower(H0)
    Mmax = Mupper(H0)
    
    if weight==True:
        def Inum(z,M):
            # we remove distance squared prior from the samples
            return distkernel.evaluate(dofz(z,H0))*z*z*gal_weight(M,alpha,H0)/dofz(z,H0)**2
    
    else:
        if distkernel is not None:
            def Inum(z,M):
                # we remove distance squared prior from the samples
                return distkernel.evaluate(dofz(z,H0))*z*z*pM_H0_schech(M,alpha,H0)/dofz(z,H0)**2
        else:    
            def Inum(z,M):
                return ncx2.pdf(rhosq,2,snr(dofz(z,H0))**2)*z*z*pM_H0_schech(M,alpha,H0)
      
    num,errnum = dblquad(Inum,Mmin,Mmax,lambda x: zofmMH0(mth,x,H0),lambda x: zmax, epsabs=0,epsrel=1.49e-4)
    return num


def px_H0nG_den(H0,alpha,mth,weight=False):
    """
    Normalising factor for px_H0nG
    """  
    Mmin = Mlower(H0)
    Mmax = Mupper(H0)
    
    if weight==True:
        def Iden(z,M):
            return z*z*gal_weight(M,alpha,H0)*survival_func(dofz(z,H0))
    
    else:
        def Iden(z,M):
            return z*z*pM_H0_schech(M,alpha,H0)*survival_func(dofz(z,H0))
    
    den,errden = dblquad(Iden,Mmin,Mmax,lambda x: zofmMH0(mth,x,H0),lambda x: zmax, epsabs=0,epsrel=1.49e-4)
    return den
    
#########################
#########################


cpdef int Nside = 8 # characterises survival_func2
cpdef int no_pix = hp.pixelfunc.nside2npix(Nside) # number of pixels on GW selection effects map
pix_ind = range(0,no_pix)

cpdef double survival_map(double H0, double z, double RA, double Dec, double gmstrad):
    """
    the survival map as a callable function
    """
    hpxmap = survival_func2(dofz(z,H0))[pix_ind]
    return hp.get_interp_val(hpxmap,np.pi/2.0-Dec,RA-gmstrad)


cpdef double pM_H0_schech(double M, double alpha, double H0):
    """
    the assumed absolute magnitude distribution (Schechter function), given the Hubble constant (un-normalised!)
    """
    return 0.4*np.log(10.0)*np.power(10.0,0.4*(Mstar(H0)-M)*(alpha+1))*np.exp(-np.power(10.0,0.4*(Mstar(H0)-M)))

cpdef double gal_weight(double M, double alpha, double H0):
    """
    the weighted absolute magnitude distribution (Lp(L)) (Schechter function), given the Hubble constant (un-normalised!)
    """
    return 0.4*np.log(10.0)*np.power(10.0,0.4*(Mstar(H0)-M)*(alpha+2))*np.exp(-np.power(10.0,0.4*(Mstar(H0)-M)))#*lofM(M)#*np.exp(-0.4*M/np.log10(np.exp(1)))

cpdef double Mstar(double H0):
    """
    the turning point of the Schechter function
    """
    return -20.46 + 5.0*np.log10(H0)

cpdef double Lstar(double H0):
    """
    the turning point of the Schechter function in luminosity
    """
    Lsun = 3.828e26
    return 1.2e10*H0**(-2)*Lsun

cpdef double Mlower(double H0):
    """
    the lower integral limit of the Schechter function, corresponding to 10L*
    """
    return -22.96 + 5.0*np.log10(H0)

cpdef double Mupper(double H0):
    """
    the upper integral limit of the Schechter function, corresponding to 0.001L*
    """
    return -12.96 + 5.0*np.log10(H0)


def dofz(z,H0):
    """
    The cosmological model - redshift of 0.01 at 40 Mpc for the true Hubble constant
    """
    return (3.0e3*z)/H0


cpdef double zofd(double d, double H0):
    """
    The cosmological model - redshift of 0.01 at 40 Mpc for the true Hubble constant
    """
    return H0*d/3.0e3

cpdef double snr(double d):
    """
    The snr at a given distance - SNR of 8 at 100 Mpc
    """   
    return 8.0*(100.0/d)

cpdef double dofsnr_opt(double rho):
    """
    The optimal distance at a given optimal SNR - SNR of 8 at 100 Mpc
    """   
    return 8.0*(100.0/rho)

def luminosity(m, d):
    """
    returns the luminosity of a source given an apparent magnitude and a distance in Mpc
    """
    return 3.0128e28*10**(-0.4*(m-5.0*(np.log10(1e6*d)-1.0)))

cpdef double pM_H0(double M, double Mmu, double dM,double H0):
    """
    the assumed absolute magnitude distribution given the Hubble constant
    """
    return norm.pdf(M,Mmu+5.0*np.log10(H0),dM)

cpdef double Msamp(int n, double Mmu, double dM, double H0):
    """
    Generates samples from the absolute magnitude distribution
    """
    return norm.rvs(Mmu+5.0*np.log10(H0),dM,size=n)

cpdef double appm(double z, double H0, double M):
    """
    the apparent magnitude given the redshift and absolute magnitude
    """
    return M + 5.0*(np.log10(1e6*dofz(z,H0)) - 1.0)

cpdef double absM(double z, double H0, double m):
    """
    the absolute magnitude given the redshift and apparent magnitude
    """
    return m - 5.0*(np.log10(1e6*dofz(z,H0)) - 1.0)

cpdef double dofmM(double m, double M):
    """
    the distance given an apparent and absolute magnitude (Mpc)
    """
    return 1e-6*10**(1.0+0.2*(m-M))

cpdef double zofmMH0(double m, double M, double H0):
    """
    the redshift given an apparent and absolute magnitude and a Hubble constant
    """
    return zofd(dofmM(m,M),H0)

cpdef double lofM(double M):
    """
    the luminosity of a source given its absolute magnitude (L0=3.0128e28 Js-1 is the zero point luminosity)
    """
    return 3.0128e28*10**(-0.4*M)

cpdef double Mofl(double l):
    """
    the absolute magnitude of a source given its luminosity
    """
    return -2.5*np.log10(l/3.0128e28)










######## 3D version under construction #########
################################################

## test without the if statement in the integrand
cpdef double I_px_H0nG_num_3D(double z,double M,double RA,double Dec,double H0,double alpha,double gmstrad,double mth,distskykernel):
    # we remove distance squared prior from the samples
    return distskykernel.evaluate([dofz(z,H0),RA,Dec])*z*z*pM_H0_schech(M,alpha,H0)*np.cos(Dec)/dofz(z,H0)**2 
def lim0num(double M,double RA,double Dec,double H0,double alpha,double gmstrad,double mth,distskykernel):
    return [zofmMH0(mth,M,H0),zmax]

cpdef double px_H0nG_num_3D(double H0,double alpha,double gmstrad,double mth,distskykernel=None):
    """
    the likelihood of an snr value given not in the catalogue and detected and H0, including RA and Dec in calculation
    """     
    ranges = [lim0num,[Mlower(H0),Mupper(H0)],[0.0,2.0*np.pi],[-np.pi/2.0,np.pi/2.0]]
    args = [H0,alpha,gmstrad,mth,distskykernel]
    opts = {'epsabs':1.49e-1,'epsrel':1.49e-1}
    num,errnum = nquad(I_px_H0nG_num_3D,ranges,args,opts=[opts,opts,opts,opts])
    #print num,errnum
    return num


## under construction
cpdef double I_px_H0nG_den_3D(double z,double M,double RA,double Dec,double H0,double alpha,double gmstrad,double mth):
    return z*z*pM_H0_schech(M,alpha,H0)*survival_map(H0,z,RA,Dec,gmstrad)*np.cos(Dec)
def lim0den(double M,double RA,double Dec,double H0,double alpha,double gmstrad,double mth):
    return [zofmMH0(mth,M,H0),zmax]

cpdef double px_H0nG_den_3D(double H0, double gmstrad, double alpha,double mth):
    """
    Normalising factor for px_H0nG including RA and Dec in calculation
    """  
    ranges = [lim0den,[Mlower(H0),Mupper(H0)],[0.0,2.0*np.pi],[-np.pi/2.0,np.pi/2.0]]
    args = [H0,alpha,gmstrad,mth]
    opts = {'epsabs':1.49e-1,'epsrel':1.49e-1}
    den,errden = nquad(I_px_H0nG_den_3D,ranges,args,opts=[opts,opts,opts,opts])
    print den,errden
    return den/4.0*np.pi



## under construction
def px_H0G_num_3D(catz,rhosq,H0,gmstrad,distskykernel=None,catm=None,catra=None,catdec=None,basic=False,distpost=False):
    """
    the likelihood of the GW data given the source is in the catalogue and detected and given H0
    
    We allow for the use of GW posterior distributions on the distance, or a simulated SNR measurement.
    We allow absolute magnitude weighting and options for ignoring GW selection effects.
    We also allow the use of the likelihood or the posterior on distance
    
    """
    # compute likelihood on Hubble including the distance prior
    num = 0.0
    den = 0.0
    bas = 0.0
        
    # setup luminosity weighting
    if catm is not None:
        weight = luminosity(catm,dofz(catz,H0))
    else:
        weight = np.ones(catz.size)
           
    # loop over all possible galaxies
    for i in xrange(catz.size):
        if distskykernel is not None:
            templik = distskykernel([dofz(catz[i],H0),catra[i],catdec[i]])*4.0*np.pi/np.cos(catdec[i]) #### remove square brackets if this doesn't work for skymaps #####
            #print(catra[i],catdec[i],dofz(catz[i],H0),templik)
            if distpost==False:
                # we remove distance squared prior from the samples
                templik /= dofz(catz[i],H0)**2
        else:
            templik = ncx2.pdf(rhosq,2,snr(dofz(catz[i],H0))**2)        
        num += templik*weight[i]
        den += weight[i]
        bas += templik   

    if basic==True:
        return bas
    else:
        #print num,den
        return num #np.squeeze(num/den)


def px_H0G_den_3D(H0,catz,catra,catdec,gmstrad):
    """
    the denominator for the likelihood of the GW data given the source is in the catalogue and detected and given H0
    """
    den = 0.0
    for i in xrange(catz.size):
        sfunc = survival_map(H0,catz[i],catra[i],catdec[i],gmstrad)
        den += sfunc
    return den


######## end construction #########
################################################


