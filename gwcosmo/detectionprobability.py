"""
Detection probability
Rachel Gray, John Veitch
"""
import lal
from   lal import ComputeDetAMResponse
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.stats import ncx2
import healpy as hp

import pkg_resources
"""
We want to create a function for $p(D|z,H_{0},I)$, so that when it is passed a value of $z$ and $H_0$,
a probability of detection is returned.  This involves marginalising over masses, inclination, polarisation, and sky location.
"""

class DetectionProbability(object):
    """
    Class to compute p(det | d_L, detectors, m1, m2, ...)
    TODO: Allow choices of distributions for intrinsic params
    """
    def __init__(self, m1_mean, m1_std, m2_mean, m2_std, detectors=['H1','L1'], psds=None, Nsamps=1000, snr_threshold=8):
        self.detectors = detectors
        self.snr_threshold = snr_threshold
        # TODO: find official place where PSDs are stored.
        if psds is not None:
            self.psds = psds
        else:
            PSD_path = pkg_resources.resource_filename('gwcosmo', 'data/other/PSD_L1_H1_mid.txt')
            PSD_data = np.genfromtxt(PSD_path)
            self.psds = interp1d(PSD_data[:,0],PSD_data[:,1])
        self.__lal_detectors = [lal.cached_detector_by_prefix[name] for name in detectors]
        self.Nsamps = Nsamps
        self.m1_mean = m1_mean
        self.m1_std = m1_std
        self.m2_mean = m2_mean
        self.m2_std = m2_std
        
        N=self.Nsamps
        self.RAs = np.random.rand(N)*2.0*np.pi
        r = np.random.rand(N)
        self.Decs = np.arcsin(2.0*r - 1.0)
        q = np.random.rand(N)
        self.incs = np.arcsin(2.0*q - 1.0)
        self.psis = np.random.rand(N)*2.0*np.pi
        self.m1 = np.random.normal(m1_mean,m1_std,N)*1.988e30
        self.m2 = np.random.normal(m2_mean,m2_std,N)*1.988e30
        self.M_min = np.min(self.m1)+np.min(self.m2)
        
    def __snr_squared(self,DL,RA,Dec,m1,m2,inc,psi,detector,gmst):
        """
        OBSOLETE?
        the optimal snr squared for one detector, marginalising over sky location, inclination, polarisation, mass
        """
        mtot = m1+m2
        mc = np.power(m1*m2,3.0/5.0)/np.power(m1+m2,1.0/5.0)
        Fplus,Fcross = lal.ComputeDetAMResponse(detector.response, RA, Dec, psi, gmst)
        A = np.sqrt(Fplus**2*(1.0+np.cos(inc)**2)**2 + Fcross**2*4.0*np.cos(inc)**2) \
        * np.sqrt(5.0*np.pi/96.0)*np.power(np.pi,-7.0/6.0) * np.power(mc,5.0/6.0) / (DL*lal.PC_SI*1e6)
        
        PSD = self.psds
        def I(f):
            return np.power(f,-7.0/3.0)/(PSD(f)**2)

        fmin = 10 # Hz
        fmax = 1/(np.power(6.0,3.0/2.0)*np.pi*mtot) * lal.C_SI**3/lal.G_SI # check units
        num = quad(I,fmin,fmax,epsabs=0,epsrel=1.49e-4)[0]
    
        return 4.0*A**2*num*np.power(lal.G_SI,5.0/3.0)/lal.C_SI**3.0
        
        
    def __snr_squared_new(self,RA,Dec,m1,m2,inc,psi,detector,gmst,interpolnum):
        """
        the optimal snr squared for one detector, marginalising over sky location, inclination, polarisation, mass
        """
        mtot = m1+m2
        mc = np.power(m1*m2,3.0/5.0)/np.power(m1+m2,1.0/5.0)
        Fplus,Fcross = lal.ComputeDetAMResponse(detector.response, RA, Dec, psi, gmst)
        A = np.sqrt(Fplus**2*(1.0+np.cos(inc)**2)**2 + Fcross**2*4.0*np.cos(inc)**2) \
        * np.sqrt(5.0*np.pi/96.0)*np.power(np.pi,-7.0/6.0) * np.power(mc,5.0/6.0)
        
        fmax = self.__fmax(mtot)
        num = interpolnum(fmax)
    
        return 4.0*A**2*num*np.power(lal.G_SI,5.0/3.0)/lal.C_SI**3.0

    def __numfmax_fmax(self,M_min):
        """
        what exactly does this do?
        """
        PSD = self.psds
        fmax = lambda m: self.__fmax(m)
        I = lambda f: np.power(f,-7.0/3.0)/(PSD(f)**2)
        f_min = 10
        f_max = fmax(M_min)

        arr_fmax = np.linspace(f_min, f_max, self.Nsamps)
        num_fmax = np.zeros(self.Nsamps)
        for i in range(self.Nsamps):
            num_fmax[i] = quad(I, f_min, arr_fmax[i],epsabs=0,epsrel=1.49e-4)[0]
        
        return interp1d(arr_fmax, num_fmax)
        
    def __fmax(self,m):
        """
        Maximum frequency for integration
        """
        return 1/(np.power(6.0,3.0/2.0)*np.pi*m) * lal.C_SI**3/lal.G_SI

    
    def pD_event(self, dl, ra, dec, m1, m2, inc, psi,gmst):
        """
        detection probability for a particular event (masses, distance, sky position and orientation)
        """
        rhosqs = [ self.__snr_squared(dl, ra, dec, m1, m2, inc, psi, det, gmst) for det in self.__lal_detectors]
        combined_rhosq = np.sum(rhosqs)
        effective_threshold = np.sqrt(len(self.detectors)) * self.snr_threshold
        return ncx2.sf(effective_threshold**2 , 4, combined_rhosq)
        
       
    def pD_dlradec(self,Nside,dl_array,dl,RA,Dec,gmst):
        """
        detection probability for a specific distance and sky position, marginalised over other parameters
        """
        no_pix = hp.pixelfunc.nside2npix(Nside)
        pix_ind = range(0,no_pix)
        theta,phi = hp.pixelfunc.pix2ang(Nside,pix_ind)
        RA_map = phi
        Dec_map = np.pi/2.0 - theta
        interpolnum = self.__numfmax_fmax(self.M_min)
        
        pofd_dLRADec = np.zeros([no_pix,dl_array.size])
        for k in range(no_pix):
            rho = np.zeros((self.Nsamps,1))
            for n in range(self.Nsamps):
                rhosqs = [ self.__snr_squared_new(RA_map[k],Dec_map[k],self.m1[n],self.m2[n],self.incs[n],self.psis[n], det, 0.0,interpolnum) for det in self.__lal_detectors]
                rho[n] = np.sum(rhosqs)

            DLcopy = dl_array.reshape((dl_array.size, 1))
            DLcopy = DLcopy.transpose()
            DLcopy = 1/(DLcopy*lal.PC_SI*1e6)**2

            effective_threshold = np.sqrt(len(self.detectors)) * self.snr_threshold
            survival = np.matmul(rho, DLcopy)
            survival = ncx2.sf(effective_threshold**2,4,survival)

            pofd_dLRADec[k,:] = np.sum(survival,0)/self.Nsamps
        
        # TODO: figure out how to precompute this object so it can be called in a useful manner.   
        survival_func_sky = interp1d(dl_array,pofd_dLRADec,bounds_error=False,fill_value=1e-10)
        hpxmap = survival_func_sky(dl)[pix_ind]
        return hp.get_interp_val(hpxmap,np.pi/2.0-Dec,RA-gmst)
        
    #def pD_dlradec_map(self,Nside,dl_array,nSamples,dl,ra,dec,gmst):
    #    """
    #    detection probability evaluated at a specific dl,ra,dec and gmst.
    #    """
    #    sfunc = self.pD__dlradec_new(Nside,dl_array,nSamples):
    #    no_pix = hp.pixelfunc.nside2npix(Nside)
    #    pix_ind = range(0,no_pix)
    #    hpxmap = survival_func_sky(dl)[pix_ind]
    #    return hp.get_interp_val(hpxmap,np.pi/2.0-Dec,RA-gmst)
        
        
    def pD_dl_single(self, dl):
        """
        Detection probability for a specific distance, averaged over all other parameters
        """
        # samples for monte carlo integral
        N=self.Nsamps
        RAs = np.random.rand(N)*2.0*np.pi
        r = np.random.rand(N)
        Decs = np.arcsin(2.0*r - 1.0)
        q = np.random.rand(N)
        incs = np.arcsin(2.0*q - 1.0)
        psis = np.random.rand(N)*2.0*np.pi
        m1 = np.random.normal(1.35,0.1,N)*1.988e30
        m2 = np.random.normal(1.35,0.1,N)*1.988e30
        
        return np.mean(
            [ self.pD_event(dl, RAs[i], Decs[i], m1[i], m2[i], incs[i], psis[i], 0.0) for i in range(N)]
            )
            
    def pD_dl(self,dl_array):
        """
        OBSOLETE?
        Detection probability over a range of distances, returned as an interpolated function.
        """
        prob = np.zeros(len(dl_array))
        for i in range(len(dl_array)):
            prob[i] = self.pD_dl_single(dl_array[i])
        
        return interp1d(dl_array,prob,bounds_error=False,fill_value=1e-10)

    def pD_dl_new(self,dl_array):
        """
        Detection probability over a range of distances, returned as an interpolated function.
        """
        interpolnum = self.__numfmax_fmax(self.M_min)
        rho = np.zeros((self.Nsamps,1))
        for n in range(self.Nsamps):
            rhosqs = [ self.__snr_squared_new(self.RAs[n],self.Decs[n],self.m1[n],self.m2[n],self.incs[n],self.psis[n], det, 0.0,interpolnum) for det in self.__lal_detectors]
            rho[n] = np.sum(rhosqs)

        DLcopy = dl_array.reshape((dl_array.size, 1))
        DLcopy = DLcopy.transpose()
        DLcopy = 1/(DLcopy*lal.PC_SI*1e6)**2

        effective_threshold = np.sqrt(len(self.detectors)) * self.snr_threshold
        survival = np.matmul(rho, DLcopy)
        survival = ncx2.sf(effective_threshold**2,4,survival)

        prob = np.sum(survival,0)/self.Nsamps
        
        return interp1d(dl_array,prob,bounds_error=False,fill_value=1e-10)

   
    def __call__(self, dl):
        """
        To call as function of dl
        """
        return self.pD_dl_single(dl)
    
