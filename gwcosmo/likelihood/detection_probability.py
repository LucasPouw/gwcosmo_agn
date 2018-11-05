"""
Detection probability
Rachel Gray, John Veitch, Ignacio Magana
"""
import lal
from   lal import ComputeDetAMResponse
import numpy as np
from scipy.interpolate import interp1d,splev,splrep
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
    def __init__(self, mass_distribution, dl_array, detectors=['H1','L1'], psds=None, Nsamps=1000, snr_threshold=8, Nside=8):
        self.detectors = detectors
        self.snr_threshold = snr_threshold
        # TODO: find official place where PSDs are stored, and link to specific detectors/observing runs
        # Also techically ASDs - rename
        if psds is not None:
            self.psds = psds
        else:
            PSD_path = pkg_resources.resource_filename('gwcosmo', 'data/other/PSD_L1_H1_mid.txt')
            PSD_data = np.genfromtxt(PSD_path)
            self.psds = interp1d(PSD_data[:,0],PSD_data[:,1])
        self.__lal_detectors = [lal.cached_detector_by_prefix[name] for name in detectors]
        self.Nsamps = Nsamps
        self.dl_array = dl_array
        self.Nside = Nside
        self.mass_distribution = mass_distribution
        
        # set up the samples for monte carlo integral
        N=self.Nsamps
        self.RAs = np.random.rand(N)*2.0*np.pi
        r = np.random.rand(N)
        self.Decs = np.arcsin(2.0*r - 1.0)
        q = np.random.rand(N)
        self.incs = np.arcsin(2.0*q - 1.0)
        self.psis = np.random.rand(N)*2.0*np.pi
        if self.mass_distribution == 'BNS':
            self.m1 = np.random.normal(1.35,0.1,N)*1.988e30
            self.m2 = np.random.normal(1.35,0.1,N)*1.988e30
            self.M_min = np.min(self.m1)+np.min(self.m2)
        if self.mass_distribution == 'BBH':
            #Based on Maya's notebook
            def inv_cumulative_power_law(u,mmin,mmax,alpha):
                if alpha != -1:
                    return (u*(mmax**(alpha+1)-mmin**(alpha+1))+mmin**(alpha+1))**(1.0/(alpha+1))
                else:
                    return np.exp(u*(np.log(mmax)-np.log(mmin))+np.log(mmin))
            self.m1 = inv_cumulative_power_law(np.random.rand(N),5.,40.,-1.)*1.988e30
            self.m2 = np.random.uniform(low=5.0,high=self.m1)
            self.M_min = np.min(self.m1)+np.min(self.m2)
        # precompute values which will be called multiple times
        self.interp_average = self.__pD_dl(self.dl_array)
        self.interp_map = self.__pD_dlradec(self.Nside,self.dl_array)
       
        
    def __snr_squared_single(self,DL,RA,Dec,m1,m2,inc,psi,detector,gmst):
        """
        the optimal snr squared for one detector, for a specific DL, RA, Dec, m1, m2, inc, psi, gmst
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
        fmax = self.__fmax(mtot)
        num = quad(I,fmin,fmax,epsabs=0,epsrel=1.49e-4)[0]
    
        return 4.0*A**2*num*np.power(lal.G_SI,5.0/3.0)/lal.C_SI**3.0
        
        
    def __snr_squared(self,RA,Dec,m1,m2,inc,psi,detector,gmst,interpolnum):
        """
        the optimal snr squared for one detector, used for marginalising over sky location, inclination, polarisation, mass
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
        lookup table for snr as a function of max frequency
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
        
        
    def __pD_dl(self,dl_array):
        """
        Detection probability over a range of distances, returned as an interpolated function.
        """
        interpolnum = self.__numfmax_fmax(self.M_min)
        rho = np.zeros((self.Nsamps,1))
        for n in range(self.Nsamps):
            rhosqs = [ self.__snr_squared(self.RAs[n],self.Decs[n],self.m1[n],self.m2[n],self.incs[n],self.psis[n], det, 0.0,interpolnum) for det in self.__lal_detectors]
            rho[n] = np.sum(rhosqs)

        DLcopy = dl_array.reshape((dl_array.size, 1))
        DLcopy = DLcopy.transpose()
        DLcopy = 1/(DLcopy*lal.PC_SI*1e6)**2

        effective_threshold = np.sqrt(len(self.detectors)) * self.snr_threshold
        survival = np.matmul(rho, DLcopy)
        survival = ncx2.sf(effective_threshold**2,4,survival)

        prob = np.sum(survival,0)/self.Nsamps
        self.spl = splrep(dl_array,prob)
        return splrep(dl_array,prob)
        
    
    def __pD_dlradec(self,Nside,dl_array):
        """
        Detection probability over a range of distances, at each pixel on a healpy map.
        """
        no_pix = hp.pixelfunc.nside2npix(Nside)
        pix_ind = range(0,no_pix)
        theta,phi = hp.pixelfunc.pix2ang(Nside,pix_ind)
        RA_map = phi
        Dec_map = np.pi/2.0 - theta
        interpolnum = self.__numfmax_fmax(self.M_min)
        
        #pofd_dLRADec = np.zeros([no_pix,dl_array.size])
        pofd_dLRADec = []
        for k in range(no_pix):
            rho = np.zeros((self.Nsamps,1))
            for n in range(self.Nsamps):
                rhosqs = [ self.__snr_squared(RA_map[k],Dec_map[k],self.m1[n],self.m2[n],self.incs[n],self.psis[n], det, 0.0,interpolnum) for det in self.__lal_detectors]
                rho[n] = np.sum(rhosqs)

            DLcopy = dl_array.reshape((dl_array.size, 1))
            DLcopy = DLcopy.transpose()
            DLcopy = 1/(DLcopy*lal.PC_SI*1e6)**2

            effective_threshold = np.sqrt(len(self.detectors)) * self.snr_threshold
            survival = np.matmul(rho, DLcopy)
            survival = ncx2.sf(effective_threshold**2,4,survival)

            norm = np.sum(survival,0)/self.Nsamps
            pofd_dLRADec.append(splrep(dl_array,norm))

        #return interp1d(dl_array,pofd_dLRADec,bounds_error=False,fill_value=1e-10)
        return pofd_dLRADec
        

    def pD_event(self, dl, ra, dec, m1, m2, inc, psi, gmst):
        """
        detection probability for a particular event (masses, distance, sky position, orientation and time)
        """
        rhosqs = [ self.__snr_squared_single(dl, ra, dec, m1, m2, inc, psi, det, gmst) for det in self.__lal_detectors]
        combined_rhosq = np.sum(rhosqs)
        effective_threshold = np.sqrt(len(self.detectors)) * self.snr_threshold
        return ncx2.sf(effective_threshold**2 , 4, combined_rhosq)
        

    def pD_dl_single(self, dl):
        """
        OBSOLETE?
        Detection probability for a specific distance, averaged over all other parameters - without using interpolation
        """       
        return np.mean(
            [ self.pD_event(dl, self.RAs[i], self.Decs[i], self.m1[i], self.m2[i], self.incs[i], self.psis[i], 0.0) for i in range(N)]
            )
            
        
#    def pD_dl_eval(self,dl):
#        """
#        Returns a probability for a given distance dl from the interpolated function.
#        Or an array of probabilities for an array of distances.
#        """
#        return self.interp_dist(dl)


    def pD_dl_eval(self,dl,spl):
        """
        Returns a probability for a given distance dl from the interpolated function.
        Or an array of probabilities for an array of distances.
        """
        return splev(dl,spl,ext=3)
        
        
    def pD_dlradec_eval(self,dl,RA,Dec,gmst):
        """
        OBSOLETE?
        detection probability evaluated at a specific dl,ra,dec and gmst.
        """
        if all(self.interp_map) == None:
            self.interp_map = self.__pD_dlradec(self.Nside,self.dl_array)
        no_pix = hp.pixelfunc.nside2npix(self.Nside)
        pix_ind = range(0,no_pix)
        survival_func_sky = self.interp_map
        hpxmap = survival_func_sky(dl)[pix_ind]
        return hp.get_interp_val(hpxmap,np.pi/2.0-Dec,RA-gmst)
        
        
    def pDdl_radec(self,RA,Dec,gmst):
        """
        Returns the probability of detection function for a specific ra, dec, and time.
        """
        ipix = hp.ang2pix(self.Nside,np.pi/2.0-Dec,RA-gmst)
        return self.interp_map[ipix]
        
        
    def __call__(self, dl):
        """
        To call as function of dl
        """
        return self.pD_dl_eval(dl,self.interp_average)
    
    
    def pD_distmax(self):
        """
        Returns twice the maximum distance given Pdet(dl) = 0.01.
        """
        return 2.*self.dl_array[np.where(self(self.dl_array)>0.01)[0][-1]]
