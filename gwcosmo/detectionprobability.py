"""
Detection probability
Rachel Gray, John Veitch
"""
import lal
from   lal import ComputeDetAMResponse
import numpy as np
from scipy.interpolate import interp1d
import pkg_resources
"""
We want to create a function for $p(D|z,H_{0},I)$, so that when it is passed a value of $z$ and $H_0$,
a probability of detection is returned.  This involves marginalising over neutron star masses, inclination, polarisation, and sky location.
"""

class DetectionProbability(object):
    """
    Class to compute p(det | d_L, detectors, m1, m2, ...)
    TODO: Allow choices of distributions for intrinsic params
    """
    def __init__(self, m1, m2, detectors=['H1','L1'], psds=None, Nsamps=5000, snr_threshold=8):
        self.detectors = detectors
        self.snr_threshold = snr_threshold
        if psds is not None:
            self.psds = psds
        else:
            PSD_path = pkg_resources.resource_filename('gwcosmo', 'data/other/PSD_L1_H1_mid.txt')
            PSD_data = np.genfromtxt(PSD_path)
            self.psds = interp1d(PSD_data[:,0],PSD_data[:,1])
        self.m1 = m1
        self.m2 = m2
        self.mtot = m1+m2
        self.mc = np.power(m1*m2,3.0/5.0)/np.power(m1+m2,1.0/5.0)
        self.__lal_detectors = [lal.cached_detector_by_prefix[name] for name in detectors]
        self.Nsamps = Nsamps
        
    def __snr_squared(self,DL,RA,Dec,inc,psi,detector,gmst):
        """
        the optimal snr squared for one detector, marginalising over sky location, inclination, polarisation, mass
        """
        Fplus,Fcross = lal.ComputeDetAMResponse(detector.response, RA, Dec, psi, gmst)
        A = np.sqrt(Fplus**2*(1.0+np.cos(inc)**2)**2 + Fcross**2*4.0*np.cos(inc)**2) \
        * np.sqrt(5.0*np.pi/96.0)*np.power(np.pi,-7.0/6.0) * np.power(self.mc,5.0/6.0) / (DL*lal.PC_SI*1e6)
        
        def I(f):
            return np.power(f,-7.0/3.0)/(PSD(f)**2)

        fmin = 10 # Hz
        fmax = 1/(np.power(6.0,3.0/2.0)*np.pi*self.mtot) * lal.C_SI**3/lal.G_SI # check units
        num = quad(I,fmin,fmax,epsabs=0,epsrel=1.49e-4)[0]
    
        return 4.0*A**2*num*np.power(lal.G_SI,5.0/3.0)/lal.C_SI**3.0
    
    def p_D_positional(self, dl, ra, dec, inc, psi):
        """
        detection probability for a particular sky position and orientation
        """
        gmst = 0 # Set to zero as we will average over sky
        rhosqs = [ self.__snr_squared(dl, ra, dec, inc, psi, det, gmst) for det in self.__lal_detectors]
        combined_rhosq = np.sum(rhosqs)
        effective_threshold = np.sqrt(len(self.detectors)) * self.snr_threshold
        return ncx2.sf(effective_threshold**2 , combined_rhosq)

    def p_D(self, dl):
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
        
        return np.mean(
            [ self.p_D_positional(dl, RAs[i], Decs[i], incs[i], psis[i]) for i in range(N)]
            )
    
    def __call__(self, dl):
        """
        To call as function of dl
        """
        return self.p_D(dl)
    
