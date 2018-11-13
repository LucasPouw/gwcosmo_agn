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
    def __init__(self, mass_distribution, detectors=['H1','L1'], psds=None, Nsamps=1000, snr_threshold=8, Nside=None):
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
        self.Nside = Nside
        self.mass_distribution = mass_distribution
        self.H0 = H0
        
        # set up the samples for monte carlo integral
        N=self.Nsamps
        self.RAs = np.random.rand(N)*2.0*np.pi
        r = np.random.rand(N)
        self.Decs = np.arcsin(2.0*r - 1.0)
        q = np.random.rand(N)
        self.incs = np.arcsin(2.0*q - 1.0)
        self.psis = np.random.rand(N)*2.0*np.pi
        self.zeds = 
        if self.mass_distribution == 'BNS':
            self.dl_array = np.linspace(0.1,400,100)
            self.m1 = np.random.normal(1.35,0.1,N)*1.988e30
            self.m2 = np.random.normal(1.35,0.1,N)*1.988e30
        if self.mass_distribution == 'BBH':
            self.dl_array = np.linspace(0.1,3000,100)
            #Based on Maya's notebook
            def inv_cumulative_power_law(u,mmin,mmax,alpha):
                if alpha != -1:
                    return (u*(mmax**(alpha+1)-mmin**(alpha+1))+mmin**(alpha+1))**(1.0/(alpha+1))
                else:
                    return np.exp(u*(np.log(mmax)-np.log(mmin))+np.log(mmin))
            self.m1 = inv_cumulative_power_law(np.random.rand(N),5.,40.,-1.)*1.988e30
            self.m2 = np.random.uniform(low=5.0,high=self.m1)
        self.M_min = np.min(self.m1)+np.min(self.m2)
        
        self.__interpolnum = self.__numfmax_fmax(self.M_min)
        
        # precompute values which will be called multiple times
        self.interp_average = self.__pD_dl(self.dl_array)
        if Nside != None:
            self.interp_map = self.__pD_dlradec(self.Nside,self.dl_array)
    
    
    def mchirp(self,m1,m2):
        """
        Calculates the source chirp mass
        
        Parameters
        ----------
        m1,m2 : source rest frame masses in kg
        
        Returns
        -------
        Source chirp mass in kg
        """
        return np.power(m1*m2,3.0/5.0)/np.power(m1+m2,1.0/5.0)
        
        
    def mchirp_obs(self,m1,m2,z=0):
        """
        Calculates the redshifted chirp mass from source masses and a redshift
        
        Parameters
        ----------
        m1,m2 : source rest frame masses in kg
        z : redshift (default=0)
        
        Returns
        -------
        Redshifted chirp mass in kg
        """
        return (1+z)*self.mchirp(m1,m2)
                
        
    def __mtot(self,m1,m2):
        """
        Calculates the total source mass of the system
        
        Parameters
        ----------
        m1,m2 : source rest frame masses in kg
        
        Returns
        -------
        Source total mass in kg        
        """
        return m1+m2
        
        
    def __mtot_obs(self,m1,m2,z=0):
        """
        Calculates the total observed mass of the system
        
        Parameters
        ----------
        m1,m2 : source rest frame masses in kg
        z : redshift (default=0)
        
        Returns
        -------
        Observed total mass in kg        
        """
        return (m1+m2)*(1+z)
        
        
    def __Fplus(self,detector, RA, Dec, psi, gmst):
        """
        Computes the 'plus' antenna pattern
        
        Parameters
        ----------
        detector : name of detector in network as a string (eg 'H1', 'L1')
        RA,Dec : sky location of the event in radians
        psi : source polarisation in radians
        gmst : Greenwich Mean Sidereal Time in seconds CHECK THIS
        
        Returns
        -------
        F_+ antenna response
        """
        return lal.ComputeDetAMResponse(detector.response, RA, Dec, psi, gmst)[0]
        
        
    def __Fcross(self,detector, RA, Dec, psi, gmst):
        """
        Computes the 'plus' antenna pattern
        
        Parameters
        ----------
        detector : name of detector in network as a string (eg 'H1', 'L1')
        RA,Dec : sky location of the event in radians
        psi : source polarisation in radians
        gmst : Greenwich Mean Sidereal Time in seconds
        
        Returns
        -------
        F_x antenna response
        """
        return lal.ComputeDetAMResponse(detector.response, RA, Dec, psi, gmst)[1]
        
        
    def __reduced_amplitude(self,RA,Dec,inc,psi,detector,gmst):
        """
        Component of the Fourier amplitude, with redshift-dependent parts removed
        
        Parameters
        ----------
        RA,Dec : sky location of the event in radians
        inc : source inclination in radians
        psi : source polarisation in radians
        detector : name of detector in network as a string (eg 'H1', 'L1')
        gmst : Greenwich Mean Sidereal Time in seconds 
        
        Returns
        -------
        [F+^2*(1+cos(i)^2)^2 + Fx^2*4*cos(i)^2]^1/2 * [5*pi/96]^1/2 * pi^-7/6
        """
        Fplus = self.__Fplus(detector, RA, Dec, psi, gmst)
        Fcross = self.__Fcross(detector, RA, Dec, psi, gmst)
        return np.sqrt(Fplus**2*(1.0+np.cos(inc)**2)**2 + Fcross**2*4.0*np.cos(inc)**2) \
        * np.sqrt(5.0*np.pi/96.0)*np.power(np.pi,-7.0/6.0)
        
        
    def __fmax(self,M):
        """
        Maximum frequency for integration, set by the frequency of the innermost stable orbit (ISO)
        fmax(M) 2*f_ISO = (6^(3/2)*pi*M)^-1
        
        Parameters
        ----------
        M : total mass of the system in kg
        
        Returns
        -------
        Maximum frequency in Hz
        """
        return 1/(np.power(6.0,3.0/2.0)*np.pi*M) * lal.C_SI**3/lal.G_SI
        
        
    def __numfmax_fmax(self,M_min):
        """
        lookup table for snr as a function of max frequency
        Calculates \int_fmin^fmax f'^(-7/3)/S_h(f') df over a range of values for fmax
        fmin: 10 Hz
        fmax(M): (6^(3/2)*pi*M)^-1
        and fmax varies from fmin to fmax(M_min)
        
        Parameters
        ----------
        M_min : total minimum mass of the distribution in kg
        
        Returns
        -------
        Interpolated 1D array of \int_fmin^fmax f'^(-7/3)/S_h(f') for different fmax's
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

        
    def __snr_squared(self,RA,Dec,m1,m2,inc,psi,detector,gmst,z=0):
        """
        the optimal snr squared for one detector, used for marginalising over sky location, inclination, polarisation, mass
        
        Parameters
        ----------
        RA,Dec : sky location of the event in radians
        m1,m2 : source masses in kg
        inc : source inclination in radians
        psi : source polarisation in radians
        detector : name of detector in network as a string (eg 'H1', 'L1')
        gmst : Greenwich Mean Sidereal Time in seconds
        
        Returns
        -------
        snr squared*dL^2 for given parameters at a single detector
        """
        mtot = self.__mtot_obs(m1,m2,z)
        mc = self.mchirp_obs(m1,m2,z)
        A = self.__reduced_amplitude(RA,Dec,inc,psi,detector,gmst) * np.power(mc,5.0/6.0) (DL*lal.PC_SI*1e6)
        
        fmax = self.__fmax(mtot)
        num = self.__interpolnum(fmax)
    
        return 4.0*A**2*num*np.power(lal.G_SI,5.0/3.0)/lal.C_SI**3.0

        
        
    def __pD_dl(self,dl_array):
        """
        Detection probability over a range of distances, returned as an interpolated function.
        """
        rho = np.zeros((self.Nsamps,1))
        for n in range(self.Nsamps):
            rhosqs = [ self.__snr_squared(self.RAs[n],self.Decs[n],self.m1[n],self.m2[n],self.incs[n],self.psis[n], det, 0.0) for det in self.__lal_detectors]
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
        
        
    def __snr_squared_old(self,RA,Dec,m1,m2,inc,psi,detector,gmst,z=0):
        """
        the optimal snr squared for one detector, used for marginalising over sky location, inclination, polarisation, mass
        
        Parameters
        ----------
        RA,Dec : sky location of the event in radians
        m1,m2 : source masses in kg
        inc : source inclination in radians
        psi : source polarisation in radians
        detector : name of detector in network as a string (eg 'H1', 'L1')
        gmst : Greenwich Mean Sidereal Time in seconds
        
        Returns
        -------
        snr squared*dL^2 for given parameters at a single detector
        """
        mtot = self.__mtot_obs(m1,m2,z)
        mc = self.mchirp_obs(m1,m2,z)
        A = self.__reduced_amplitude(RA,Dec,inc,psi,detector,gmst) * np.power(mc,5.0/6.0)
        
        fmax = self.__fmax(mtot)
        num = self.__interpolnum(fmax)
    
        return 4.0*A**2*num*np.power(lal.G_SI,5.0/3.0)/lal.C_SI**3.0

        
        
    def __pD_dl_old(self,dl_array):
        """
        Detection probability over a range of distances, returned as an interpolated function.
        """
        rho = np.zeros((self.Nsamps,1))
        for n in range(self.Nsamps):
            rhosqs = [ self.__snr_squared_old(self.RAs[n],self.Decs[n],self.m1[n],self.m2[n],self.incs[n],self.psis[n], det, 0.0) for det in self.__lal_detectors]
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
        
        #pofd_dLRADec = np.zeros([no_pix,dl_array.size])
        pofd_dLRADec = []
        for k in range(no_pix):
            rho = np.zeros((self.Nsamps,1))
            for n in range(self.Nsamps):
                rhosqs = [ self.__snr_squared(RA_map[k],Dec_map[k],self.m1[n],self.m2[n],self.incs[n],self.psis[n], det, 0.0) for det in self.__lal_detectors]
                rho[n] = np.sum(rhosqs)

            DLcopy = dl_array.reshape((dl_array.size, 1))
            DLcopy = DLcopy.transpose()
            DLcopy = 1/(DLcopy*lal.PC_SI*1e6)**2

            effective_threshold = np.sqrt(len(self.detectors)) * self.snr_threshold
            survival = np.matmul(rho, DLcopy)
            survival = ncx2.sf(effective_threshold**2,4,survival)

            norm = np.sum(survival,0)/self.Nsamps
            pofd_dLRADec.append(splrep(dl_array,norm))

        return pofd_dLRADec
        

    def snr_squared_single(self,DL,RA,Dec,m1,m2,inc,psi,detector,gmst):
        """
        UNUSED
        the optimal snr squared for one detector, for a specific DL, RA, Dec, m1, m2, inc, psi, gmst
        """
        mtot = m1+m2
        mc = self.mchirp_obs(m1,m2)
        Fplus,Fcross = lal.ComputeDetAMResponse(detector.response, RA, Dec, psi, gmst)
        A = self.__reduced_amplitude(RA,Dec,inc,psi,detector,gmst) * np.power(mc,5.0/6.0) / (DL*lal.PC_SI*1e6)
        
        PSD = self.psds
        def I(f):
            return np.power(f,-7.0/3.0)/(PSD(f)**2)

        fmin = 10 # Hz
        fmax = self.__fmax(mtot)
        num = quad(I,fmin,fmax,epsabs=0,epsrel=1.49e-4)[0]
    
        return 4.0*A**2*num*np.power(lal.G_SI,5.0/3.0)/lal.C_SI**3.0


    def pD_event(self, dl, ra, dec, m1, m2, inc, psi, gmst):
        """
        detection probability for a particular event (masses, distance, sky position, orientation and time)
        """
        rhosqs = [ self.snr_squared_single(dl, ra, dec, m1, m2, inc, psi, det, gmst) for det in self.__lal_detectors]
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
            

    def pD_dl_eval(self,dl):
        """
        Returns a probability for a given distance dl from the interpolated function.
        Or an array of probabilities for an array of distances.
        """
        return splev(dl,self.interp_average,ext=3)
        
        
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
        return self.pD_dl_eval(dl)
    
    
    def pD_distmax(self):
        """
        Returns twice the maximum distance given Pdet(dl) = 0.01.
        """
        return 2.*self.dl_array[np.where(self.pD_dl_eval(self.dl_array)>0.01)[0][-1]]
        

