"""
Detection probability
Rachel Gray, John Veitch, Ignacio Magana
"""
#from __future__ import absolute_import
import lal
from   lal import ComputeDetAMResponse
import numpy as np
from scipy.interpolate import interp1d,splev,splrep,interp2d
from scipy.integrate import quad
from scipy.stats import ncx2
import healpy as hp
from gwcosmo.utilities.standard_cosmology import *
import pickle

import pkg_resources
import os
"""
We want to create a function for $p(D|z,H_{0},I)$, so that when it is passed a value of $z$ and $H_0$,
a probability of detection is returned.  This involves marginalising over masses, inclination, polarisation, and sky location.
"""

class DetectionProbability(object):
    """
    Class to compute p(det | d_L, detectors, m1, m2, ...)
    
    Parameters
    ----------
    mass_distribution : str
        choice of mass distribution ('BNS' or 'BBH')
    detectors : list of str, optional
        list of detector names (default=['H1','L1'])
    psds : str, optional
        path to file containing the relevant PSDs (default=None)
    Nsamps : int, optional
        Number of samples for monte carlo integration (default=1000)
    snr_theshold : float, optional
        snr threshold for an individual detector (default=8)
    Nside : int, optional
        If using variable pdet across the sky, specify the resolution of the healpy map
    Omega_m : float, optional
        matter fraction of the universe (default=0.3)
    linear : bool, optional
        if True, use linear cosmology (default=False)
    """
    def __init__(self, mass_distribution, detectors=['H1','L1'], psds=None, Nsamps=1000, snr_threshold=8, Nside=None, Omega_m=0.3, linear=False):
        self.detectors = detectors
        self.snr_threshold = snr_threshold
        # TODO: find official place where PSDs are stored, and link to specific detectors/observing runs
        # Also techically ASDs - rename
        data_path = pkg_resources.resource_filename('gwcosmo', 'data/other/')
        if psds is not None:
            self.psds = psds
        else:
            PSD_data = np.genfromtxt(data_path + 'PSD_L1_H1_mid.txt')
            self.psds = interp1d(PSD_data[:,0],PSD_data[:,1])
        self.__lal_detectors = [lal.cached_detector_by_prefix[name] for name in detectors]
        self.Nsamps = Nsamps
        self.Nside = Nside
        self.mass_distribution = mass_distribution
        self.Omega_m = Omega_m
        self.linear = linear
        self.H0vec = np.linspace(10,200,50)
        self.cosmo = fast_cosmology(Omega_m=self.Omega_m,linear=self.linear)
        # TODO: For higher values of z (z=10) this goes outside the range of the psds and gives an error
        self.z_array = np.logspace(-4.0,0.5,50)
        
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
            interp_av_path = data_path + 'BNS_z_H0_pD_array.p'
            self.dl_array = np.linspace(1.0e-100,400.0,500)
        if self.mass_distribution == 'BBH':
            #Based on Maya's notebook
            def inv_cumulative_power_law(u,mmin,mmax,alpha):
                if alpha != -1:
                    return (u*(mmax**(alpha+1)-mmin**(alpha+1))+mmin**(alpha+1))**(1.0/(alpha+1))
                else:
                    return np.exp(u*(np.log(mmax)-np.log(mmin))+np.log(mmin))
            self.m1 = inv_cumulative_power_law(np.random.rand(N),5.,40.,-1.)*1.988e30
            self.m2 = np.random.uniform(low=5.0,high=self.m1)
            interp_av_path = data_path + 'BBH_z_H0_pD_array.p'
            self.dl_array = np.linspace(1.0e-100,2500.0,500)
            
        self.M_min = np.min(self.m1)+np.min(self.m2)
        
        # precompute values which will be called multiple times, if not precomputed
        if os.path.isfile(interp_av_path):
            z,H0,prob = pickle.load(open(interp_av_path,'rb'))
            self.__interpolnum = self.__numfmax_fmax(self.M_min)
        else:
            z,H0,prob = self.__pD_zH0_array(self.H0vec)
        # TODO: test how different interpolations and fill values effect results.  Do values go below 0 and above 1?
        self.interp_average = interp2d(z,H0,prob,kind='cubic')
        self.interp_average_basic = self.__pD_dl_basic(self.dl_array)
        if Nside != None:
            self.interp_map = self.__pD_dlradec(self.Nside,self.dl_array)


    def mchirp(self,m1,m2):
        """
        Calculates the source chirp mass
        
        Parameters
        ----------
        m1,m2 : float
            source rest frame masses in kg
        
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
        m1,m2 : float
            source rest frame masses in kg
        z : float
            redshift (default=0)
        
        Returns
        -------
        float
            Redshifted chirp mass in kg
        """
        return (1+z)*self.mchirp(m1,m2)
                
        
    def __mtot(self,m1,m2):
        """
        Calculates the total source mass of the system
        
        Parameters
        ----------
        m1,m2 : float
            source rest frame masses in kg
        
        Returns
        -------
        float
            Source total mass in kg        
        """
        return m1+m2
        
        
    def __mtot_obs(self,m1,m2,z=0):
        """
        Calculates the total observed mass of the system
        
        Parameters
        ----------
        m1,m2 : float
            source rest frame masses in kg
        z : float
            redshift (default=0)
        
        Returns
        -------
        float
            Observed total mass in kg        
        """
        return (m1+m2)*(1+z)
        
        
    def __Fplus(self,detector, RA, Dec, psi, gmst):
        """
        Computes the 'plus' antenna pattern
        
        Parameters
        ----------
        detector : str
            name of detector in network (eg 'H1', 'L1')
        RA,Dec : float
            sky location of the event in radians
        psi : float
            source polarisation in radians
        gmst : float
            Greenwich Mean Sidereal Time in seconds
        
        Returns
        -------
        float
            F_+ antenna response
        """
        return lal.ComputeDetAMResponse(detector.response, RA, Dec, psi, gmst)[0]
        
        
    def __Fcross(self,detector, RA, Dec, psi, gmst):
        """
        Computes the 'plus' antenna pattern
        
        Parameters
        ----------
        detector : str
            name of detector in network (eg 'H1', 'L1')
        RA,Dec : float
            sky location of the event in radians
        psi : float
            source polarisation in radians
        gmst : float
            Greenwich Mean Sidereal Time in seconds
        
        Returns
        -------
        float
            F_x antenna response
        """
        return lal.ComputeDetAMResponse(detector.response, RA, Dec, psi, gmst)[1]
        
        
    def __reduced_amplitude(self,RA,Dec,inc,psi,detector,gmst):
        """
        Component of the Fourier amplitude, with redshift-dependent parts removed
        computes: [F+^2*(1+cos(i)^2)^2 + Fx^2*4*cos(i)^2]^1/2 * [5*pi/96]^1/2 * pi^-7/6
        
        Parameters
        ----------
        RA,Dec : float
            sky location of the event in radians
        inc : float
            source inclination in radians
        psi : float
            source polarisation in radians
        detector : str
            name of detector in network as a string (eg 'H1', 'L1')
        gmst : float
            Greenwich Mean Sidereal Time in seconds 
        
        Returns
        -------
        float
            Component of the Fourier amplitude, with redshift-dependent parts removed
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
        M : float
            total mass of the system in kg
        
        Returns
        -------
        float
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
        M_min : float
            total minimum mass of the distribution in kg
        
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

        
    def __snr_squared(self,RA,Dec,m1,m2,inc,psi,detector,gmst,z,H0):
        """
        the optimal snr squared for one detector, used for marginalising over sky location, inclination, polarisation, mass
        
        Parameters
        ----------
        RA,Dec : float
            sky location of the event in radians
        m1,m2 : float
            source masses in kg
        inc : float
            source inclination in radians
        psi : float
            source polarisation in radians
        detector : str
            name of detector in network as a string (eg 'H1', 'L1')
        gmst : float
            Greenwich Mean Sidereal Time in seconds
                    
        Returns
        -------
        float
            snr squared*dL^2 for given parameters at a single detector
        """
        mtot = self.__mtot_obs(m1,m2,z)
        mc = self.mchirp_obs(m1,m2,z)
        A = self.__reduced_amplitude(RA,Dec,inc,psi,detector,gmst) * np.power(mc,5.0/6.0)/ (self.cosmo.dl_zH0(z,H0)*lal.PC_SI*1e6)
        
        fmax = self.__fmax(mtot)
        num = self.__interpolnum(fmax)
    
        return 4.0*A**2*num*np.power(lal.G_SI,5.0/3.0)/lal.C_SI**3.0


    def __pD_zH0(self,H0):
        """
        Detection probability over a range of distances, returned as an interpolated function.
        
        Parameters
        ----------
        H0 : float
            value of Hubble constant in kms-1Mpc-1
        
        Returns
        -------
        interpolated probabilities of detection over an array of luminosity distances, for a specific value of H0
        """
        rho = np.zeros((self.Nsamps,1))
        prob = np.zeros(len(self.z_array))
        for i in range(len(self.z_array)):
            for n in range(self.Nsamps):
                rhosqs = [ self.__snr_squared(self.RAs[n],self.Decs[n],self.m1[n],self.m2[n],self.incs[n],self.psis[n], det, 0.0, self.z_array[i],H0) for det in self.__lal_detectors]
                rho[n] = np.sum(rhosqs)

            effective_threshold = np.sqrt(len(self.detectors)) * self.snr_threshold
            survival = ncx2.sf(effective_threshold**2,4,rho)
            prob[i] = np.sum(survival,0)/self.Nsamps
        
        return prob
        
        
    def __pD_zH0_array(self,H0vec):
        """
        Function which calculates p(D|z,H0) for a range of redshift and H0 values
        
        Parameters
        ----------
        H0vec : array_like
            array of H0 values in kms-1Mpc-1
        
        Returns
        -------
        list of arrays?
            redshift, H0 values, and the corresponding p(D|z,H0) for a grid
        """
        prob = np.array([self.__pD_zH0(H0) for H0 in H0vec])
        path = pkg_resources.resource_filename('gwcosmo', 'data/other/{}_z_H0_pD_array.p'.format(self.mass_distribution))
        pickle.dump((self.z_array,H0vec,prob),open(path,'wb'))
        return (self.z_array,H0vec,prob)
        

    def pD_dlH0_eval(self,dl,H0):
        """
        Returns the probability of detection at a given value of luminosity distance and H0.
        Note that this is slower than the function pD_zH0_eval(z,H0).
        
        Parameters
        ----------
        dl : float or array_like
            value(s) of luminosity distances in Mpc
        H0 : float
            H0 value in kms-1Mpc-1
        
        Returns
        -------
        float or array_like
            Probability of detection at the given luminosity distance and H0, marginalised over masses, inc, pol, and sky location
        """
        z = np.array([z_dlH0(x,H0) for x in dl])
        return self.interp_average(z,H0)
        
        
    def pD_zH0_eval(self,z,H0):
        """
        Returns the probability of detection at a given value of redshift and H0.
        
        Parameters
        ----------
        z : float or array_like
            value(s) of redshift
        H0 : float
            H0 value in kms-1Mpc-1
        
        Returns
        -------
        float or array_like
            Probability of detection at the given redshift and H0, marginalised over masses, inc, pol, and sky location
        """
        return self.interp_average(z,H0)


    def pD_H0_zinterp(self,H0):
        """
        Hopefully something faster than 2d interpolation over z and H0 and p(D|z,H0)
        """
        # TODO: write this function, so that an integral for any given H0 over z is fast
        pass


    def __call__(self, dl, H0):
        """
        To call as function of dl and H0
        
        Parameters
        ----------
        dl : float or array_like
            value(s) of luminosity distances in Mpc
        H0 : float
            H0 value in kms-1Mpc-1
        
        Returns
        -------
        float or array_like
            Returns Pdet(dl,H0).
        """
        return self.pD_dlH0_eval(dl, H0)
    
    
    def pD_distmax(self, dl, H0):
        """
        Parameters
        ----------
        dl : float or array_like
            value(s) of luminosity distances in Mpc
        H0 : float
            H0 value in kms-1Mpc-1
        
        Returns
        -------
        float or array_like
            Returns twice the maximum distance given corresponding to Pdet(dl,H0) = 0.01.
        """
        return 2.*dl[np.where(self.pD_dlH0_eval(dl, H0)>0.01)[0][-1]]


    def __snr_squared_basic(self,RA,Dec,m1,m2,inc,psi,detector,gmst,dl):
        """
        the optimal snr squared for one detector, used for marginalising over sky location, inclination, polarisation, mass
        
        Parameters
        ----------
        RA,Dec : float
            sky location of the event in radians
        m1,m2 : float
            source masses in kg
        inc : float
            source inclination in radians
        psi : float
            source polarisation in radians
        detector : str
            name of detector in network as a string (eg 'H1', 'L1')
        gmst : float
            Greenwich Mean Sidereal Time in seconds
                    
        Returns
        -------
        float
            snr squared*dL^2 for given parameters at a single detector
        """
        mtot = self.__mtot(m1,m2)
        mc = self.mchirp(m1,m2)
        A = self.__reduced_amplitude(RA,Dec,inc,psi,detector,gmst) * np.power(mc,5.0/6.0)/ (dl*lal.PC_SI*1e6)
        
        fmax = self.__fmax(mtot)
        num = self.__interpolnum(fmax)
    
        return 4.0*A**2*num*np.power(lal.G_SI,5.0/3.0)/lal.C_SI**3.0      


    def __pD_dl_basic(self,H0):
        """
        Detection probability over a range of distances, returned as an interpolated function.
        
        Parameters
        ----------
        H0 : float
            value of Hubble constant in kms-1Mpc-1
        
        Returns
        -------
        interpolated probabilities of detection over an array of luminosity distances, for a specific value of H0
        """
        rho = np.zeros((self.Nsamps,len(self.dl_array)))
        for n in range(self.Nsamps):
            rhosqs = [ self.__snr_squared_basic(self.RAs[n],self.Decs[n],self.m1[n],self.m2[n],self.incs[n],self.psis[n], det, 0.0, self.dl_array) for det in self.__lal_detectors]
            rho[n] = np.sum(rhosqs,0)

        effective_threshold = np.sqrt(len(self.detectors)) * self.snr_threshold
        survival = ncx2.sf(effective_threshold**2,4,rho)
        prob = np.sum(survival,0)/self.Nsamps
        self.spl = splrep(self.dl_array,prob)
        return splrep(self.dl_array,prob)


    def pD_dl_eval_basic(self,dl):
        """
        Returns a probability for a given distance dl from the interpolated function.
        Or an array of probabilities for an array of distances.
        """
        return splev(dl,self.spl,ext=1)


    # TODO: repair pixel-based functions
    def __pD_dlradec(self,Nside,dl_array):
        """
        NEEDS FIXING
        Detection probability over a range of distances, at each pixel on a healpy map.
        """
        no_pix = hp.pixelfunc.nside2npix(Nside)
        pix_ind = range(0,no_pix)
        theta,phi = hp.pixelfunc.pix2ang(Nside,pix_ind)
        RA_map = phi
        Dec_map = np.pi/2.0 - theta
        
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


    def pDdl_radec(self,RA,Dec,gmst):
        """
        NEEDS FIXING
        Returns the probability of detection function for a specific ra, dec, and time.
        """
        ipix = hp.ang2pix(self.Nside,np.pi/2.0-Dec,RA-gmst)
        return self.interp_map[ipix]
    
