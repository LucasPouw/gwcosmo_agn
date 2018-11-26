import numpy as np, healpy as hp
from scipy import interpolate
import sys

class confidence(object):
    def __init__(self, counts):
        # Sort in descending order in frequency
        self.counts_sorted = np.sort(counts.flatten())[::-1]
        # Get a normalized cumulative distribution from the mode
        self.norm_cumsum_counts_sorted = np.cumsum(self.counts_sorted) / np.sum(counts)
        # Set interpolations between heights, bins and levels
        self._set_interp()
    def _set_interp(self):
        self._length = len(self.counts_sorted)
        # height from index
        self._height_from_idx = interpolate.interp1d(np.arange(self._length), self.counts_sorted, bounds_error=False, fill_value=0.)
        # index from height
        self._idx_from_height = interpolate.interp1d(self.counts_sorted[::-1], np.arange(self._length)[::-1], bounds_error=False,               fill_value=self._length)
        # level from index
        self._level_from_idx = interpolate.interp1d(np.arange(self._length), self.norm_cumsum_counts_sorted, bounds_error=False, fill_value=1.)
        # index from level
        self._idx_from_level = interpolate.interp1d(self.norm_cumsum_counts_sorted, np.arange(self._length), bounds_error=False, fill_value=self._length)
    def level_from_height(self, height):
        return self._level_from_idx(self._idx_from_height(height))
    def height_from_level(self, level):
        return self._height_from_idx(self._idx_from_level(level))


# RA and dec from HEALPix index
def ra_dec_from_ipix(nside, ipix, nest=False):
    (theta, phi) = hp.pix2ang(nside, ipix, nest=nest)
    return (phi, np.pi/2.-theta)

# HEALPix index from RA and dec
def ipix_from_ra_dec(nside, ra, dec, nest=False):
    (theta, phi) = (np.pi/2.-dec, ra)
    return hp.ang2pix(nside, theta, phi, nest=nest)

def twodskyprob(skymap_file, ra, dec, z_gal, z_min, z_max, sky_level=0.99):

    skymap = hp.read_map(skymap_file)
    nside = hp.npix2nside(len(skymap))

    # Map each galaxy to a pixel on the skymap
    ipix_gal = ipix_from_ra_dec(nside, ra, dec)

    # Height of probability contour corresponding to confidence level set above
    skyconf_obj = confidence(skymap)
    sky_height = skyconf_obj.height_from_level(sky_level)

    # Pixels of skymap inside the probability contour
    ipix_above_height, = np.where(skymap > sky_height)

    # Indices of galaxies inside the probability contour
    idx_gal_above_height = np.array([ig in ipix_above_height for ig in ipix_gal])
    
    # Impose a cut on z (min and max values chosen above)
    valid_idx, = np.where((z_min<z_gal)&(z_gal<z_max)&(idx_gal_above_height))
    sys.stderr.write('%d galaxies in %d%% sky.\n'%(len(valid_idx), int(100*sky_level)))
    valid_gal_ra_arr = ra[valid_idx]
    valid_gal_dec_arr = dec[valid_idx]
    valid_gal_z_arr = z_gal[valid_idx]
    valid_gal_sky_prob_arr = skymap[ipix_gal[valid_idx]]
    valid_gal_sky_conf_arr = np.vectorize(skyconf_obj.level_from_height)(valid_gal_sky_prob_arr)

    return valid_gal_ra_arr, valid_gal_dec_arr, valid_gal_z_arr, valid_gal_sky_prob_arr, valid_gal_sky_conf_arr


### PDET ###
def __pD_zH0_interp(self,H0vec):
    """
    OBSOLETE
    Function which calculates p(D|z,H0) for a range of redshift and H0 values

    Parameters
    ----------
    H0vec : array_like
        numpy array of H0 values in kms-1Mpc-1

    Returns
    -------
    2D interpolation object over z and H0
    """
    # TODO pickle z,H0,prob without interp, and try different interp options
    prob = np.array([self.__pD_zH0(H0) for H0 in H0vec])
    interp = interp2d(self.z_array,H0vec,prob)
    interp_av_path = pkg_resources.resource_filename('gwcosmo', 'likelihood/pD_zH0_interp.p')
    pickle.dump(interp,open(interp_av_path,'wb'))
    return interp


def __snr_squared_old(self,RA,Dec,m1,m2,inc,psi,detector,gmst,z=0):
    """
    OBSOLETE
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
    OBSOLETE
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
    UNUSED
    detection probability for a particular event (masses, distance, sky position, orientation and time)
    """
    rhosqs = [ self.snr_squared_single(dl, ra, dec, m1, m2, inc, psi, det, gmst) for det in self.__lal_detectors]
    combined_rhosq = np.sum(rhosqs)
    effective_threshold = np.sqrt(len(self.detectors)) * self.snr_threshold
    return ncx2.sf(effective_threshold**2 , 4, combined_rhosq)


def pD_dl_single(self, dl):
    """
    OBSOLETE
    Detection probability for a specific distance, averaged over all other parameters - without using interpolation
    """       
    return np.mean(
        [ self.pD_event(dl, self.RAs[i], self.Decs[i], self.m1[i], self.m2[i], self.incs[i], self.psis[i], 0.0) for i in range(N)]
        )


def pD_dl_eval_old(self,dl):
    """
    OBSOLETE
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

