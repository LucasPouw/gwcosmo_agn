#usr/bin/env python

try:
  from lal import C_SI
except ImportError:
  C_SI = 299792458.0


def sph2vec(ra,dec):
    ra=ra/180.*np.pi
    dec=dec/180.*np.pi
    return np.array([np.sin(np.pi/2.-dec)*np.cos(ra),np.sin(np.pi/2.-dec)*np.sin(ra),np.cos(np.pi/2.-dec)])

def zhelio_to_zcmb(ra,dec,z_helio):
    ra_cmb = 168.
    dec_cmb = -7.0
    v_cmb = 370.
    z_gal_cmb = v_cmb*np.dot(sph2vec(ra_cmb,dec_cmb),sph2vec(ra,dec))/(C_SI/1e3)
    z_cmb = (1.+z_gal_cmb)*(1.+z_helio) - 1.
    return z_cmb

