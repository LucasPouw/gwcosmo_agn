#usr/bin/env python
import numpy as np, healpy as hp
from skymap2d.py import ipix_from_ra_dec

def z_uncertainty(skymap_file,z_list,ra,dec,sigmaz=0.0005,lumweights=None):

    '''
    A function which "smears" out galaxies in the catalog, therefore incorporating redshift uncetainties. 
    TO DO: Convert redhsift to luminosity distance (d). Calculate the likelihood with lumweights_uncert and d
    '''

    z_uncert = [] 
    ra_uncert = np.repeat(ra,100)
    dec_uncert = np.repeat(dec,100)
    if lumweights is not None:
        lumweights_uncert = np.repeat(lumweights,100)
    for z in z_list:
        z_uncert.append(z+sigmaz*np.random.randn(100)) 
    z_uncert = np.array(z_uncert).flatten()
    sel = z_uncert>0.
    z_uncert = z_uncert[sel]; ra_uncert = ra_uncert[sel]; dec_uncert = dec_uncert[sel]
    if lumweights is not None:
        lumweights_uncert = lumweights_uncert[sel]
    if lumweights is None:
        lumweights_uncert = np.ones_like(z_uncert)

    skymap = hp.read_map(skymap_file)
    nside = hp.npix2nside(len(skymap))
    ipix_gal = ipix_from_ra_dec(nside, ra_uncert, dec_uncert)
            
    return lumweights_uncert, skymap, z_uncert, ipix_gal
    
    



