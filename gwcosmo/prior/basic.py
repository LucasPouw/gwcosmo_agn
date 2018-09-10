"""
Rachel: Some basic priors (wasn't sure where else to put them)
"""

def pz_nG(z):
    """
    prior on z outside the galaxy catalogue (unnormalised)
    """
    return z*z
