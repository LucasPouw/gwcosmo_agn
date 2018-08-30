"""Module containing functionality for creation and management of completion functions."""
# Global 
import numpy as np
from astropy import units as u
import astropy.constants as constants

# Global
# Set to 1.0 to use NGC4993 only, else set to 0.0

def pd(x,blue_luminosity_density):
    coverh = (constants.c.to('km/s') / (70 * u.km / u.s / u.Mpc)).value
    tmpd = coverh * x
    tmpp = (3.0*coverh*4.0*np.pi*0.33333*blue_luminosity_density*(tmpd-50.0)**2)
    return np.ma.masked_where(tmpd<50.,tmpp).filled(0)

class completionFunction(object):
    ''' Class for completion function objects
    '''
    def __init__(self, method='default', function = 1.0):
        """Completion function class... 
        Parameters
        """
        self.method = method
        self.function = function

    def generateCompletion(self,lumB,dist,x,useNGC4993only=0):
        if useNGC4993only>0:
           blue_luminosity_density = 1.98e-2
           
        else:
            tmpLBden = np.cumsum(lumB)[np.argmax(dist>73.)]/(4.0*np.pi*0.33333*np.power(73.0,3))
            blue_luminosity_density = tmpLBden

        p = pd(x,blue_luminosity_density)
        #self.function = p

        return p