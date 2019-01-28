#Global Imports
import numpy as np
import gwcosmo

from scipy.integrate import cumtrapz
from scipy.optimize import fmin
from scipy.interpolate import splev, splrep, interp1d
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import leastsq

def HDI(credint,y,x):
    cdfvals = cumtrapz(y,x)
    sel = cdfvals > 0.
    x = x[1:][sel]
    cdfvals = cdfvals[sel]
    ppf = interp1d(cdfvals,x,fill_value = 0.,bounds_error=False)
    def intervalWidth(lowTailPr):
        ret = ppf(credint + lowTailPr) - ppf(lowTailPr)
        if (ret > 0.):
            return ret
        else:
            return 1e4
    HDI_lowTailPr = fmin(intervalWidth, 1.-credint)[0]
    return ppf(HDI_lowTailPr), ppf(HDI_lowTailPr+credint)

def MAP(y,x):
    sp = UnivariateSpline(x,y,s=0.)
    x_highres = np.linspace(20,140,100000)
    y_highres = sp(x_highres)
    return x_highres[np.argmax(y_highres)]

def confidence_interval(posterior,H0,level=0.683):
    MAP_post = MAP(posterior,H0)
    a, b = HDI(level,posterior,H0)
    print('Result: H0 = %.0f + %.0f - %.0f (MAP and 68.3 percent HDI)' %(MAP_post,b-MAP_post,MAP_post-a))
    return MAP_post,a,b