#from pylab import *
import sys
from numpy import *
try:
  from lal import *
except ImportError:
  sys.stderr.write('populate_posteriors.py: Could not import swig bindings for lal. Not all functionalities will be supported.\n')
  pass
from scipy.special import gammainc
from scipy.integrate import quad,dblquad
from scipy.stats import norm

def detection_probability(threshold,abs_mag):
    if threshold<abs_mag:
        return 1
    else:
        return 0

class SchectherMagFunction(object):
    def __init__(self,Mstar,alpha,phistar):
        self.Mstar=Mstar
        self.phistar=phistar
        self.alpha=alpha
        self.norm=None
    def evaluate(self,m):
        return 0.4*log(10.0)*self.phistar*pow(10.0,-0.4*(self.alpha+1.0)*(m-self.Mstar))*exp(-pow(10,-0.4*(m-self.Mstar)))
    def normalise(self,mmin,mmax):
        if self.norm == None:
            self.norm = quad(self.evaluate,mmin,mmax)[0]
    def pdf(self,m):
        return self.evaluate(m)/self.norm

class SelectionFunction(object):
    def __init__(self,threshold):
        self.threshold = threshold
    def evaluate(self,m):
        if m<self.threshold: return 1
        else: return 0

def AbsoluteMagnitude(app_m,dl):
    return app_m - 5.0*log10(dl)-25.0

class ProbabilityOfObserving(object):
    def __init__(self,Mstar,alpha,phistar,threshold,omega):
        self.Mstar=Mstar
        self.phistar=phistar
        self.alpha=alpha
        self.omega = omega
        self.threshold = threshold
        self.LuminosityFunction = SchectherMagFunction(self.Mstar,self.alpha,self.phistar)
        self.SelectionFunction = SelectionFunction(self.threshold)
        self.VolumeDistribution=UniformComovingVolumeDistribution
        self.norm = None
    def evaluate(self,z,m):
        dl = LuminosityDistance(self.omega,z)
        SF = self.SelectionFunction.evaluate(m)
        if SF==0:
            return -inf
        return log(self.LuminosityFunction.evaluate(AbsoluteMagnitude(m,dl)))+log(self.VolumeDistribution(self.omega,z,-1))


"""
    typical values for the r band (http://arxiv.org/abs/0806.4930)
"""
Mstar = -20.73 + 5.*log10(0.7)
alpha = -1.23
phistar = 0.009 * (0.7*0.7*0.7) #Mpc^3