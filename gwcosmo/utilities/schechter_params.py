"""
Rachel Gray 2020
"""

class SchechterParams():
    """
    Returns the source frame Schechter function parameters for a given band.

    Note Mstar is here is M* - 5log10(h)

    ugriz parameters are from https://iopscience.iop.org/article/10.1086/375776/pdf
    (tables 1 and 2)
    B parameters are from https://doi.org/10.1046/j.1365-8711.2002.05831.x
    K parameters are from section 2.1 of
    https://iopscience.iop.org/article/10.3847/0004-637X/832/1/39
    (note paper quotes M*=-23.55 but means M*=-23.55 + 5log10(h))
    """

    def __init__(self, band):
        """
        Parameters
        ----------
        band : observation band (B,K,u,g,r,i,z)
        """

        self.Mstar = None
        self.alpha = None
        self.Mmin = None
        self.Mmax = None

        self.alpha, self.Mstar, self.Mmin, self.Mmax = self.values(band)

    def values(self, band):
        if band == 'B':
            return -1.21, -19.70, -22.96, -12.96
        elif band == 'K':
            return -1.02, -23.55, -27.0, -12.96
        elif band == 'u':
            return -0.92, -17.93, -21.93, -15.54 #TODO check Mmin and Mmax
        elif band == 'g':
            return -0.89, -19.39, -23.38, -16.10 #TODO check Mmin and Mmax
        elif band == 'r':
            return -1.05, -20.44, -24.26, -16.11 #TODO check Mmin and Mmax
        elif band == 'i':
            return -1.00, -20.82, -23.84, -17.07 #TODO check Mmin and Mmax
        elif band == 'z':
            return -1.08, -21.18, -24.08, -17.34 #TODO check Mmin and Mmax
        elif band == 'W1':
            return -1.12, -24.09, -28, -16.6 # https://iopscience.iop.org/article/10.1088/0004-637X/697/1/506/pdf Tab 3 (All 3.6)
                                             # https://arxiv.org/pdf/1702.07829.pdf (Mmin Mmax Fig2)

        else:
            raise Exception("Expected 'W1', B', 'K', 'u', 'g', 'r', 'i' or 'z' band argument")


