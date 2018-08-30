#Always prefer setuptools over distutils
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
# To use a consistent encoding
from codecs import open
from os import path
import numpy, scipy
import cython_gsl

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

# check whether user has Cython
try:
    import Cython
except ImportError:
    have_cython = False
else:
    have_cython = True

# set extension
ext_modules = [ Extension("gwcosmo._gwcosmo", sources =[ "gwcosmo/_gwcosmo.pyx"], include_dirs=['gwcosmo/',numpy.get_include(),scipy.get_include(),cython_gsl.get_cython_include_dir()], libraries=['m']+cython_gsl.get_libraries(), library_dirs=[cython_gsl.get_library_dir()] ) ]

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
        long_description = f.read()

setup(
        name = 'gwcosmo',
        version = '0.0.1',
        description = 'GW cosmology utilities',
        long_description=long_description,
        author = 'Rachel Gray',
        author_email='rachel.gray@ligo.org',
        license='MIT',
        cmdclass = {'build_ext': build_ext},
        classifiers =[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        #'Intended Audience :: Developers',
        #'Topic :: Data Analysis :: Bayesian Inference',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        ],

        keywords='cosmology',
        #packages=find_packages(exclude=['contrib','docs','tests*','examples']),
        packages=['gwcosmo'],
        install_requires=['numpy','scipy','healpy','cython','cythongsl'],
        setup_requires=['numpy','cythongsl','cython'],
        #tests_require=['corner'],
        package_data={"": ['*.c', '*.pyx', '*.pxd', '*.p']},
        # To provide executable scripts, use entry points in preference to the
        # "scripts" keyword. Entry points provide cross-platform support and allow
        # pip to create the appropriate form of executable for the target platform.
        entry_points={
        #    'console_scripts':['sample=sample:main',
        #        ],
            },
        #test_suite='tests',
        include_dirs = [cython_gsl.get_include()],
        ext_modules=ext_modules
        )

