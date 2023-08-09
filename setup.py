import sys
from setuptools import setup

setup_requires = ['setuptools >= 30.3.0']
if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
    setup_requires.append('pytest-runner')
if {'build_sphinx'}.intersection(sys.argv):
    setup_requires.extend(['recommonmark',
                           'sphinx'])

def readme():
    with open('README.md') as f:
        return f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(name='gwcosmo',
      version='1.0.0',
      description='A package to estimate cosmological parameters using gravitational-wave observations',
      url='https://git.ligo.org/lscsoft/gwcosmo',
      author='Cosmology R&D Group',
      author_email='cbc+cosmo@ligo.org',
      license='GNU',
      packages=['gwcosmo', 'gwcosmo.likelihood', 'gwcosmo.prior', 'gwcosmo.utilities','gwcosmo.plotting','gwcosmo.maps'],
      package_dir={'gwcosmo': 'gwcosmo'},
      scripts=['bin/gwcosmo_compute_redshift_prior', 'bin/gwcosmo_dark_siren_posterior', 'bin/gwcosmo_bright_siren_posterior', 'bin/create_injections', 'bin/create_injections_dag'],
      include_package_data=True,
      install_requires=reqs,
      setup_requires=setup_requires,
      zip_safe=False)

