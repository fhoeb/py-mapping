from setuptools import setup, find_packages

setup(name='py-mapping',
      version='1.0',
      description='Generating and converting between coefficients for quantum impurity models star and chain '
                  'geometry baths',
      author='Fabian Hoeb, Ish Dhand, Alexander Nuesseler',
      install_requires=['numpy>=1.12', 'scipy>=0.19', 'py-orthpol>=1.1'],
      author_email='fabian.hoeb@uni-ulm.de, ish.dhand@uni-ulm.de, alexander.nuesseler@uni-ulm.de',
      packages=find_packages(where='.'))
