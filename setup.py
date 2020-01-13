from setuptools import setup, find_packages

setup(name='diversity_algorithms',
      version='0.0.1',
      install_requires=['gym>=0.2.3','deap','scoop','dill','alphashape'],
      packages=find_packages(),
      author='Alex Coninx',
      author_email='coninx@isir.upmc.fr'
)
