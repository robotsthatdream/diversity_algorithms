from setuptools import setup, find_packages

setup(name='diversity_algorithms',
      install_requires=['gym>=0.2.3','deap','scoop','dill','alphashape'],
      version='0.1.0',
      packages=find_packages(),
      include_package_data=True,
      author='Alex Coninx',
      author_email='coninx@isir.upmc.fr'
)
