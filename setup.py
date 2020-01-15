from setuptools import setup, find_packages

setup(name='diversity_algorithms',
      version='0.0.2',
      install_requires=['gym>=0.2.3','deap','scoop','dill'],
      packages=find_packages(),
      include_package_data=True,
      author='Alex Coninx',
      author_email='coninx@isir.upmc.fr'
)
