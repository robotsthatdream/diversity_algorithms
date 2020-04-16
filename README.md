# Diversity algorithms

## Content

This package contains the code of diversity algorithms, including Novelty Search and QD algorithms.

The name of the different packages are self-explaining. Experiments contains the source files of the experiments, this is propbably the first place to look at if you want to look at how the code is structured and called for an experiment.

## Dependencies
* gym, numpy, dill, deap
* scoop (required for parallelism - not strictly necessary to run but highly recommended)
* For the maze tasks : [fastsim_gym](https://github.com/alexendy/fastsim_gym), which requires [PyFastsim](https://github.com/alexendy/pyfastsim) - which itself requires a patched version of [libfastsim](https://github.com/jbmouret/libfastsim) (patch provided in the PyFastsim repository)
* For the billiard task : [Gym_billiard](https://github.com/GPaolo/Billiard)


## How to use it ?

First, install the module with ``pip3 install .`` (use -e flag if you want editable/link installation)

Then the scripts to run are the ``gym_<algo>.py`` in experiments. Launch for example:
```
python3 -m scoop gym_novelty.py
```
for novelty search on the default environment (Lehman & Stanley 2011 hard maze) with default parameters.

See parameters in the code (the ``gym_<algo>.py`` files) for task, hyperparameters, variants, etc. For example ``python3 -m scoop gym_novelty.py -e Fastsim-Pugh2015 -g 1000`` to run Novelty Search for 1000 generations on the maze from the Pugh et al. 2015 paper.


If you omit the ``-m scoop`` parameter it will run correctly but without parallelism.

This will create a directory named after the date and time you have launched that command to store the results of the experiments in the form of bd_XXXX.log files for the behavior descriptors generated at generation XXXX and population files pop_genYYY.npz for generation YYY. bd files are plain text files and pop files are numpy-zipped data files.

## Publications using the library

* Doncieux, S. and Paolo, G. and Laflaquiere, A. and Coninx, A. (2020). Novelty Search makes Evolvability Inevitable. In Genetic and Evolutionary Computation Conference (GECCO ’20), July 8–12, 2020, Cancún, Mexico. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3377930.3389840
