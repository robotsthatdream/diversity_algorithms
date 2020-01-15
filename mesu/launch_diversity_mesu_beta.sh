#!/bin/bash
#PBS -q beta
#PBS -l walltime=11:59:59
#PBS -l select=5:ncpus=24
#PBS -N diversity_algo

# Alex Coninx
# ISIR - Sorbonne Universite / CNRS
# 13/01/2020



# Complete this to where the installation script was run
DIVERSITY_BASE_DIR=/home/${USER}/src/diversity
RESULTS_DIR=/scratchbeta/${USER}/diversity

# Experiment specifications
EXPERIMENT_SCRIPT="gym_novelty.py"
DIVERSITY_ARGUMENTS=""

# /!\ If you change this you should also probably manually change /!\
#        the ncpus= and select= directive in the PBS directive above
#        N_WORKERS_SCOOP must be select*ncpus or lower
N_WORKERS_SCOOP=119


echo "Creating dir ${RESULTS_DIR} if not existing"

mkdir -p ${RESULTS_DIR}


module purge
module load python-3.6

CMD_LINE="python -m scoop -n ${N_WORKERS_SCOOP} --python-interpreter=${DIVERSITY_BASE_DIR}/diversity_algorithms_dev/mesu/python_wrapper_subprocess_pbs.sh  ${DIVERSITY_BASE_DIR}/diversity_algorithms_dev/diversity_algorithms/experiments/${EXPERIMENT_SCRIPT} ${DIVERSITY_ARGUMENTS}"

echo "Loading virtualenv env in $DIVERSITY_BASE_DIR"

cd $DIVERSITY_BASE_DIR

source env/bin/activate


echo "Going to $RESULTS_DIR"

cd $RESULTS_DIR

echo "Command line will be : '$CMD_LINE'"

echo "Running..."

${CMD_LINE}

echo "Python terminated"
