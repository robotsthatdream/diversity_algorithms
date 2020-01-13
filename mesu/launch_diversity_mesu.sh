#!/bin/bash
#PBS -q alpha
#PBS -l select=1:ncpus=64
#PBS -N diversity_algo

# Alex Coninx
# ISIR - Sorbonne Universite / CNRS
# 13/01/2020


USER_NAME=coninx

# Complete this to where the installation script was run
DIVERSITY_BASE_DIR=/home/${USER_NAME}/src/diversity
RESULTS_DIR=/scratchalpha/${USER_NAME}/diversity

# Experiment specifications
# Experiment specifications
EXPERIMENT_SCRIPT="gym_novelty.py"
DIVERSITY_ARGUMENTS=""

# /!\ If you change this you should also probably manually change /!\
#        the ncpus= directive in the PBS directive above
N_WORKERS_SCOOP=64


echo "Creating dir ${RESULTS_DIR} if not existing"

mkdir -p ${RESULTS_DIR}


module purge
module load python-3.6

CMD_LINE="python -m scoop -n ${N_WORKERS_SCOOP} ${DIVERSITY_BASE_DIR}/diversity_algorithms_dev/diversity_algorithms/experiments/${EXPERIMENT_SCRIPT} ${DIVERSITY_ARGUMENTS}"

echo "Loading virtualenv env in $DIVERSITY_BASE_DIR"

cd $DIVERSITY_BASE_DIR

source env/bin/activate


echo "Going to $RESULTS_DIR"

cd $RESULTS_DIR

echo "Command line will be : '$CMD_LINE'"

echo "Running..."

${CMD_LINE}

echo "Python terminated"
