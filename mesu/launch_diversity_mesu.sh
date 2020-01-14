#!/bin/bash
#PBS -q alpha
#PBS -l walltime=11:59:59
#PBS -l select=1:ncpus=128
#PBS -N diversity_algo

# Alex Coninx
# ISIR - Sorbonne Universite / CNRS
# 13/01/2020



# Complete this to where the installation script was run
DIVERSITY_BASE_DIR=/home/${USER}/src/diversity
RESULTS_DIR=/scratchalpha/${USER}/diversity

# Experiment specifications
EXPERIMENT_SCRIPT="gym_novelty.py"
DIVERSITY_ARGUMENTS=""

# /!\ If you change this you should also probably manually change /!\
#        the ncpus= directive in the PBS directive above
N_WORKERS_SCOOP=128


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
