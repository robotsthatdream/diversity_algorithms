#!/bin/bash
#PBS -q beta
#PBS -l walltime=12:00:00
#PBS -l select=9:ncpus=24
#PBS -N evolvability_analysis

# Alex Coninx
# ISIR - Sorbonne Universite / CNRS
# 13/01/2020



# Complete this to where the installation script was run
DIVERSITY_BASE_DIR=/home/${USER}/src/diversity
RESULTS_DIR=/scratchbeta/${USER}/diversity_results

# Experiment specifications
EVOLVABILITY_SCRIPT=evolvability_ns_new.py
TARGET_RUN="${RESULTS_DIR}/NS_12x12_2000_evo/Fastsim-12x12_NS_2020_01_17-17:44:54_0"
POPFILE_BASE_NAME=population_all_dist_to_explored_area_dist_to_parent_rank_novelty_gen
GENS_TO_RUN="1 25 50 75"

# /!\ If you change this you should also probably manually change /!\
#        the ncpus= and select= directive in the PBS directive above
#        N_WORKERS_SCOOP must be select*ncpus or lower
N_WORKERS_SCOOP=216


echo "Creating dir ${RESULTS_DIR} if not existing"

mkdir -p ${RESULTS_DIR}


module purge
module load python-3.6

echo "Loading virtualenv env in $DIVERSITY_BASE_DIR"
cd $DIVERSITY_BASE_DIR
source env/bin/activate

echo "Going to $RESULTS_DIR"

cd $RESULTS_DIR

echo "Command line will be : '$CMD_LINE'"




for gen in ${GENS_TO_RUN}
do
	echo "Running for gen ${gen}..."
	CMD_LINE="python -m scoop -n ${N_WORKERS_SCOOP}  ${DIVERSITY_BASE_DIR}/diversity_algorithms_dev/diversity_algorithms/analysis/${EVOLVABILITY_SCRIPT} -p ${TARGET_RUN}/${POPFILE_BASE_NAME}${gen}.npz"
	${CMD_LINE}
	echo "Python terminated"
done
echo "Evolvability computation finished
