#!/bin/bash

# Alex Coninx
# ISIR - Sorbonne Universite / CNRS
# 14/01/2020

OLD_PWD=$PWD

# Complete this to where the installation script was run
DIVERSITY_BASE_DIR=/home/${USER}/src/diversity

module purge
module load python-3.6

cd $DIVERSITY_BASE_DIR

source env/bin/activate

cd $OLD_PWD

python "$@"

deactivate
module purge
