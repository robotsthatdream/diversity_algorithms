#!/bin/sh

# Alex Coninx
# ISIR - Sorbonne Universite / CNRS
# 7/10/2019

WORK_DIR=$PWD


while true; do
    read -p "This will install the diversity algorithms and all dependencies in ${WORK_DIR}. Do you want that ? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done




echo "Please provide the login of an account with access to the diversity repository. Git will ask for the password later"
read -p "Login : " GIT_USER




mkdir -p $WORK_DIR
cd $WORK_DIR


# Load non-obsolete development tools
module load python-3.6 # also makes python-3.6 the default python
module load gcc/8.2 # also makes gcc-8.2 the default gcc


# Setup venv
virtualenv -p /opt/dev/Langs/Conda3/bin/python env

source env/bin/activate


# 1) Install pybind11
echo
echo "====================================="
echo "===== (1/5) Installing pybind11 ====="
echo "====================================="
echo
cd $WORK_DIR
git clone https://github.com/pybind/pybind11.git
cd pybind11
# Install the pybind11 python module
pip3 install  .
# Where we can find pybind11 (especially its includes)
PYBIND11_DIR=$WORK_DIR/pybind11
 

# 2) Install and patch fastsim
echo
echo "===================================================="
echo "===== (2/5) Patching and installing libfastsim ====="
echo "===================================================="
echo
cd $WORK_DIR
git clone https://github.com/jbmouret/libfastsim.git
# We need to clone the pyfastsim repository now to get the patch
git clone https://github.com/alexendy/pyfastsim.git
cd libfastsim
# Patch libfastsim
patch -p1 < ../pyfastsim/fastsim-boost2std-fixdisplay.patch
# Build and install
python2.7 ./waf configure --prefix=./install
python2.7 ./waf build
python2.7 ./waf install
# Where we installed fastsim
FASTSIM_DIR=$WORK_DIR/libfastsim/install

# 3) Install pyfastsim
echo
echo "======================================"
echo "===== (3/5) Installing pyfastsim ====="
echo "======================================"
echo
cd $WORK_DIR/pyfastsim
CPPFLAGS="-I${PYBIND11_DIR}/include -I${FASTSIM_DIR}/include" LDFLAGS="-L${FASTSIM_DIR}/lib" pip3 install .


# 4) Install fastsim gym
echo
echo "====================================================="
echo "===== (5/5) Installing the diversity algorithms ====="
echo "====================================================="
echo
cd $WORK_DIR
git clone https://github.com/alexendy/fastsim_gym.git
cd fastsim_gym
pip3 install .
# Also installs Gym and dependancies : numpy and scipy especially


# 5) install the experiments
echo
echo "====================================================="
echo "===== (5/5) Installing the diversity algorithms ====="
echo "====================================================="
echo
cd $WORK_DIR
git clone https://$GIT_USER@github.com/robotsthatdream/diversity_algorithms_dev.git
cd diversity_algorithms_dev
pip3 install .
# Also install dependencies
echo
echo
echo "********************************************************************************"
echo "Diversity algorithms module have been installed in $WORK_DIR"
echo "********************************************************************************"

deactivate

module purge
