#!/bin/bash

#create main folder
mkdir -p mHMM/{src,utils,models,notebooks,data/{simulated,results/summary}}

#create empty notebooks
for nb in 00_setup 01_model_math 02_simulator 03_estimation 04_SSE 05_power_analysis 06_vpc_sensitivity; do
    touch mHMM/notebooks/${nb}.ipynb
done

#create Python modules
touch mHMM/src/{emissions.py,transitions.py,mhmm_forward.py,simulator.py}
touch mHMM/utils/{forward_utils.py,sampling_utils.py,viterbi_utils.py,math_utils.py}

#Create stan folder and placeholder model file
mkdir -p mHMM/models/stan 
echo "// Stan model for mixed HIdden Markov mOdel (to be filled)" > mHMM/models/stan/mhmm_model.stan

#create README.md
if [ ! -f "README.md" ]; then
    cat << 'EOF' > README.md 


This subproject reproduces the paper:

> *Handling underlying discrete variables with bivariate mixed hidden Markov models in NONMEM*  
> using open-source tools (Python + Stan) instead of NONMEM.

Refer to notebooks in the `notebooks/` folder for the step-by-step workflow.
EOF

# Create environment.yml
cat << 'EOF' > mHMM/environment.yml
name: mhmm_env
channels:
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - scipy
  - pandas
  - matplotlib
  - seaborn
  - tqdm
  - joblib
  - numba
  - cmdstanpy
  - notebook
  - arviz
EOF

# Create top-level run_all notebook placeholder
touch mHMM/run_all.ipynb

echo "mHMM project structure created successfully!"  