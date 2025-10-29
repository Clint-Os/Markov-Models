Markov Models Repository
Overview

This repository explores Markov models and their extensions, focusing on their implementation and application in continuous and discrete time settings.
It is organized into separate modules for different model types — such as Continuous-Time Markov Models (CTMMs) and Hidden Markov Models (HMMs) — each with notebooks, code, and references to the mathematical formulations from relevant papers.

Goals

Implement and visualize Markov processes step by step in Python.

Reproduce results from published papers (CTMMs, mHMMs, etc.).

Provide educational insight into:

State transition dynamics

Likelihood estimation

Inference and parameter estimation

Model evaluation

Getting Started

1. Clone the repository

git clone https://github.com/<your-username>/<repo-name>.git
cd markov-models


2. Create a virtual environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate if you are using Windows


3. Install dependencies

pip install -r requirements.txt
mHMM has an environment.yml file


4. Open Jupyter

jupyter notebook


Then explore the models within each subfolder.

Modules
Continuous-Time Markov Models (ctmm)

Implements continuous-time transition dynamics, likelihood computation, and time-dependent state probabilities.
Includes parameter estimation and visualization of disease or event transitions over time.

Markov/Hidden Markov Models (mHMM)

Covers both observable and hidden states, forward-backward algorithms, EM training, and applications to sequence data.

References --

Norris, J. R. Markov Chains. Cambridge University Press.

Recent papers reproduced in the repository (see each module for details).

Author

Clinton Osebe
Learning, implementing, and reproducing Markov-based models for educational and research purposes.
