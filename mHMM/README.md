Mixed Hidden Markov Model (mHMM) Reproduction — COPD Case Study
Overview

This repository reproduces the analysis from the paper:
“Handling underlying discrete variables with bivariate mixed Hidden Markov Models in NONMEM.”

The original study implemented a mixed HMM for longitudinal FEV1 and PRO data in COPD patients using NONMEM + SAEM.
This project reimplements the full workflow entirely in open-source Python, using Stan, NumPy, and CmdStanPy for estimation.

We have documented the steps through modular Jupyter notebooks for transparency and reproducibility.

Repository Structure
mHMM/
|
├── notebooks/
│   ├── 00_setup.ipynb            → environment checks, folder creation
│   ├── 01_model_math.ipynb       → model equations, emission/transition definitions
│   ├── 02_simulator.ipynb        → dataset generation under reference scenario
│   ├── 03_estimation.ipynb       → parameter estimation (Monte Carlo MLE + Stan)
│   ├── 04_SSE.ipynb              → stochastic simulation–estimation (bias, RMSE, coverage)
│   ├── 05_power_analysis.ipynb   → Monte Carlo mapped power (MCMP) study
│   ├── 06_vpc_sensitivity.ipynb  → visual predictive checks and sensitivity analysis
│
├── models/
│   |
│   └── stan/
│       └── mhmm_marginal.stan    → Stan implementation of mHMM likelihood
├── Src/   
│   ├── emissions.py              → EmissionModel class (bivariate FEV1/PRO)
│   ├── transitions.py            → TransitionModel class
├── data/
│   ├── simulated/                → simulated datasets (e.g., ref_scenario.csv)
│   ├── results/                  → estimation, SSE, and power outputs
│   └── logs/                     → optional run logs
├── Utils
│   └── utils.py                  → utility functions
│   ├── math_utils.py             → math functions
│   ├── forward_utils.py          → forward algorithm functions
│   ├── viterbi_utils.py          → Viterbi algorithm functions
│   ├── sampling_utils.py         → sampling functions                  
├── run_all.ipynb                 → master runner
└── setup_mhmm.sh                 → project structure bootstrap script

Workflow Summary
Step	Notebook	Objective
1	00_setup.ipynb	Verify environment, create directory structure, import dependencies.
2	01_model_math.ipynb	Implement emission and transition components of the mHMM using population mode parameters and variance terms.
3	02_simulator.ipynb	Generate subject-level data from the model under the reference scenario (Table 1).
4	03_estimation.ipynb	Estimate parameters using Monte Carlo MLE and Bayesian (Stan) inference.
5	04_SSE.ipynb	Run stochastic simulation–estimation replicates to evaluate bias, RMSE, and coverage.
6	05_power_analysis.ipynb	Conduct Monte Carlo Mapped Power (MCMP) analysis to estimate statistical power under different hypotheses.
7	06_vpc_sensitivity.ipynb	Perform visual predictive checks and sensitivity analyses to assess model fit and robustness.

Key Methodological Notes

Replaces NONMEM + SAEM with Stan + HMC for fully open-source estimation.

Marginal likelihoods approximated using Monte Carlo integration over random effects.

Transition probabilities follow the same logistic form (Eqs. 7–10 in the paper).

Supports bivariate emission structure with state-specific correlations (qR, qE).

Reproduces all major analyses: estimation, SSE validation, power, and VPCs.

How to Run

Clone the repository and create the environment:

bash mHMM/setup_mhmm.sh


Launch Jupyter Lab:

jupyter lab


Execute the notebooks in order (00 → 06).
For full runs, start with run_all.ipynb (optional master runner).

Citation

If you use this workflow, please cite both:

The original NONMEM mHMM paper.

This open-source reproduction (link to your GitHub/Zenodo DOI).

License

Open-source under the MIT License.
Feel free to fork and adapt for your own HMM or mixed-effects modeling studies.
