# Shift Before You Learn: Enabling Low-Rank in RL

This repository contains code accompanying the paper [Shift Before You Learn: Enabling Low-Rank Representations in Reinforcement Learning](https://arxiv.org/abs/2509.05193). The experiments explore **successor measures, rank approximations, and policy performance** in gridworld environments of varying complexity.

---

## Contents

- **H1_4rooms.ipynb / H1_4rooms.py**  
  Experiments on a `Cross-4` grid environment.  
  - Visualization of the 4-room maze layout.
  - Construction of the transition matrix under a uniform random policy.
  - Singular value decomposition (SVD) analysis of shifted successor matrices.
  - Rank-4 approximations and evaluation using two-to-infinity norm.
  - Comparison of $P^k$ and $M_{\pi,k}$ matrices across shifts.

- **H2.ipynb / H2.py**  
  Experiments in `Medium-maze`, `Large-maze`, and `U-maze` environments.  
  - Compute and visualize occupancy matrices under a uniform random policy.
  - Compute singular value decay of successor measures for different shifts `k`.
  - Policy evaluation using rank-$r$ approximations for exact and TD-learned successor measures.
  - Sample complexity study: accuracy vs number of trajectories.
  - Visualizations include maze plots with optimal action arrows and singular value decay plots.

- **H3.ipynb / H3.py**  
  Experiments using **reweighted SVD**.  
  - Reweighted low-rank approximations using stationary measure $\nu$.
  - Comparison of standard SVD vs $\nu$-weighted SVD in policy evaluation.
  - Visualization of accuracy, norm differences, and stationary measure distributions.

- **H4.ipynb / H4.py**  
  Model-based vs TD-learned successor measures.  
  - Sample complexity analysis for different ranks.
  - Evaluation of learned policies in `Medium-maze`.
  - Performance comparison between model-based and TD-based approaches.

- **maze_environments.py**  
  Implementation of custom gridworld environments:
  - `Cross-4`, `Medium-maze`, `Large-maze`, `U-maze`
  - Functions for valid transitions, state indexing, and generating datasets.

- **policies.py**  
  Implemented policies:
  - `UnifRandomPolicy`
  - `GreedySMPolicy`
  - `AveragedGoalPolicy`
  - Utility functions for policy evaluation.

- **successor_measure.py**  
  Functions to compute successor measures:
  - Exact occupancy matrices
  - TD-learned successor matrices
  - Model-based successor matrices

- **utils.py**  
  Utility functions for:
  - Rank-$r$ approximations
  - Two-to-infinity norm calculations
  - Other matrix operations

---

## Installation

Requires Python ≥ 3.8. Install dependencies using:

```bash
pip install numpy matplotlib
