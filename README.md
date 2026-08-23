<div align="center">
  <img src="assets/gradMFM.png" alt="gradMFM" width="220"><br>
  <strong>From resting-state dynamics to latent whole-brain circuitry</strong>
</div>

<p align="center">
  A gradient-based training framework for fitting macroscopic brain models and
  recovering mechanistically interpretable latent parameters.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9">
  <img src="https://img.shields.io/badge/JAX-0.4-7E57C2" alt="JAX 0.4">
  <img src="https://img.shields.io/badge/BrainPy-2.6-00A6A6" alt="BrainPy 2.6">
  <img src="https://img.shields.io/badge/GPU-CUDA-76B900?logo=nvidia&logoColor=white" alt="CUDA accelerated">
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#why-gradmfm">Why gradMFM</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#released-example">Released example</a> ·
  <a href="#manuscript">Manuscript</a>
</p>

---

## Overview

`gradMFM` turns whole-brain modeling into a differentiable inverse problem.
Rather than tuning a simulator by hand, the framework optimizes a biophysical
model directly against empirical brain activity. Static and dynamic observables
guide the recovery of regional circuit properties, long-range coupling, and
latent structural connectivity.

Built on BrainPy and JAX, `gradMFM` combines interpretable neural dynamics with
automatic differentiation, GPU acceleration, and validation-guided model
selection. It is designed for demanding, high-dimensional inference while
keeping the resulting parameters tied to an explicit generative model.

<p align="center">
  <strong>Empirical activity</strong>
  &nbsp;&rarr;&nbsp; differentiable brain model
  &nbsp;&rarr;&nbsp; gradient optimization
  &nbsp;&rarr;&nbsp; <strong>latent circuitry</strong>
</p>

## Why gradMFM

**Fit realistic brain dynamics.** The objective can combine complementary
features of empirical activity, including functional connectivity and its
temporal dynamics, so that optimization is driven by more than a single static
summary.

**Infer mechanisms, not only predictions.** Trainable quantities remain part of
the biophysical model. The fitted solution therefore provides candidate regional
and network-level mechanisms that can be inspected, compared, and tested.

**Train high-dimensional models progressively.** A staged optimization strategy
introduces difficult parameter families in a controlled sequence, making joint
inference more stable than attempting to fit every latent variable at once.

**Separate scientific design from execution.** Dataset settings, model choices,
and training schedules are decoupled from the optimization engine. The same
workflow can support individual- or group-level analyses across different
atlases, cohorts, and structural priors.

**Evaluate out of sample.** Training, validation, and test simulations have
distinct roles. Checkpoints are selected in validation space, while held-out
simulations are reserved for final assessment.

## Quick start

Create the released environment:

```bash
git clone https://github.com/BrianYahoo/gradMFM.git
cd gradMFM
conda env create -f requirements/environment.yml
conda activate gradmfm
```

Run the complete reference workflow:

```bash
cd code/bash
bash run.sh
bash post.sh
```

`run.sh` launches GPU training and `post.sh` performs validation and test. Review
the GPU ID and seed range at the top of each script before execution.

<p align="center">
  <a href="requirements/README.md"><strong>Environment guide</strong></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="code/bash/"><strong>Workflow scripts</strong></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="code/script/"><strong>Modeling source</strong></a>
</p>

## Released example

This repository intentionally provides one complete HCP Human Glasser example.
It demonstrates the full lifecycle of the framework, from empirical targets and
structural initialization to optimization, checkpoint selection, and held-out
simulation. The example is a reference implementation of the method, not a
restriction on the datasets or parcellations that `gradMFM` can support.

The public release includes the derived atlas labels, empirical FC/FCD
biomarkers, and structural inputs needed to run this example. Other datasets
analyzed in the accompanying study are not distributed in this repository. They
may be made available by the authors upon reasonable request, subject to the
applicable data-use agreements and institutional requirements.

## Adapting the framework

New studies can reuse the optimization engine while supplying their own atlas,
empirical observables, structural initializer, and experiment configuration.
This separation keeps the modeling workflow consistent while allowing the
scientific question, spatial scale, cohort, and trainable parameter set to
change.

The released code supports reproducible seed-wise execution and both individual-
and group-level modeling. Its components can also be extended with alternative
biophysical dynamics, observation models, objectives, or optimization schedules.

## Manuscript

### Multi-step gradient-based whole-brain modeling infers latent circuitry from resting-state fMRI

**Boran Yang, Xiaoyu Chen, Zhenyuan Jin, Douglas Zhou, and Songting Li**

The accompanying manuscript presents the scientific formulation, validation,
and applications of the framework. Citation metadata will be added when the
manuscript becomes publicly available.

## Contact

For scientific correspondence or data requests, contact Douglas Zhou
(`zdz@sjtu.edu.cn`) or Songting Li (`songting@sjtu.edu.cn`).
