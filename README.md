# gradMFM

Macroscopic brain network modelling for connectome inference, functional
dynamics, and disease-associated circuit mechanisms.

This repository provides the computational framework for the study:

**Robust inference of brain connectome reveals disrupted wiring mechanisms in
psychiatric disorders**

## Overview

Understanding how anatomical connectivity shapes large-scale brain dynamics is a
central problem in systems neuroscience. `gradMFM` implements a
machine-learning-based framework that infers the structural connectome by
optimizing a biophysical macroscopic model against empirical resting-state fMRI
features. Starting from empirical structural connectivity (SC), the model fits
static functional connectivity (FC) and dynamic functional connectivity (FCD)
biomarkers through gradient-based optimization in BrainPy and JAX.

The framework is designed for high-throughput analysis of HCP and Parkinson's
disease cohorts, and for extension to psychiatric-disorder datasets. In the
associated manuscript, inferred connectomes support realistic simulations of
resting-state FC and FCD, improve correspondence with tracer-derived
connectivity in cross-species validation, preserve hierarchical organization
across atlases, and reveal altered wiring mechanisms in major depressive
disorder.

## Key Features

- **Gradient-based connectome inference**: estimates SC and neural parameters by
  fitting empirical FC and FCD targets.
- **BrainPy/JAX implementation**: uses differentiable dynamical systems,
  automatic differentiation, JIT compilation, and GPU acceleration.
- **Multi-stage optimization**: separates pretraining, FC-constrained SC
  inference, FC refinement, and joint FC/FCD fitting.
- **Biophysical output layers**: supports linear, Volterra, and Balloon-style
  readouts for simulated BOLD-like activity.
- **Dataset-aware configuration**: decouples species, atlas, connectivity
  metric, tractography approach, random seed, and training step.
- **Automated workflows**: bash entry points drive seed-wise and step-wise
  execution for scalable experiments.

## Scientific Scope

`gradMFM` is built around a generative question: which anatomical wiring pattern
is sufficient to reproduce the observed static and dynamic fMRI organization?
Rather than treating diffusion tractography as a fixed ground truth, the
framework uses empirical SC as an initialization and infers a refined structural
connectome that better explains functional dynamics.

The modelling pipeline supports:

- inference of subject- or group-level structural connectomes;
- validation against FC and FCD biomarkers;
- cross-atlas and cross-species comparison of inferred parameters;
- disease-specific analysis of altered structural projections and disrupted
  hierarchical differentiation.

## Repository Layout

```text
gradMFM/
  code/
    bash/
      run.sh                 # GPU training workflow, steps 0-3
      post.sh                # CPU validation/test workflow, steps 4-5
    requirements/
      README.md
      environment.yml
      requirements.txt
      requirements-lock.txt
    script/
      running.py             # main execution entry point
      func_settings.py       # experiment settings and data loading
      set_hmG.py             # Human Glasser training schedule
      func_dyn.py            # neural mass dynamics
      func_model.py          # MFM model wrappers
      func_loss.py           # FC/FCD objective functions
      func_metrics.py        # FC/FCD metrics and parameter checks
      func_train.py          # optimization loop
      func_vali.py           # validation routine
      func_test.py           # test and visualization routine
      func_out.py            # output-layer dynamics
  data/
    input/
      human/
        Glasser/
          fiber_count_edr.npy
          fiber_count_dti.npy
          fc.npy
          biomarkers.npz
```

## Installation

The package versions used in the local `brainpy` conda environment are recorded
in `code/requirements/`.

To create a comparable environment:

```bash
conda env create -f code/requirements/environment.yml
conda activate gradmfm
```

For an existing environment, install the minimal direct dependencies:

```bash
pip install -r code/requirements/requirements.txt
```

For the pinned CUDA-enabled JAX backend observed in the source environment:

```bash
pip install -r code/requirements/requirements-lock.txt
```

## Data Convention

Input files are expected under:

```text
data/input/<species>/<atlas>/
```

For the Human Glasser example, the current scripts expect:

```text
data/input/human/Glasser/fiber_count_<approach>.npy
data/input/human/Glasser/fc.npy
data/input/human/Glasser/biomarkers.npz
```

For example, with `metric=fiber_count` and `approach=edr`, the SC initializer is:

```text
data/input/human/Glasser/fiber_count_edr.npy
```

## Running the Pipeline

The main entry point is:

```bash
python ../script/running.py <gpu_id> <species> <atlas> <metric> <approach> <seed> <step>
```

The scripts use relative paths to `../../data`, so run them from `code/bash` or
`code/script`, not from the repository root.

Example from `code/bash`:

```bash
cd code/bash
python ../script/running.py 0 human Glasser fiber_count edr 1 0
```

The provided bash workflows execute the full Human Glasser schedule over seeds:

```bash
cd code/bash
bash run.sh
bash post.sh
```

The default Human Glasser stages are:

| Step | Name | Objective | Trainable variables | Output layer |
| ---: | --- | --- | --- | --- |
| 0 | `pretrain` | FC | `G`, `w`, `I`, `sigma` | linear |
| 1 | `train-conn-ac` | FC | `SC`, `I`, `sigma` | linear |
| 2 | `train-conn-fc` | FC | `SC`, `I`, `sigma` | Volterra |
| 3 | `train-conn-fcd` | FC + FCD | `SC`, `I`, `sigma` | Volterra |
| 4 | `validation` | held-out simulation | none | configured from checkpoint |
| 5 | `test` | final evaluation and figures | none | configured from checkpoint |

Results are written under `data/results/`, and figures under `figures/results/`,
relative to the repository root when scripts are launched from `code/bash` or
`code/script`.

## Outputs

Training checkpoints are saved as BrainPy pytrees (`.bp`) and include:

- model, loss, and optimizer states;
- epoch-wise loss curves;
- FC/FCD fit metrics;
- inferred global coupling `G`;
- local recurrent weights `w`;
- regional input currents `I`;
- noise amplitudes `sigma`;
- inferred structural connectivity `SC`.

These outputs provide the basis for group-level comparison, connectome
visualization, hierarchy analyses, and disease-associated projection mapping.

## Manuscript Abstract

Understanding how anatomical connectivity shapes large-scale brain dynamics
remains a major challenge. In this study, we propose a machine-learning-based
framework that can infer the structural connectome by fitting empirical fMRI
characteristics. With inferred connectome, the biophysical model can generate
highly realistic static and dynamic resting-state functional connectivity (FC).
We then perform cross-species validation to confirm that the inferred connectome
matches tracer-derived connectivity more closely than diffusion tractography,
particularly for long-range projections. The cross-atlas validation also shows
that the inferred parameters exhibit a consistent hierarchical organization,
supporting the framework's biological plausibility. Applying the same framework
to major depressive disorder (MDD), we recover group-specific FC alterations and
identify reduced hierarchical differentiation together with altered structural
projections from MDD-related cortical regions. These findings position the
framework as a principled approach for linking structural architecture,
functional dynamics, and disease-associated network alterations.

## Citation

Citation information will be added when the manuscript is publicly available.

