# gradMFM

Macroscopic brain network modelling for connectome inference, functional
dynamics, and disease-associated circuit mechanisms.

This repository provides the computational framework for the study:

**Robust inference of brain connectome reveals network pathophysiology in
psychiatric disorders**

## Overview

Understanding how anatomical connectivity gives rise to large-scale brain
dynamics remains a central challenge in neuroscience. `gradMFM` is a multi-step
gradient-based optimization framework that treats the mean-field model (MFM) as
a trainable recurrent dynamical system. It infers latent structural
connectivity (SC) and regional circuit heterogeneity from empirical functional
connectivity (FC) and functional connectivity dynamics (FCD).

Rather than relying on diffusion tractography as a fixed anatomical substrate,
the framework initializes SC from an exponential distance rule (EDR) prior
constructed from atlas geometry, then reshapes this latent connectome through
functional constraints. The implementation uses BrainPy and JAX for
differentiable whole-brain dynamics, backpropagation through time (BPTT), and
GPU-accelerated parameter estimation.

The associated manuscript evaluates gradMFM across healthy human, major
depressive disorder (MDD), macaque, and marmoset datasets. Human analyses use
HCP resting-state fMRI under Glasser and Desikan parcellations, psychiatric
analyses fit disease-specific NC and MDD models under the HarvardOxford atlas,
and non-human primate analyses validate inferred SC against tracer-derived
projection matrices. The released repository contains a compact Human Glasser
example dataset and the source code needed to reproduce the core modelling
workflow.

## Key Features

- **Gradient-based connectome inference**: estimates latent SC and regional
  circuit parameters by fitting empirical FC and FCD targets.
- **BrainPy/JAX implementation**: uses differentiable dynamical systems,
  BPTT, automatic differentiation, JIT compilation, and GPU acceleration.
- **EDR-based structural prior**: constructs an initial connectome from
  distance-dependent atlas geometry rather than imposing tractography-specific
  biases.
- **Multi-step optimization curriculum**: progressively fits regional
  heterogeneity, introduces SC inference, adds a hemodynamic output layer, and
  refines the model with FC/FCD constraints.
- **Biophysical output layers**: uses a Volterra BOLD readout in the main
  pipeline and supports Balloon-Windkessel simulation for control analyses.
- **Dataset-aware configuration**: decouples species, atlas, connectivity
  metric, tractography approach, random seed, and training step.
- **Automated workflows**: bash entry points drive seed-wise and step-wise
  execution for scalable experiments.

## Scientific Scope

`gradMFM` is built around a generative question: which anatomical wiring pattern
is sufficient to reproduce the observed static and dynamic fMRI organization?
The framework treats the inferred connectome as a biologically constrained
latent variable: it starts from an EDR-derived spatial prior, optimizes regional
circuit heterogeneity and inter-regional coupling, and evaluates the resulting
model in BOLD-level FC/FCD space.

The modelling pipeline supports:

- inference of subject- or group-level structural connectomes;
- validation against FC and FCD biomarkers;
- cross-atlas and cross-species comparison of inferred parameters;
- tracer-based external validation in non-human primates;
- disease-specific analysis of altered structural projections, reduced
  hierarchical differentiation, and network pathophysiology.

## Repository Layout

```text
gradMFM/
  code/
    bash/
      run.sh                 # GPU training workflow, steps 0-3
      post.sh                # CPU validation/test workflow, steps 4-5
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
    atlas/
      human/
        Glasser/
          label.npy          # ROI labels for the Human Glasser atlas
    input/
      human/
        Glasser/
          fiber_count_edr.npy
          fiber_count_dti.npy
          fc.npy
          biomarkers.npz
  requirements/
    README.md                # runtime environment notes
    environment.yml          # conda environment template
    requirements.txt         # minimal direct dependencies
    requirements-lock.txt    # CUDA-enabled JAX runtime pins
```

The repository currently includes the Human Glasser example files used by the
default scripts. The paper-level framework also supports additional parcellation
and species settings when the corresponding atlas files, empirical FC/FCD
targets, and initial-connectome inputs are provided.

## Installation

The runtime files for this source-code release are recorded in `requirements/`.

To create the repository environment:

```bash
conda env create -f requirements/environment.yml
conda activate gradmfm
```

For an existing environment, install the minimal direct dependencies:

```bash
pip install -r requirements/requirements.txt
```

For the pinned CUDA-enabled JAX backend used by the release:

```bash
pip install -r requirements/requirements-lock.txt
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

Atlas labels are stored separately:

```text
data/atlas/<species>/<atlas>/label.npy
```

In the paper, EDR initialization is constructed from atlas geometry and labels.
The compact release stores the resulting ROI-level initial matrices under
`data/input/` for direct use by the training scripts.

This source-code release publicly includes only the HCP Human Glasser subset as
a complete runnable example of the framework. Other datasets analyzed in the
manuscript are not distributed with this repository; they may be made available
from the authors upon reasonable request, subject to the corresponding data-use
agreements and institutional requirements.

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
visualization, hierarchy analyses, tracer-alignment tests, and disease-associated
projection mapping.

## Manuscript Abstract

Understanding how anatomical connectivity gives rise to large-scale brain
dynamics remains a central challenge in neuroscience. Here we introduce
gradMFM, a multi-step gradient-based optimization framework that infers latent
structural connectivity (SC) and regional circuit heterogeneity from empirical
functional connectivity (FC) and its dynamics (FCD). The optimized regional
parameters preserve a consistent hierarchical organization across cortical
parcellations, supporting biological interpretability. Cross-species validation
in macaque and marmoset datasets shows that gradMFM-inferred SC aligns more
closely with tracer-derived projections than diffusion MRI estimates,
particularly for long-range connections. Applying gradMFM to major depressive
disorder (MDD), we construct disease-specific whole-brain models that reproduce
disease-specific FC abnormalities and reveal reduced hierarchical
differentiation together with altered projections. Together, these findings
establish a principled route for inferring hidden anatomical architecture from
spontaneous brain activity and for linking structural coupling, functional
dynamics, and disease-specific network pathology.

## Citation

Citation information will be added when the manuscript is publicly available.
