# Runtime Environment

This directory defines the Python runtime for the released `gradMFM` source
code. The dependency set is intentionally scoped to the modelling, optimization,
validation, and visualization scripts in `code/script`.

Use `environment.yml` to create a fresh conda environment named `gradmfm`.
The accompanying requirements files provide pip-style alternatives for minimal
installation or for reproducing the CUDA-enabled JAX backend.

## Files

- `requirements.txt`: minimal direct runtime dependencies for the scripts.
- `requirements-lock.txt`: direct dependencies plus key backend packages,
  including the CUDA-enabled JAX runtime used by the release.
- `environment.yml`: conda environment template for creating the `gradmfm`
  runtime with Python 3.9.

The lock file and environment template include the JAX CUDA wheel index because
the pinned `jaxlib` version is `0.4.28+cuda12.cudnn89`.

## Direct Runtime Imports

- `brainpy`
- `brainpy.math`
- `jax`
- `matplotlib`
- `numpy`
- `pandas`
- `seaborn`
- `tqdm`

The remaining imports are from the Python standard library or this repository.

## Recommended Setup

Create and activate the repository runtime:

```bash
conda env create -f code/requirements/environment.yml
conda activate gradmfm
```

For an existing environment, install the minimal package set:

```bash
pip install -r code/requirements/requirements.txt
```

For a CUDA 12 JAX setup matching the release pins:

```bash
pip install -r code/requirements/requirements-lock.txt
```

CPU-only users should install the platform-appropriate `jaxlib` build from the
official JAX installation instructions instead of the CUDA-specific wheel pinned
in `requirements-lock.txt`.

## Verification Commands

After activating `gradmfm`, verify package-level imports with:

```bash
python -c "import brainpy, jax, numpy, pandas, matplotlib, seaborn, tqdm"
```

Verify the repository entry point from the repository root:

```bash
python -c "import sys; sys.path.insert(0, 'code/script'); import func_settings, running"
```
