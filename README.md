<p align="center">
  <img src="assets/gradMFM.png" alt="gradMFM" width="260">
</p>

<p align="center">
  <strong>From resting-state dynamics to latent whole-brain circuitry</strong>
</p>

<p align="center">
  A differentiable biophysical modeling framework for jointly inferring regional
  circuit heterogeneity and model-constrained structural connectivity.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9">
  <img src="https://img.shields.io/badge/JAX-0.4-7E57C2" alt="JAX 0.4">
  <img src="https://img.shields.io/badge/BrainPy-2.6-00A6A6" alt="BrainPy 2.6">
  <img src="https://img.shields.io/badge/GPU-CUDA-76B900?logo=nvidia&logoColor=white" alt="CUDA accelerated">
</p>

<p align="center">
  <a href="#vision">Vision</a> ·
  <a href="#research-scope">Research scope</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#data-release">Data</a> ·
  <a href="#manuscript">Manuscript</a>
</p>

---

## Vision

Whole-brain functional connectivity is easy to observe but difficult to explain.
`gradMFM` turns this problem into a differentiable inverse model: resting-state
FC and FCD constrain a biophysical mean-field system whose regional properties
and long-range coupling can be inferred together.

The framework combines a geometry-derived structural prior, freely varying
regional circuit parameters, hemodynamic modeling, and staged gradient-based
optimization. The result is not a single descriptive network statistic, but a
candidate latent circuit model that can be examined for functional fidelity,
solution consistency, and anatomical correspondence.

<p align="center">
  <strong>Resting-state fMRI</strong>
  &nbsp;&rarr;&nbsp; FC and FCD
  &nbsp;&rarr;&nbsp; differentiable MFM
  &nbsp;&rarr;&nbsp; regional heterogeneity and latent SC
</p>

<br>

<table>
  <tr>
    <td align="center" width="25%">
      <strong>Biophysical</strong><br>
      <sub>Interpretable neural-mass dynamics rather than a black-box predictor</sub>
    </td>
    <td align="center" width="25%">
      <strong>Differentiable</strong><br>
      <sub>BrainPy and JAX enable end-to-end optimization through time</sub>
    </td>
    <td align="center" width="25%">
      <strong>High-dimensional</strong><br>
      <sub>Regional heterogeneity and structural coupling are inferred jointly</sub>
    </td>
    <td align="center" width="25%">
      <strong>Evaluable</strong><br>
      <sub>Functional fit, recovery, consistency, and tracer correspondence</sub>
    </td>
  </tr>
</table>

## Research scope

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>Human cortex</h3>
      HCP analyses across Glasser and Desikan-Killiany parcellations test
      functional reconstruction and the organization of inferred regional
      parameters along cortical hierarchy.
    </td>
    <td width="33%" valign="top">
      <h3>Cross-species anatomy</h3>
      Macaque and marmoset models provide an external anatomical benchmark by
      comparing inferred coupling with tracer-derived projections, including
      long-range connections.
    </td>
    <td width="33%" valign="top">
      <h3>Computational psychiatry</h3>
      Group-level MDD models reproduce observed functional differences and
      generate exploratory hypotheses about latent circuit organization.
    </td>
  </tr>
</table>

The accompanying study further evaluates the framework through multi-step
ablation, synthetic parameter recovery, cross-run consistency, and comparisons
with established whole-brain optimization approaches.

## Quick start

Create the released environment:

```bash
git clone https://github.com/BrianYahoo/gradMFM.git
cd gradMFM
conda env create -f requirements/environment.yml
conda activate gradmfm
```

Run the complete Human Glasser example:

```bash
cd code/bash
bash run.sh
bash post.sh
```

`run.sh` performs GPU training and `post.sh` performs validation and test. The
released scripts are configured for repeated optimization runs; review the GPU
ID and seed range at the top of each script before execution.

Model selection is performed entirely in validation space using FC correlation,
FC mean-squared error, and FCD Kolmogorov-Smirnov distance. Test simulations are
reserved for independent evaluation.

<p align="center">
  <a href="requirements/README.md"><strong>Environment guide</strong></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="code/bash/"><strong>Workflow scripts</strong></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="code/script/"><strong>Source code</strong></a>
</p>

## Data release

This repository includes the derived HCP Human Glasser subset required for a
complete runnable demonstration, including the atlas labels, empirical FC and
FCD biomarkers, and EDR- and DTI-derived structural inputs.

The manuscript additionally analyzes HCP Desikan-Killiany,
REST-meta-MDD Harvard-Oxford, macaque Markov, and marmoset MBMv3/Paxinos data.
These datasets are not distributed with this repository and may be made
available by the authors upon reasonable request, subject to the applicable
data-use agreements and institutional requirements.

## Manuscript

### Multi-step gradient-based whole-brain modeling infers latent circuitry from resting-state fMRI

**Boran Yang, Xiaoyu Chen, Zhenyuan Jin, Douglas Zhou, and Songting Li**

The study presents `gradMFM` as a differentiable route from non-invasive
functional dynamics to latent circuit models, with evaluation across functional
fidelity, synthetic recovery, solution consistency, cortical hierarchy,
cross-species tracer correspondence, and group-level MDD modeling.

## Citation and contact

Citation metadata will be added when the manuscript becomes publicly available.

For scientific correspondence or data requests, contact Douglas Zhou
(`zdz@sjtu.edu.cn`) or Songting Li (`songting@sjtu.edu.cn`).
