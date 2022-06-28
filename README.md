
Integration of Clinical Embeddings with Neural Ordinary Differential Equations.


**Abstract from MLHC 2022 Paper:**
> Early diagnosis of disease can result in improved health outcomes, such as higher survival rates and lower treatment costs. With the massive amount of information in electronic health records (EHRs), there is great potential to use machine learning (ML) methods to model disease progression aimed at early prediction of disease onset and other outcomes. In this work, we employ recent innovations in neural ODEs to harness the full temporal information of EHRs. We propose ICE-NODE (Integration of Clinical Embeddings with Neural Ordinary Differential Equations), an architecture that temporally integrates embeddings of clinical codes and neural ODEs to learn and predict patient trajectories in EHRs. We apply our method to the publicly available MIMIC-III and MIMIC-IV datasets, reporting improved prediction results compared to state-of-the-art methods, specifically for clinical codes that are not frequently observed in EHRs. We also show that ICE-NODE is more competent at predicting certain medical conditions, like acute renal failure and pulmonary heart disease, and is also able to produce patient risk trajectories over time that can be exploited for further predictions.

![](https://raw.githubusercontent.com/barahona-research-group/ICE-NODE/main/figures/figure1.svg)

## Cloning ICE-NODE (MLHC 2022 version)

The current codebase is under development for refactoring and future extensions. Therefore, this notebook clones a particular snapshot (Git Tag) for the code version that is reproducing our work described in the paper.


Having `git` installed and the main repository cloned, change the command line directory to `notebooks` and run the following command to switch to `mlhc2022` version:


```bash
$ git checkout mlhc2022
```

## Installing Dependencies


**Step 1**: dependencies of `ICE-NODE` and other libraries used in analyses are contained in a `requirements.txt` file (except JAX and JAXopt libraries, which are installed separately). 
Having Conda installed, run the following commands to create a new environment with the needed libraries.


```bash
$ conda create -n mlhc-env python=3.8.12
```

**Step 2**: now activate the newly installed environment.


```bash
$ conda activate mlhc-env
```

**Step 3**: with `pip`, install the needed libraries except JAX and JAXopt.

```bash
$ pip install -r requirements/mlhc2022.txt
```

**Step 4**: install JAX library. JAX is the deep learning engine that we use in `ICE-NODE`, with version `0.3.13`. To install JAX, please follow the instructions on the JAX main repository: [google/jax](https://github.com/google/jax).

**Step 5**: install JAXopt.

```bash
$ pip install jaxopt
```

## Notebooks


To use one of the notebooks provided, please ensure that the newly installed environment, which also has JAX installed, is activated first. We provide four notebooks in this folder, each carries out the following tasks:


### Preparation of MIMIC-III and MIMIC-IV

- Notebook 1 (`notebooks/dx_mimic3_preparation.ipynb`): Preparation of MIMIC-III dataset
- Notebook 2 (`notebooks/dx_mimic4_preparation.ipynb`): Preparation of MIMIV-IV dataset

### Training and Performance Analysis

- Notebook 3 (`notebooks/dx_training.ipynb`): Training `ICE-NODE` and the baselines.
- Notebook 4 (`notebooks/dx_fine_analysis.ipynb`): Performance Analysis and Trajectory Reconstruction.

