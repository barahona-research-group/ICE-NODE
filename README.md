This folder contains four notebooks to reproduce the results in the `ICE-NODE` paper.

**Paper title:** _`ICE-NODE`: Integration of Clinical Embeddings with Neural Ordinary Differential Equations._


**Abstract:**
> Early diagnosis of disease can result in improved health outcomes, such as higher survival rates and lower treatment costs. With the massive amount of information in electronic health records (EHRs), there is great potential to use machine learning (ML) methods to model disease progression aimed at early prediction of disease onset and other outcomes. In this work, we employ recent innovations in neural ODEs to harness the full temporal information of EHRs. We propose ICE-NODE (Integration of Clinical Embeddings with Neural Ordinary Differential Equations), an architecture that temporally integrates embeddings of clinical codes and neural ODEs to learn and predict patient trajectories in EHRs. We apply our method to the publicly available MIMIC-III and MIMIC-IV datasets, reporting improved prediction results compared to state-of-the-art methods, specifically for clinical codes that are not frequently observed in EHRs. We also show that ICE-NODE is more competent at predicting certain medical conditions, like acute renal failure and pulmonary heart disease, and is also able to produce patient risk trajectories over time that can be exploited for further predictions.

![](https://raw.githubusercontent.com/A-Alaa/ICENODE/main/figures/figure1.svg)

## Cloning ICE-NODE (paper version)

The current codebase is under development for refactoring and future extensions. Therefore, this notebook clones a particular snapshot (Git Tag) for the code version that is reproducing our work described in the paper.


Having `git` installed and the main repository cloned, change the command line directory to `MLHC_experiments` and run the following command to clone the snapshot:


```bash
$ git clone git@github.com:A-Alaa/ICE-NODE.git repo --branch main --single-branch  --depth 1
```

## Installing Dependencies

Dependencies of `ICE-NODE` and other libraries used in analyses are contained in a Conda Environment file (except JAX library, which is installed separately). 
Having Conda installed, run the following command to install the environment file in the same folder `MLHC_experiments`.


```bash
$ conda env create -f repo/env/icenode.yml --prefix ./icenode-env
```

Now activate the newly installed environment.


```bash
$ conda activate ./icenode-env
```


### Installing JAX library


JAX is the deep learning engine that we use in `ICE-NODE`, with version `0.3.13`. To install JAX, please follow the instructions on the JAX main repository: [google/jax](https://github.com/google/jax).


## Notebooks


To use one of the notebooks provided, please ensure that the newly installed environment, which also has JAX installed, is activated first. We provide four notebooks in this folder, each carries out the following tasks:


### Preparation of MIMIC-III and MIMIC-IV

- Notebook 1 (`notebooks/dx_mimic3_preparation.ipynb`): Preparation of MIMIC-III dataset
- Notebook 2 (`notebooks/dx_mimic4_preparation.ipynb`): Preparation of MIMIV-IV dataset

### Training and Performance Analysis

- Notebook 3 (`notebooks/dx_training.ipynb`): Training `ICE-NODE` and the baselines.
- Notebook 4 (`notebooks/dx_fine_analysis.ipynb`): Performance Analysis and Trajectory Reconstruction.


Note: We provide pretrained models in this repository since some training experiments (such as RETAIN on MIMIC-IV) takes more than two weeks. You may use our pretrained models in the performance analysis as instructed in `fine_analysis.ipynb`.

