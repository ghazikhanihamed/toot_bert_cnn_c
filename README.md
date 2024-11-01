# TooT-PLM-ionCT: Protein Language Model-based Framework for Ion Channel and Ion Transporter Classification

[![DOI](https://img.shields.io/badge/DOI-10.1002/prot.26694-blue)](https://doi.org/10.1002/prot.26694)  
*Hamed Ghazikhani, Gregory Butler*  
*First published: 24 April 2024*  

## Overview

This repository contains the implementation of **TooT-PLM-ionCT**, a framework developed for precise classification of ion channels (ICs) and ion transporters (ITs) from membrane proteins (MPs) using Protein Language Models (PLMs). TooT-PLM-ionCT is tailored for three primary classification tasks:

1. **IC vs. MP** - Distinguishing ion channels from other membrane proteins.
2. **IT vs. MP** - Separating ion transporters from other membrane proteins.
3. **IC vs. IT** - Differentiating between ion channels and ion transporters.

TooT-PLM-ionCT leverages six PLMs (ProtBERT, ProtBERT-BFD, ESM-1b, ESM-2 with 650M and 15B parameters) and employs both traditional classifiers and deep learning models to enhance classification accuracy and robustness.

## Abstract

In this study, we introduce TooT-PLM-ionCT, a comprehensive framework for protein classification tasks focused on ion channels and transporters. Our system consolidates three distinct models tailored to (1) identify ICs from MPs, (2) segregate ITs from MPs, and (3) differentiate ICs from ITs. Built on six PLMs, TooT-PLM-ionCT was validated on existing datasets, achieving state-of-the-art results in IC-MP classification and showing superior performance in other tasks.

We introduced a new dataset for enhanced model robustness and generalization across bioinformatics challenges. Results indicate that TooT-PLM-ionCT adapts well to novel data, with high classification accuracy. Additionally, we examined the influence of dataset balancing, PLM fine-tuning, and floating-point precision on model performance.

A web server for public access is available at [TooT-PLM-ionCT Web Server](https://tootsuite.encs.concordia.ca/service/TooT-PLM-ionCT), allowing users to test protein sequences for the IC-MP, IT-MP, and IC-IT classification tasks.

## Models and Data

### Protein Language Models (PLMs)

The following PLMs are used in this framework:

- **ProtBERT**
- **ProtBERT-BFD**
- **ESM-1b**
- **ESM-2 (650M)**
- **ESM-2 (15B)**

Each model can be configured for either frozen or fine-tuned representations, with precision settings (half or full) adjustable in `config.yaml`.

### Dataset

The models were originally validated on a public dataset used by previous researchers and evaluated on an additional dataset introduced in this study for improved robustness.

## Results

TooT-PLM-ionCT demonstrated high accuracy across tasks, with state-of-the-art results in IC-MP discrimination. The study also identified critical factors that influence classification accuracy, including:

- Dataset balancing
- Frozen vs. fine-tuned PLM representations
- Floating-point precision (half vs. full)

## Web Server

A web interface for TooT-PLM-ionCT is available at [TooT-PLM-ionCT Web Server](https://tootsuite.encs.concordia.ca/service/TooT-PLM-ionCT), allowing users to classify protein sequences interactively.

## Citation

If you use TooT-PLM-ionCT in your research, please cite our paper:

```plaintext
@article{ghazikhani2024tootplmionct,
  title={Exploiting protein language models for the precise classification of ion channels and ion transporters},
  author={Hamed Ghazikhani and Gregory Butler},
  journal={Proteins: Structure, Function, and Bioinformatics},
  volume={92},
  Issue={8},
  pages={998--1055},
  year={2024},
  publisher={Wiley},
  doi={10.1002/prot.26694}
}
