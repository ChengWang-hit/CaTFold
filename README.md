# CaTFold: An Efficient Deep Learning Approach for RNA Secondary Structure Prediction
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/871105419.svg)](https://doi.org/10.5281/zenodo.13937606)
[![Code Ocean](https://8277274.fs1.hubspotusercontent-na1.net/hubfs/8277274/Code%20Ocean%20U4%20Theme%20Assets/code-ocean-logo-white.svg)](xxx)

This repository contains the official source code and models for the paper:

> **CaTFold: An Efficient Deep Learning Approach for Out-of-Family RNA Secondary Structure Prediction**
>
> *Cheng Wang, Gaurav Sharma, Haozhuo Zheng, Ping Li, Yang Liu*
>
<!-- > **[Link to Paper Coming Soon]** -->

---

## Table of Contents
- [Overview](#overview)
- [Setup and Installation](#setup-and-installation)
- [Quick Start: Prediction on Your Own Data](#quick-start-prediction-on-your-own-data)
- [Reproducing Our Results](#reproducing-our-results)
  - [Fine-tuning and Evaluation](#fine-tuning-and-evaluation)
  - [Pre-training from Scratch](#pre-training-from-scratch)
- [Reproducibility with Code Ocean](#reproducibility-with-code-ocean)
- [Citation](#citation)
- [License](#license)

---

## Overview
CaTFold is a deep learning-based method designed for accurate and efficient prediction of RNA secondary structures, with a strength in handling out-of-family sequences. This repository provides all the necessary tools to use our pre-trained models, reproduce the results from our paper, and train or fine-tune the model on new datasets.

---

## Setup and Installation
Follow these steps to set up the environment. A GPU is highly recommended for both training and inference.

**1. Clone the Repository**
```bash
git clone https://github.com/ChengWang-hit/CaTFold.git
cd CaTFold
```

**2. Install Dependencies**
We recommend creating a virtual environment (e.g., using `conda`). All required packages are listed in `requirements.txt`.

```bash
# Create and activate a conda environment (optional but recommended)
conda create -n catfold python=3.8
conda activate catfold

# Install packages
pip install -r requirements.txt
```

**3. Download Datasets**
All datasets required for training and evaluation are hosted on Zenodo. Please see the detailed instructions in the `data/README.md` file for download links and setup.

**4. Download Pre-trained Checkpoints**
We provide pre-trained model checkpoints to allow for immediate inference and fine-tuning. Please see the instructions in the `checkpoints/README.md` file for download links.

---

## Quick Start: Prediction on Your Own Data

1.  **Prepare your input file**: Ensure your sequences are in a `.fasta` file. The file can contain one or more sequences.

2.  **Configure the paths**: Open the `code/inference_fasta/config.json` file and modify the `fasta_path`,  `output_dir` and `checkpoint_path`.
    ```json
    {
      "fasta_path": "path_to_your_input.fasta",
      "output_dir": "path_to_your_output_directory",
      "checkpoint_path": "path_to_checkpoint"
    }
    ```

3.  **Run the prediction script**:
    ```bash
    python code/inference_fasta/inference.py
    ```
    The predicted secondary structures will be saved as `.bpseq` files in your specified output directory.

---

## Reproducing Our Results
This section details how to reproduce the main experiments from our paper.

### Fine-tuning and Evaluation
**1. Reproduce Paper Results**

To evaluate our checkpoints on all benchmark datasets and reproduce the main results reported in the paper, simply run the following script. Ensure that all datasets and checkpoints have been downloaded and placed correctly.

```bash
python code/finetune/test_all_dataset.py
```

**2. Fine-tune on Datasets in the Paper**

To fine-tune the CaTFold model on dataset, use the following command. You will need to prepare your data and create a corresponding configuration file.

```bash
python path_to_dataset_code.py
```

### Pre-training from Scratch
To pre-train the CaTFold model from scratch (including the warm-up and main pre-training stages), use the following command. This requires a multi-GPU setup.

First, ensure your pre-training dataset path is correctly specified in the training configuration file. Then, run the distributed training script:

```bash
# This example uses 2 GPUs (0 and 1)
torchrun --standalone --nproc_per_node=2 code/pretrain/pretrain_ddp.py
```

---

## Reproducibility with Code Ocean
We have also made our work available as a Code Ocean Compute Capsule. This provides a pre-configured environment with all dependencies, data, and code, allowing you to run our main experiments with a single click.

> **Access the Capsule: [xxx](xxx)**

---

<!-- ## Citation
If you find CaTFold useful in your research, please consider citing our paper:

```bibtex
@article{wang2023catfold,
  title={CaTFold: An Efficient Deep Learning Approach for Out-of-Family RNA Secondary Structure Prediction},
  author={Wang, Cheng and Sharma, Gaurav and Zheng, Haozhuo and Li, Ping and Liu, Yang},
  journal={Journal or Conference Name},
  year={2023},
  volume={XX},
  pages={YY-ZZ},
  doi={Your Paper's DOI}
}
``` -->
<!--  -->
<!-- --- -->

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.