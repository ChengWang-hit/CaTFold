# Datasets
This directory is the location for all datasets required for pre-training, fine-tuning, and evaluating the CaTFold model.

---

## Setup Instructions
1.  **Download the Datasets**

    All necessary datasets are available for download from [![Zenodo](https://img.shields.io/badge/Zenodo-10.5281/zenodo.13937606-blue)](https://doi.org/10.5281/zenodo.13937606).

2.  **Unzip the Archives**

    After downloading, unzip each `.tar.gz` archive. You can use the following command in your terminal:
    ```bash
    tar -xzvf <filename>.tar.gz
    ```

3.  **Place Folders Here**

    Place all the unzipped folders directly into this `data/` directory.

---

## Final Directory Structure
After you have downloaded and unzipped all the archives, the structure of this `data/` directory should look like this:
```
data/
├── ArchiveII/
├── bpRNA_1m/
├── bpRNA_new/
├── data_for_pretrain/
├── RNAStralign/
└── README.md
```

## Creat Your Own Data for Fine-Tuning

In addition to the provided datasets, you can easily prepare your own custom data for fine-tuning the pretrained CaTFold.

Follow these steps:

1.  **Create a Dataset Folder**
    
    Inside this `data/` directory, create a new folder for your custom dataset.

2.  **Add Your Data**
    
    Place all your RNA sequences in the **`.bpseq` format** inside a subdirectory named `bpseq/`. The structure should look like this:
    ```
    data/
    └── your_dataset/
        └── bpseq/
            ├── rna_1.bpseq
            ├── rna_2.bpseq
            └── ...
    ```

3.  **Run the Preparation Script**
    
    Execute the `code/prepare_data.py` script to process your files. You will need to **modify the script** to point to your new directory.

    After running the script, it will generate the necessary `.pickle` file and `mask_matrix/` directory inside `data/your_dataset/`, which can then be used for fine-tuning.


**Note:** For details on how these datasets are used in the training and evaluation pipelines, please refer to the main `README.md` in the project's root directory.