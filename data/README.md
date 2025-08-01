# Datasets
This directory is the location for all datasets required for pre-training, fine-tuning, and evaluating the CaTFold model.

---

## Setup Instructions
1.  **Download the Datasets**

    All necessary datasets are available for download from [Zenodo URL].

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


**Note:** For details on how these datasets are used in the training and evaluation pipelines, please refer to the main `README.md` in the project's root directory.