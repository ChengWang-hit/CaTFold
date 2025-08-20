# Source Code
This directory contains all the source code for the CaTFold project.

The main components are organized into the following directories:

*   **`pretrain/`**: Contains the scripts for training the model from scratch on the primary dataset.

*   **`finetune/`**: Contains the scripts for adapting a pre-trained model to a new or specific dataset.

*   **`inference_fasta/`**: Contains the scripts for predicting secondary structures from a `.fasta` file. The code is designed to handle files containing one or more sequences, and for each sequence, it generates two outputs: a structural prediction saved in the standard **`.bpseq` format** and a **visualization** of the predicted structure allowing for immediate visual inspection of the results.

*   **`prepare_data.py`**: A script for processing and preparing your data for fine-tuning. It is designed to parse a directory of `.bpseq` files, extract sequences and their corresponding structures, and convert them into `.pickle` format.

---
**Note:** For detailed instructions on how to run the code in each directory, including command-line arguments and examples, please refer to the main `README.md` file in the project's root directory.