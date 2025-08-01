# Source Code
This directory contains all the source code for the CaTFold project.

The main components are organized into the following directories:

*   **`pretrain/`**: Contains the scripts for training the model from scratch on the primary dataset.

*   **`finetune/`**: Contains the scripts for adapting a pre-trained model to a new or specific dataset.

*   **`inference_fasta/`**: Contains the scripts for predicting secondary structures from a `.fasta` file. The code is designed to handle files containing one or more sequences, and for each sequence, it generates a corresponding prediction saved in the `.bpseq` format.

---
**Note:** For detailed instructions on how to run the code in each directory, including command-line arguments and examples, please refer to the main `README.md` file in the project's root directory.