# CaTFold

This repository contains code for replicating results from the associated paper:

Cheng Wang, Yang Liu, Gaurav Sharma, "CaTFold: Improving RNA secondary structure prediction by incorporating prior information"

## Dependencies

The program requires Python 3.8.x and the following packages:

* PyTorch v2.0.1
* tensorboardx v2.6.2.2
* munch v4.0.0
* networkx v3.0
* numpy v1.23.1
* scipy v1.10.1
* tqdm v4.66.1
* matplotlib v3.7.3
* seaborn v0.13.2
* pandas v2.0.3

GPU is highly recommended.

## Data preparation

Clone this repository, unzip data.zip and organize it into the following path:

`CaTFold/data/ArchiveII/all.pickle`

`CaTFold/data/RNAStralign/train_filtered.pickle`

`CaTFold/data/RNAStralign/test.pickle`

## Data preprocessing

```bash
python3 code/data_preprocess.py
```

This command will generate the following new files:

`CaTFold/data/ArchiveII/max600.pickle`

`CaTFold/data/RNAStralign/train_filtered_max600.pickle`

`CaTFold/data/RNAStralign/test_max600.pickle`

`CaTFold/data/ArchiveII/legal_pairs_all`

`CaTFold/data/ArchiveII/legal_pairs_max600`

`CaTFold/data/RNAStralign/legal_pairs_train_filtered`

`CaTFold/data/RNAStralign/legal_pairs_train_filtered_max600`

`CaTFold/data/RNAStralign/legal_pairs_test`

`CaTFold/data/RNAStralign/legal_pairs_test_max600`

## Evaluate CaTFold

```bash
python3 code/inference.py
```

This command measures the performance of the checkpoint on The ArchiveII and RNAStralign_test dataset, and outputs the result to 'results/output.txt' :

### Using Code Ocean

We recommend using the Code Ocean version of this program, which can be run using Code Ocean's built-in interface. (Site: xxx)

## Training CaTFold

Pre-training CaTFold:

```bash
python3 code/pretrain_ddp.py
```

Refining CaTFold:

```bash
python3 code/refine_ddp.py
```
