# CaTFold

[![GitHub Source Code](https://img.shields.io/badge/GitHub-Source%20Code-blue?logo=github)](https://github.com/ChengWang-hit/CaTFold) [![Code Ocean DOI](https://img.shields.io/badge/Code%20Ocean-10.24433%2FCO.8228673.v1-blue)](https://doi.org/10.24433/CO.8228673.v1)

CaTFold is a deep-learning method for RNA secondary-structure prediction. This repository provides the code and configurations required for benchmark evaluation, FASTA inference, pretraining, and fine-tuning. Model checkpoints, benchmark datasets, and pretraining sequences are distributed separately through Cloudflare R2 because of their size.

This repository accompanies the study:

**CaTFold: a lightweight pretrained model for RNA secondary structure prediction**

Cheng Wang, Gaurav Sharma, Haozhuo Zheng, Ning Wang, Ping Li, and Yang Liu.

## Repository structure

```text
CaTFold/
├── code/                  # model, training, evaluation, and inference code
├── configs/               # runnable JSON configurations
├── demo/                  # example FASTA file and example predictions
├── results/               # released evaluation results and paper figures
├── data/                  # downloaded datasets; not tracked by Git
├── checkpoints/           # downloaded model weights; not tracked by Git
└── logs/                  # newly trained model weights
```

All commands below assume that the current working directory is the repository root.

## 1. Environment setup

The environment uses Python 3.11 and PyTorch 2.1.0 with CUDA 11.8. A CUDA-capable GPU is recommended for inference and required for practical training and full benchmark evaluation.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and create the environment:

```bash
git clone https://github.com/ChengWang-hit/CaTFold.git
cd CaTFold
uv sync --frozen
```

Confirm that PyTorch can see the expected devices:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## 2. Download checkpoints and data

The released checkpoints and datasets are provided as two archives. Each archive contains a `CaTFold/` root directory whose paths match this GitHub repository. From the cloned repository root, move to its parent directory, download both archives with resume support, and extract them:

```bash
cd ..

wget -c https://pub-0c2e3470e23a48379f0e2ae3ca84b21b.r2.dev/CaTFold_checkpoints.tar.gz
wget -c https://pub-0c2e3470e23a48379f0e2ae3ca84b21b.r2.dev/CaTFold_data.tar.gz

tar -xzf CaTFold_checkpoints.tar.gz
tar -xzf CaTFold_data.tar.gz

cd CaTFold
```

The extracted `checkpoints/` and `data/` directories merge directly into the clone without renaming or moving anything. The expected layout is:

```text
checkpoints/
├── pretrain_clean_max600.pt
├── pretrain_unfiltered_max600.pt
├── finetune_bprna_new.pt
├── finetune_bprna_1m.pt
├── finetune_pdb.pt
├── finetune_archiveii.pt
└── familyfold_*.pt
data/
├── ArchiveII/
├── PDB/
├── RNAStralign/
├── bpRNA_1m/
├── bpRNA_new/
└── pretrain/
    ├── clean_max600.fa.gz
    └── unfiltered_max600.fa.gz
```

`clean_max600` is the main corpus used for the four conventional benchmarks; it was filtered against downstream data at the family and sequence-similarity levels. `unfiltered_max600` is the additional corpus used for the FamilyFold comparison. Despite its filename, exact duplicates and exact matches to downstream sequences were removed.

## 3. Generate candidate-pair masks

Run this step once after extracting the benchmark datasets and before evaluation or fine-tuning:

```bash
uv run python code/data_processing/generate_masks.py \
  --config configs/data_processing/generate_masks.json
```

This command creates the benchmark `mask_matrix/` directories and updates the corresponding pickle files with `mask_matrix_idx`.

## 4. Reproduce benchmark evaluation

Evaluate the four conventional benchmarks and all nine FamilyFold families:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python code/evaluation/evaluate.py \
  --config configs/evaluation/all_benchmarks.json \
  --device cuda:0
```

Use `--device cpu` if a GPU is unavailable. With `--device auto` (the default), the program uses CUDA when available and otherwise uses the CPU.

Outputs are written to `results/evaluation/`:

- `conventional_benchmarks.csv`: aggregate results for bpRNA-new, PDB, bpRNA-1m, and ArchiveII;
- `familyfold_per_family.csv`: results for each held-out ArchiveII family;
- `all_benchmarks.csv`: the combined aggregate table;
- `per_sequence/all.csv`: one metric row per RNA;
- `per_sequence/all.pickle`: sequences, reference structures, predicted structures, checkpoint names, and per-sequence metrics.

## 5. Predict structures for personal sequences

Write your input sequences to `demo/demo.fasta` in standard FASTA format. The file may contain one or more records; each record begins with an identifier line starting with `>`, followed by one or more sequence lines:

```text
>rna_1
GGGAAACCC
>rna_2
GCGCUUCGCC
```

Run inference with a released fine-tuned checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python code/inference/predict.py \
  --fasta demo/demo.fasta \
  --checkpoint checkpoints/finetune_bprna_new.pt \
  --config configs/inference/predict.json \
  --output demo \
  --device cuda:0
```

The output directory contains one subdirectory per FASTA record. Each record produces:

- `prediction.bpseq`: predicted base pairs in BPSEQ format;
- `logits.npy`: the symmetric matrix of raw, pre-sigmoid logits;
- `secondary_structure.png`: a secondary-structure drawing generated with forgi.

The command overwrites files for an existing sequence identifier and writes a portable run summary to `summary.json`. The supported input alphabet is A, C, G, U, and T; T is normalized to U.

## 6. Pretraining

### Generate the pretraining data

The downloadable pretraining corpora contain sequences only. CaTFold pretraining requires pseudo-secondary structures generated with CONTRAfold. The two input files are placed by the release archive at:

```text
data/pretrain/clean_max600.fa.gz
data/pretrain/unfiltered_max600.fa.gz
```

CONTRAfold is not bundled with this repository. Install it separately, then set the blank `contrafold_path` field in `configs/data_processing/contrafold_predict.json` to its executable path.

Generate one dot-bracket structure per sequence and write the two pretraining HDF5 files with:

```bash
uv run python code/data_processing/contrafold_predict.py \
  --config configs/data_processing/contrafold_predict.json
```

The script reads the compressed FASTA files directly, normalizes T to U, validates the RNA alphabet, displays progress, and writes:

```text
data/pretrain/clean_max600.h5
data/pretrain/unfiltered_max600.h5
```

Each HDF5 file must contain equally sized `seq` and `ss` datasets. `seq` stores RNA sequences and `ss` stores their dot-bracket structures in the same order. Completed batches are recorded in the output, so an interrupted run resumes from the last committed batch.

### Run pretraining

The paper setup uses four GPUs, a per-GPU batch size of 16, BF16 mixed precision, AdamW, weighted binary cross-entropy, and 10 epochs. To pretrain on the clean corpus:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch \
  --num_processes 4 \
  code/pretrain/train.py \
  --config configs/pretrain/clean_max600.json
```

To pretrain on the unfiltered corpus:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch \
  --num_processes 4 \
  code/pretrain/train.py \
  --config configs/pretrain/unfiltered_max600.json
```

## 7. Fine-tuning

Fine-tuning requires a pretrained checkpoint, extracted benchmark datasets, and the candidate masks generated in Section 3.

The four conventional benchmark configurations are intended for four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes 4 \
  code/finetune/train.py --config configs/finetune/bprna_new.json

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes 4 \
  code/finetune/train.py --config configs/finetune/bprna_1m.json

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes 4 \
  code/finetune/train.py --config configs/finetune/pdb.json

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes 4 \
  code/finetune/train.py --config configs/finetune/archiveii.json
```

FamilyFold uses leave-one-family-out training and runs each configuration for 5 epochs with the frozen unfiltered CaTFold backbone and the fusion model configured in `configs/finetune/familyfold/base.json`. The following loop runs all nine configurations sequentially on one GPU:

```bash
for family in 5s 16s 23s rnase_p group_i_intron srp trna telomerase tmrna; do
  CUDA_VISIBLE_DEVICES=0 uv run accelerate launch --num_processes 1 \
    code/finetune/train.py \
    --config "configs/finetune/familyfold/${family}.json"
done
```

## Code Ocean

The published Code Ocean Compute Capsule provides a preconfigured environment for running the released evaluation and inference workflows.

**Code Ocean DOI:** [10.24433/CO.8228673.v1](https://doi.org/10.24433/CO.8228673.v1)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
