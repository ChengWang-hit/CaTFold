#!/usr/bin/env python3
"""Generate CONTRAfold structures for pretraining FASTA files."""

import argparse
import concurrent.futures
import gzip
import json
import re
import subprocess
import tempfile
from functools import partial
from itertools import islice
from pathlib import Path

import h5py
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRUCTURE_PATTERN = re.compile(r"^[().\[\]{}<>]+$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def open_fasta(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_fasta(path):
    header = None
    sequence_lines = []
    record_index = 0
    with open_fasta(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    if not sequence_lines:
                        raise ValueError(f"Empty FASTA record in {path}: {header}")
                    record_index += 1
                    yield validate_sequence(
                        "".join(sequence_lines), path, record_index
                    )
                header = line[1:].strip()
                sequence_lines = []
            else:
                if header is None:
                    raise ValueError(f"Sequence found before the first header: {path}")
                sequence_lines.append(line)
        if header is not None:
            if not sequence_lines:
                raise ValueError(f"Empty FASTA record in {path}: {header}")
            record_index += 1
            yield validate_sequence("".join(sequence_lines), path, record_index)


def validate_sequence(sequence, path, record_index):
    sequence = sequence.upper().replace("T", "U")
    invalid = sorted(set(sequence) - set("ACGU"))
    if invalid:
        raise ValueError(
            f"Invalid bases in {path}, record {record_index}: {''.join(invalid)}"
        )
    return sequence


def parse_structure(stdout, sequence_length):
    for line in reversed(stdout.splitlines()):
        fields = line.strip().split()
        if fields and STRUCTURE_PATTERN.fullmatch(fields[0]):
            structure = fields[0]
            if len(structure) != sequence_length:
                raise ValueError(
                    "CONTRAfold returned a structure with a different length"
                )
            return structure
    raise ValueError("Could not find a dot-bracket structure in CONTRAfold output")


def predict_sequence(sequence, contrafold, timeout, temporary_dir):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".fa",
            dir=temporary_dir,
            delete=False,
        ) as handle:
            handle.write(f">sequence\n{sequence}\n")
            temporary_path = Path(handle.name)
        result = subprocess.run(
            [str(contrafold), "predict", str(temporary_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"CONTRAfold exited with code {result.returncode}: {message[-500:]}"
            )
        return parse_structure(result.stdout, len(sequence))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_output(path, source_fasta, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()
    handle = h5py.File(path, "a")
    string_type = h5py.string_dtype(encoding="utf-8")
    present = [name in handle for name in ("seq", "ss")]
    if any(present) and not all(present):
        handle.close()
        raise ValueError(f"Incomplete HDF5 datasets in {path}")
    if not any(present):
        for name in ("seq", "ss"):
            handle.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=string_type,
                chunks=True,
                compression="gzip",
            )
    saved_records = min(len(handle["seq"]), len(handle["ss"]))
    committed = int(handle.attrs.get("committed_records", saved_records))
    if committed > saved_records:
        handle.close()
        raise ValueError(f"Invalid resume position in {path}")
    handle["seq"].resize((committed,))
    handle["ss"].resize((committed,))
    handle.attrs["committed_records"] = committed
    handle.attrs["source_fasta"] = source_fasta
    handle.flush()
    return handle, committed


def append_batch(handle, sequences, structures):
    start = int(handle.attrs["committed_records"])
    stop = start + len(sequences)
    handle["seq"].resize((stop,))
    handle["ss"].resize((stop,))
    handle["seq"][start:stop] = sequences
    handle["ss"][start:stop] = structures
    handle.flush()
    handle.attrs["committed_records"] = stop
    handle.flush()


def process_dataset(specification, config, contrafold, overwrite):
    fasta_path = PROJECT_ROOT / specification["fasta"]
    output_path = PROJECT_ROOT / specification["output"]
    expected_records = specification.get("expected_records")
    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    handle, completed = prepare_output(
        output_path, specification["fasta"], overwrite
    )
    with handle:
        if expected_records is not None and completed == expected_records:
            handle.attrs["complete"] = True
            print(f"Already complete: {output_path} ({completed:,} sequences)")
            return
        if expected_records is not None and completed > expected_records:
            raise ValueError(f"Output contains too many records: {output_path}")

        records = read_fasta(fasta_path)
        for _ in range(completed):
            if next(records, None) is None:
                raise ValueError("FASTA ended before the saved resume position")

        predict = partial(
            predict_sequence,
            contrafold=contrafold,
            timeout=int(config["timeout"]),
        )
        with tqdm(
            total=expected_records,
            initial=completed,
            desc=specification["name"],
            unit="sequence",
        ) as progress:
            with tempfile.TemporaryDirectory(
                prefix="catfold_contrafold_"
            ) as temporary_dir:
                predict_with_temp = partial(predict, temporary_dir=temporary_dir)
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=int(config["workers"])
                ) as executor:
                    while True:
                        sequences = list(
                            islice(records, int(config["batch_size"]))
                        )
                        if not sequences:
                            break
                        structures = []
                        for structure in executor.map(
                            predict_with_temp, sequences
                        ):
                            structures.append(structure)
                            progress.update(1)
                        append_batch(handle, sequences, structures)

        completed = int(handle.attrs["committed_records"])
        if expected_records is not None and completed != expected_records:
            raise ValueError(
                f"Expected {expected_records:,} records, but generated {completed:,}"
            )
        handle.attrs["complete"] = True
        handle.flush()
        print(f"Saved: {output_path} ({completed:,} sequences)")


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("workers", "batch_size", "timeout"):
        if int(config[name]) <= 0:
            raise ValueError(f'Configuration value "{name}" must be positive')
    contrafold_value = config.get("contrafold_path", "").strip()
    if not contrafold_value:
        raise ValueError('Set "contrafold_path" in the configuration file')
    contrafold = Path(contrafold_value).expanduser()
    if not contrafold.is_file():
        raise FileNotFoundError(f"CONTRAfold executable not found: {contrafold}")
    for specification in config["datasets"]:
        process_dataset(specification, config, contrafold, args.overwrite)


if __name__ == "__main__":
    main()
