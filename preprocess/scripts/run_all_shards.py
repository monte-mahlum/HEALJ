import subprocess
import sys
from pathlib import Path

"""
Wrapper around build_split_shards.py for train, val, and test, so you you only run once, 
assuming it doesn't stop midway like it did for me. If that happens, just re-run this script
WITH DIFFERENT --prefix ARGUMENTS in build_split_shards.py calls to avoid overwriting existing shards.

To run from repo root with .venv activated:
python preprocess/scripts/run_all_shards.py
"""

def run(cmd):
    print("\n=== Running ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

PY = sys.executable  # current Python (your .venv python)


# Common arguments
COMMON = [
    "preprocess/scripts/build_split_shards.py",
    "--annot_tsv", "preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv",
    "--esm_weights", "preprocess/esm_weights/esm1b_t33_650M_UR50S.pt",
    "--go_vocab_json", "preprocess/data/processed/go_vocab_train.json",
    "--shard_size", "256",
]

def main():
    """
    # Train
    run([
        PY, *COMMON,
        "--split_fasta", "preprocess/data/raw/nrPDB-GO_2019.06.18_train_sequences.fasta",
        "--out_dir", "preprocess/data/processed/all_shards/train",
        "--prefix", "train2",
        "--max_items", "30000",
    ])
"""
    # Val
    run([
        PY, *COMMON,
        "--split_fasta", "preprocess/data/raw/nrPDB-GO_2019.06.18_val_sequences.fasta",
        "--out_dir", "preprocess/data/processed/all_shards/validate",
        "--prefix", "val",
        "--max_items", "2000",
    ])

    # Test
    run([
        PY, *COMMON,
        "--split_fasta", "preprocess/data/raw/nrPDB-GO_2019.06.18_test_sequences.fasta",
        "--out_dir", "preprocess/data/processed/all_shards/test",
        "--prefix", "test",
        "--max_items", "2000",
    ])

if __name__ == "__main__":
    main()
