# HEALJ

Julia reimplementation of the [HEAL model](https://github.com/ZhonghuiGu/HEAL/tree/main) for protein function prediction on the nrPDB-GO dataset (release 2019-06-18).

---

## 1. Repository layout

- Python (preprocessing) lives under `preprocess/`
- Julia code lives everywhere else

Key paths to get started:

- `preprocess/.venv/` – Python virtual environment (gitignored; instructions below).
- `preprocess/scripts/` – extract GO labels and mine attributed protein graphs.  
- `preprocess/esm_weights/` – ESM-1b weights  (gitignored; instructions below).
- `preprocess/data/raw/` – raw FASTA (sequence data) and GO annotations.
- `preprocess/data/processed/shards/` – attributed graph data stored as HDF5  (gitignored, but I will upload a sample soon)
- `scripts/` – training and testing (heavily reliant on `src/`) requires preprocesed HDF5 shards.

---

## 2. Python preprocessing

All commands assume you are in the repo root `~/HEALJ`.

### 2.1. Create and activate Python virtual environment (venv)
Full requirements list can be found at `preprocess/requirements.txt` and `preprocess/requirements.lock.txt`.

**Windows**

```powershell
cd HEALJ

python -m venv preprocess\.venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\preprocess\.venv\Scripts\Activate.ps1

pip install -U pip
pip install "numpy<2" scipy biopython torch fair-esm h5py
```

**macOS/Linux**
```bash
cd HEALJ

python3 -m venv preprocess/.venv
source preprocess/.venv/bin/activate

pip install -U pip
pip install "numpy<2" scipy biopython torch fair-esm h5py
```
### 2.2. Download ESM-1b weights

**Windows **
```powershell
cd HEALJ
mkdir preprocess\esm_weights

curl -L -o preprocess/esm_weights/esm1b_t33_650M_UR50S.pt `
  https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt

curl -L -o preprocess/esm_weights/esm1b_t33_650M_UR50S-contact-regression.pt `
  https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt
```

**macOS/Linux**
```bash
cd HEALJ
mkdir -p preprocess/esm_weights

curl -L -o preprocess/esm_weights/esm1b_t33_650M_UR50S.pt \
  https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt

curl -L -o preprocess/esm_weights/esm1b_t33_650M_UR50S-contact-regression.pt \
  https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt
```

### 2.3. Expected raw data
The following should be under preprocess/data/raw/ (if not, modify `preprocess/scripts/` accordingly):

nrPDB-GO_2019.06.18_sequences.fasta

nrPDB-GO_2019.06.18_annot.tsv

nrPDB-GO_2019.06.18_train_sequences.fasta

nrPDB-GO_2019.06.18_val_sequences.fasta

nrPDB-GO_2019.06.18_test_sequences.fasta

### 2.4. Build GO vocabulary
**Windows**
```powershell
cd HEALJ


# You may have to include:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\preprocess\.venv\Scripts\Activate.ps1

python preprocess\scripts\build_go_vocab.py `
  --annot_tsv preprocess\data\raw\nrPDB-GO_2019.06.18_annot.tsv `
  --out_json  preprocess\data\processed\go_vocab_train.json
```

**macOS/Linux**
```bash
cd HEALJ

source preprocess/.venv/bin/activate

python preprocess/scripts/build_go_vocab.py \
  --annot_tsv preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv \
  --out_json  preprocess/data/processed/go_vocab_train.json

```
  
### 2.5. Build HDF5 shards, i.e., Execute Preprocessing
This preprocessing can be quit and rerun while skipping all proteins whose attributed graphs has already been mined. However, it is critical that the prefix is changed for every rerun, otherwise the old graphs will be rewritten.

- After execution, HDF5 shards exist the specified `out dir`

**Windows**
```powershell
cd HEALJ

# Training Data
python preprocess\scripts\build_split_shards.py `
  --split_fasta   preprocess/data/raw/nrPDB-GO_2019.06.18_train_sequences.fasta `
  --annot_tsv     preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv `
  --esm_weights   preprocess/esm_weights/esm1b_t33_650M_UR50S.pt `
  --go_vocab_json preprocess/data/processed/go_vocab_train.json `
  --out_dir       preprocess/data/processed/shards/train `
  --prefix        train `
  --shard_size    256 `
  --max_items     30000    # set to zero for full preprocessing of dataset

# Validation Data
python preprocess\scripts\build_split_shards.py `
  --split_fasta   preprocess/data/raw/nrPDB-GO_2019.06.18_val_sequences.fasta `
  --annot_tsv     preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv `
  --esm_weights   preprocess/esm_weights/esm1b_t33_650M_UR50S.pt `
  --go_vocab_json preprocess/data/processed/go_vocab_train.json `
  --out_dir       preprocess/data/processed/shards/validate `
  --prefix        val `
  --shard_size    256 `
  --max_items     3400     # set to zero for full preprocessing of dataset
Test

powershell
Copy code
python preprocess\scripts\build_split_shards.py `
  --split_fasta   preprocess/data/raw/nrPDB-GO_2019.06.18_test_sequences.fasta `
  --annot_tsv     preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv `
  --esm_weights   preprocess/esm_weights/esm1b_t33_650M_UR50S.pt `
  --go_vocab_json preprocess/data/processed/go_vocab_train.json `
  --out_dir       preprocess/data/processed/shards/test `
  --prefix        test `
  --shard_size    256 `
  --max_items     3500     # set to zero for full preprocessing of dataset
```

**macOS/Linux**
```bash
cd HEALJ

# Training data
python preprocess/scripts/build_split_shards.py \
  --split_fasta   preprocess/data/raw/nrPDB-GO_2019.06.18_train_sequences.fasta \
  --annot_tsv     preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv \
  --esm_weights   preprocess/esm_weights/esm1b_t33_650M_UR50S.pt \
  --go_vocab_json preprocess/data/processed/go_vocab_train.json \
  --out_dir       preprocess/data/processed/shards/train \
  --prefix        train \
  --shard_size    256 \
  --max_items     30000    # set to 0 for full preprocessing of dataset

# Validation data
python preprocess/scripts/build_split_shards.py \
  --split_fasta   preprocess/data/raw/nrPDB-GO_2019.06.18_val_sequences.fasta \
  --annot_tsv     preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv \
  --esm_weights   preprocess/esm_weights/esm1b_t33_650M_UR50S.pt \
  --go_vocab_json preprocess/data/processed/go_vocab_train.json \
  --out_dir       preprocess/data/processed/shards/validate \
  --prefix        val \
  --shard_size    256 \
  --max_items     3400     # set to 0 for full preprocessing of dataset

# Test data
python preprocess/scripts/build_split_shards.py \
  --split_fasta   preprocess/data/raw/nrPDB-GO_2019.06.18_test_sequences.fasta \
  --annot_tsv     preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv \
  --esm_weights   preprocess/esm_weights/esm1b_t33_650M_UR50S.pt \
  --go_vocab_json preprocess/data/processed/go_vocab_train.json \
  --out_dir       preprocess/data/processed/shards/test \
  --prefix        test \
  --shard_size    256 \
  --max_items     3500     # set to 0 for full preprocessing of dataset
```

## 3. Julia / HEALJ
### 3.1. Instantiate the environment
**Windows**
```powershell
cd HEALJ

julia --project=. -e "using Pkg; Pkg.instantiate()"
```

**macOS/Linux**
```bash
cd HEALJ

julia --project=. -e 'using Pkg; Pkg.instantiate()'
```


### 3.2. Optional: dataset smoke test
**Windows**
```powershell
cd HEALJ

julia --project=. scripts/avg_graph_stats.jl
```

**macOS/Linux**
```bash
cd HEALJ

julia --project=. scripts/avg_graph_stats.jl

```


### 3.3. Training & Testing
First you must check that the path to your HDF5 shards is specified in `scripts/trainin.jl`. If you are using the smoke shards, you may need to write `preprocess/data/processed/smoke_shards/` wherever you see `preprocess/data/processed/shards/`. This script can be executed via simple command below and will first creat a folder artifacts/run_YYYYMMDD_HHMMSS/ with the following files (storing per-epoch loss data and model parameters, respectively):

`artifacts/run_20251211_134513/history_YYYYMMDD_HHMMSS.csv`
`artifacts/run_20251211_134513/model_YYYYMMDD_HHMMSS.jls`

Additionally, once training is completed, the resulting model is run on test data. With this, a new folder and one file are created containing rudimentary per-protein results (graph_id, graph_ref, logits, true labels).

`artifacts/test_results_YYYYMMDD_HHMMSS/raw_results_YYYYMMDD_HHMMSS.jls`


**Windows**
```powershell
cd HEALJ

julia --project=. analysis/train_and_test.jl
```

**macOS/Linux**
```bash
cd HEALJ

julia --project=. analysis/train_and_test.jl
```

### 3.4. Analysis


Convert test results to a CSV while doing a few preliminary computations.

**Windows**
```powershell
cd HEALJ

julia --project=. analysis/analyze_test_results.jl `
  artifacts\test_results_YYYYMMDD_HHMMSS\raw_results_YYYYMMDD_HHMMSS.jls

```

**macOS/Linux**
```bash
cd HEALJ

julia --project=. analysis/analyze_test_results.jl \
  artifacts/test_results_YYYYMMDD_HHMMSS/raw_results_YYYYMMDD_HHMMSS.jls
```
  
Compute final metrics (Appendix A of manuscript):

**Windows**
```powershell
cd HEALJ

julia --project=. analysis/final_metrics.jl `
  artifacts\test_results_YYYYMMDD_HHMMSS\raw_results_YYYYMMDD_HHMMSS_analysis.csv

```

**macOS/Linux**
```bash
cd HEALJ

julia --project=. analysis/final_metrics.jl \
  artifacts/test_results_YYYYMMDD_HHMMSS/raw_results_YYYYMMDD_HHMMSS_analysis.csv
```
