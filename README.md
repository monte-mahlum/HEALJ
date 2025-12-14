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
- `preprocess/data/processed/shards/` – HDF5 graph shards  
- `scripts/` – training and testing (heavily reliant on src/)

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
  
### 2.5. Build HDF5 shards
Train

powershell
Copy code
python preprocess\scripts\build_split_shards.py `
  --split_fasta   preprocess/data/raw/nrPDB-GO_2019.06.18_train_sequences.fasta `
  --annot_tsv     preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv `
  --esm_weights   preprocess/esm_weights/esm1b_t33_650M_UR50S.pt `
  --go_vocab_json preprocess/data/processed/go_vocab_train.json `
  --out_dir       preprocess/data/processed/all_shards/train `
  --prefix        train `
  --shard_size    256 `
  --max_items     30000
Validation

powershell
Copy code
python preprocess\scripts\build_split_shards.py `
  --split_fasta   preprocess/data/raw/nrPDB-GO_2019.06.18_val_sequences.fasta `
  --annot_tsv     preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv `
  --esm_weights   preprocess/esm_weights/esm1b_t33_650M_UR50S.pt `
  --go_vocab_json preprocess/data/processed/go_vocab_train.json `
  --out_dir       preprocess/data/processed/all_shards/validate `
  --prefix        val `
  --shard_size    256 `
  --max_items     3400
Test

powershell
Copy code
python preprocess\scripts\build_split_shards.py `
  --split_fasta   preprocess/data/raw/nrPDB-GO_2019.06.18_test_sequences.fasta `
  --annot_tsv     preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv `
  --esm_weights   preprocess/esm_weights/esm1b_t33_650M_UR50S.pt `
  --go_vocab_json preprocess/data/processed/go_vocab_train.json `
  --out_dir       preprocess/data/processed/all_shards/test `
  --prefix        test `
  --shard_size    256 `
  --max_items     3500
At this point, HDF5 shards exist under:

preprocess/data/processed/all_shards/train/

preprocess/data/processed/all_shards/validate/

preprocess/data/processed/all_shards/test/

3. Julia / HEALJ
3.1. Instantiate the environment
bash
Copy code
cd HEALJ
julia --project=. -e 'using Pkg; Pkg.instantiate()'
3.2. Optional: dataset smoke test
bash
Copy code
julia --project=. scripts/avg_graph_stats.jl
(Or any other smoke script that calls HEALJ.DataLoader.make_datasets().)

3.3. Training
Example:

bash
Copy code
julia --project=. scripts/train.jl
Outputs go under:

artifacts/run_YYYYMMDD_HHMMSS/

e.g.:

artifacts/run_20251211_134513/history_20251211_134513.csv

artifacts/run_20251211_134513/model_20251211_134513.jls

3.4. Testing and analysis
Run test:

bash
Copy code
julia --project=. analysis/test.jl
This writes raw test results to:

artifacts/test_results_YYYYMMDD_HHMMSS/raw_results_YYYYMMDD_HHMMSS.jls

Convert to per-protein metrics:

bash
Copy code
julia --project=. analysis/analyze_test_results.jl \
  artifacts/test_results_YYYYMMDD_HHMMSS/raw_results_YYYYMMDD_HHMMSS.jls
Compute final aggregate metrics:

bash
Copy code
julia --project=. analysis/final_metrics.jl \
  artifacts/test_results_YYYYMMDD_HHMMSS/raw_results_YYYYMMDD_HHMMSS_analysis.csv
