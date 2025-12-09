
Python Preprocessing (ESM + graph shard builder)

Repo tree (Python lives under `preprocess/`)
- Python venv: `preprocess/.venv/`
- Python scripts: `preprocess/scripts/`
- ESM weights: `preprocess/esm_weights/`
- Raw inputs: `preprocess/data/raw/`
- Cached PDBs: `preprocess/data/structures/pdb/`
- Outputs (HDF5 shards): `preprocess/data/processed/shards/{train,validate,test}/`

---

1) Create + activate the Python venv (inside preprocess)
```bash
cd /Users/montemahlum/HEALJ

python3 -m venv preprocess/.venv
source preprocess/.venv/bin/activate
pip install -U pip
pip install "numpy<2" scipy biopython torch fair-esm h5py
2) Download ESM-1b weights (local only)
bash
Copy code
cd /Users/montemahlum/HEALJ
mkdir -p preprocess/esm_weights

curl -L -o preprocess/esm_weights/esm1b_t33_650M_UR50S.pt \
  https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt

curl -L -o preprocess/esm_weights/esm1b_t33_650M_UR50S-contact-regression.pt \
  https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt

ls -lh preprocess/esm_weights
3) Data files (what’s what)
PDBch dataset (PDB-based)
preprocess/data/raw/nrPDB-GO_2019.06.18_sequences.fasta — full FASTA snapshot

preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv — protein → GO term labels

preprocess/data/raw/nrPDB-GO_2019.06.18_train_sequences.fasta — train split

preprocess/data/raw/nrPDB-GO_2019.06.18_val_sequences.fasta — val split

preprocess/data/raw/nrPDB-GO_2019.06.18_test_sequences.fasta — test split

preprocess/data/raw/nrPDB-GO_2019.06.18_test.csv — test metadata (IDs/labels in CSV form)

AFch dataset (AlphaFold / AF-model-based)
preprocess/data/raw/nrAF-Model-GO_train_sequences.fasta, ..._val_..., ..._test_... — AF-model train/val/test splits

preprocess/data/raw/nrAF_PDB_train_sequences.fasta — AF/PDB-linked training subset

Swiss-Model dataset
preprocess/data/raw/nrSwiss-Model-GO_sequences.fasta — Swiss-Model sequences

preprocess/data/raw/nrSwiss-Model-GO_annot.tsv — Swiss-Model protein → GO labels

4) Build go_vocab_train.json (if you need to recreate it)
bash
Copy code
cd /Users/montemahlum/HEALJ
source preprocess/.venv/bin/activate

python preprocess/scripts/build_go_vocab.py \
  --annot_tsv preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv \
  --out_json preprocess/data/processed/go_vocab_train.json
5) Generate sample HDF5 shards (train/validate/test)
These commands:

read .fasta split IDs

map IDs → GO labels using the TSV

download/cache PDB structures under preprocess/data/structures/pdb/

compute ESM-1b per-residue embeddings

write .h5 shard files into preprocess/data/processed/shards/...

Train shards (sample)
bash
Copy code
cd /Users/montemahlum/HEALJ
source preprocess/.venv/bin/activate

python preprocess/scripts/build_split_shards.py \
  --split_fasta preprocess/data/raw/nrPDB-GO_2019.06.18_train_sequences.fasta \
  --annot_tsv preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv \
  --esm_weights preprocess/esm_weights/esm1b_t33_650M_UR50S.pt \
  --go_vocab_json preprocess/data/processed/go_vocab_train.json \
  --out_dir preprocess/data/processed/shards/train \
  --prefix train \
  --shard_size 256 \
  --max_items 1000
Validate shards (sample)
bash
Copy code
cd /Users/montemahlum/HEALJ
source preprocess/.venv/bin/activate

python preprocess/scripts/build_split_shards.py \
  --split_fasta preprocess/data/raw/nrPDB-GO_2019.06.18_val_sequences.fasta \
  --annot_tsv preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv \
  --esm_weights preprocess/esm_weights/esm1b_t33_650M_UR50S.pt \
  --go_vocab_json preprocess/data/processed/go_vocab_train.json \
  --out_dir preprocess/data/processed/shards/validate \
  --prefix val \
  --shard_size 256 \
  --max_items 250
Test shards (sample)
bash
Copy code
cd /Users/montemahlum/HEALJ
source preprocess/.venv/bin/activate

python preprocess/scripts/build_split_shards.py \
  --split_fasta preprocess/data/raw/nrPDB-GO_2019.06.18_test_sequences.fasta \
  --annot_tsv preprocess/data/raw/nrPDB-GO_2019.06.18_annot.tsv \
  --esm_weights preprocess/esm_weights/esm1b_t33_650M_UR50S.pt \
  --go_vocab_json preprocess/data/processed/go_vocab_train.json \
  --out_dir preprocess/data/processed/shards/test \
  --prefix test \
  --shard_size 256 \
  --max_items 250