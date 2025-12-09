import argparse, re, json, math
from pathlib import Path
import numpy as np
import h5py
import torch
import esm
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from scipy.spatial import cKDTree
import urllib.request

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a:i for i,a in enumerate(AA20)}

def read_fasta_ids(fasta_path: Path):
    ids = []
    for line in fasta_path.read_text().splitlines():
        if line.startswith(">"):
            ids.append(line[1:].strip().split()[0])
    return ids

def load_annotations(annot_tsv: Path):
    m = {}
    for line in annot_tsv.read_text().splitlines():
        if not line.strip():
            continue
        pid = line.split("\t")[0].strip()
        gos = re.findall(r"GO:\d{7}", line)
        if gos:
            m.setdefault(pid, set()).update(gos)
    return m

def download_pdb_if_needed(pdb_id: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdb_id}.pdb"
    if out_path.exists():
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    urllib.request.urlretrieve(url, out_path)
    return out_path

def parse_pdb_chain(pdb_path: Path, chain_id: str):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", str(pdb_path))
    model = next(struct.get_models())
    chain = model[chain_id]

    coords, aa_idx, seq_chars = [], [], []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        if "CA" not in res:
            continue
        aa = seq1(res.get_resname(), custom_map={})
        if aa not in AA_TO_IDX:
            continue
        coords.append(res["CA"].get_coord())
        aa_idx.append(AA_TO_IDX[aa])
        seq_chars.append(aa)

    coords = np.asarray(coords, dtype=np.float32)
    aa_idx = np.asarray(aa_idx, dtype=np.int16)
    seq = "".join(seq_chars)
    return coords, aa_idx, seq

def build_edge_index(coords: np.ndarray, cutoff=10.0):
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cutoff, output_type="set")
    src, dst = [], []
    for i, j in pairs:
        src += [i, j]
        dst += [j, i]
    return np.vstack([src, dst]).astype(np.int32)

def one_hot_native(aa_idx: np.ndarray):
    x = np.zeros((aa_idx.shape[0], 20), dtype=np.float32)
    x[np.arange(aa_idx.shape[0]), aa_idx.astype(np.int64)] = 1.0
    return x

def esm_per_tok(model, alphabet, seq: str, layer: int = 33):
    batch_converter = alphabet.get_batch_converter()
    data = [("p", seq)]
    _, _, tokens = batch_converter(data)
    with torch.no_grad():
        out = model(tokens, repr_layers=[layer], return_contacts=False)
        rep = out["representations"][layer]
    per_tok = rep[0, 1:len(seq)+1, :].cpu().contiguous()
    return per_tok.numpy().astype(np.float32)

def parse_pdbch_id(pid: str):
    m = re.match(r"^([0-9A-Za-z]{4})[-_]?([A-Za-z0-9])$", pid)
    if not m:
        raise ValueError(f"Cannot parse PDBch id '{pid}'. Expected like '154L-A'.")
    return m.group(1).lower(), m.group(2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_fasta", required=True)
    ap.add_argument("--annot_tsv", required=True)
    ap.add_argument("--esm_weights", required=True)
    ap.add_argument("--go_vocab_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True)  # train / val / test
    ap.add_argument("--pdb_dir", default="data/structures/pdb")
    ap.add_argument("--shard_size", type=int, default=256)
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()

    split_ids = read_fasta_ids(Path(args.split_fasta))
    ann_map = load_annotations(Path(args.annot_tsv))

    vocab_obj = json.loads(Path(args.go_vocab_json).read_text())
    go_vocab = vocab_obj["go_to_idx"]
    go_terms = vocab_obj["go_terms"]
    num_labels = len(go_terms)

    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.esm_weights)
    esm_model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)

    total = len(split_ids) if args.max_items == 0 else min(len(split_ids), args.max_items)
    num_shards = math.ceil(total / args.shard_size)

    global_written = 0
    for s in range(num_shards):
        a = s * args.shard_size
        b = min(total, (s + 1) * args.shard_size)
        out_path = out_dir / f"{args.prefix}_{s:04d}.h5"

        with h5py.File(out_path, "w") as f:
            f.attrs["num_labels"] = num_labels
            f.create_dataset("go_terms", data=np.array(go_terms, dtype="S10"))

            n = 0
            for pid in split_ids[a:b]:
                gos = ann_map.get(pid, set())
                y = np.zeros((num_labels,), dtype=np.float32)
                for go in gos:
                    idx = go_vocab.get(go, None)
                    if idx is not None:
                        y[idx] = 1.0

                try:
                    pdb_id, chain_id = parse_pdbch_id(pid)
                    pdb_path = download_pdb_if_needed(pdb_id, pdb_dir)
                    coords, aa_idx, seq = parse_pdb_chain(pdb_path, chain_id)
                    if coords.shape[0] == 0:
                        continue

                    edge_index = build_edge_index(coords, cutoff=10.0)
                    native_x = one_hot_native(aa_idx)
                    x = esm_per_tok(esm_model, alphabet, seq, layer=33)
                    if x.shape[0] != coords.shape[0]:
                        continue

                    grp = f.create_group(str(n))
                    grp.attrs["id"] = pid
                    grp.create_dataset("edge_index", data=edge_index, compression="gzip")
                    grp.create_dataset("native_x", data=native_x, compression="gzip")
                    grp.create_dataset("x", data=x, compression="gzip")
                    grp.create_dataset("y", data=y, compression="gzip")
                    n += 1
                except Exception:
                    continue

            f.attrs["num_graphs"] = n
        global_written += n
        print(f"Wrote {out_path} with {n} graphs")

    print(f"Done. Total graphs written across shards: {global_written}  labels={num_labels}")

if __name__ == "__main__":
    main()

