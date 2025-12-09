import argparse, re, json
from pathlib import Path

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_fasta", required=True)
    ap.add_argument("--annot_tsv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    train_ids = read_fasta_ids(Path(args.train_fasta))
    ann_map = load_annotations(Path(args.annot_tsv))

    # Build vocab ONLY from training ids
    go_terms = []
    seen = set()
    for pid in train_ids:
        for go in ann_map.get(pid, []):
            if go not in seen:
                seen.add(go)
                go_terms.append(go)

    out = {
        "go_terms": go_terms,
        "go_to_idx": {go:i for i,go in enumerate(go_terms)}
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out))
    print(f"Wrote {args.out_json} with {len(go_terms)} GO terms.")

if __name__ == "__main__":
    main()

