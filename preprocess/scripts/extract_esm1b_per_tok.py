import argparse
from pathlib import Path
import torch
import torch.serialization as serialization
import esm

"""
Helper for build_split_shards.py to extract per-token ESM-1b embeddings for a single sequence.
"""

def read_fasta_one(path: Path):
    lines = path.read_text().splitlines()
    header = None
    seq = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is None:
                header = line[1:].strip()
            else:
                # stop at first record (keep it simple)
                break
        else:
            seq.append(line)
    if header is None:
        raise ValueError("No FASTA header found")
    return header, "".join(seq)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to esm1b_t33_650M_UR50S.pt")
    ap.add_argument("--fasta", required=True, help="FASTA file (single sequence)")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--layer", type=int, default=33, help="Representation layer (33 for ESM-1b)")
    args = ap.parse_args()

    model_path = args.model
    fasta_path = Path(args.fasta)
    out_path = Path(args.out)

    name, seq = read_fasta_one(fasta_path)

    # Load model from local checkpoint file with safe_globals
    with serialization.safe_globals([argparse.Namespace]):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_converter = alphabet.get_batch_converter()
    data = [(name, seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[args.layer], return_contacts=False)
        rep = out["representations"][args.layer]   # (1, L+2, hidden)

    # drop BOS/EOS -> (L, hidden)
    per_tok = rep[0, 1:len(seq)+1, :].cpu().contiguous().numpy().astype(np.float32)

    payload = {
        "name": name,
        "seq": seq,
        "layer": args.layer,
        "per_tok": per_tok,          # torch.FloatTensor (L, hidden)
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"saved {out_path}  per_tok shape={tuple(per_tok.shape)}")

if __name__ == "__main__":
    main()


