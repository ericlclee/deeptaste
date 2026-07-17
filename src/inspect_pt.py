"""Pretty-print the contents of a .pt file (dict of tensors / lists).

    python src/inspect_pt.py data/processed/features.pt
    python src/inspect_pt.py data/processed/features.pt --key text_emb
    python src/inspect_pt.py data/processed/features.pt --row 0
"""

import argparse

import torch


def summarize_tensor(v: torch.Tensor) -> str:
    s = f"{str(tuple(v.shape)):16} {str(v.dtype):14}"
    if v.dtype.is_floating_point and v.numel():
        s += f" min={v.min():.3g} max={v.max():.3g} mean={v.float().mean():.3g}"
        if torch.isnan(v).any():
            s += "  !! NaN"
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--key", help="dump one key in full")
    p.add_argument("--row", type=int, help="show every field for one restaurant index")
    args = p.parse_args()

    d = torch.load(args.path, weights_only=False)

    if not isinstance(d, dict):
        print(summarize_tensor(d) if torch.is_tensor(d) else repr(d))
        return

    if args.key:
        v = d[args.key]
        print(v if not torch.is_tensor(v) else v)
        return

    if args.row is not None:
        i = args.row
        name = d.get("names", ["?"] * (i + 1))[i]
        print(f"row {i}: {name}\n")
        for k, v in d.items():
            if torch.is_tensor(v) and v.shape and v.shape[0] == len(d.get("names", [])):
                print(f"  {k:12} {v[i].tolist() if v[i].numel() <= 12 else str(tuple(v[i].shape)) + ' vector'}")
        return

    print(f"{args.path}\n")
    for k, v in d.items():
        if torch.is_tensor(v):
            print(f"  {k:14} {summarize_tensor(v)}")
        elif isinstance(v, list):
            preview = ", ".join(map(str, v[:3]))
            print(f"  {k:14} list[{len(v)}]      e.g. {preview} ...")
        else:
            print(f"  {k:14} {type(v).__name__}: {v}")


if __name__ == "__main__":
    main()
