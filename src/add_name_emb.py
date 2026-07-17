"""Incrementally add SBERT name embeddings to features.pt.

Name is a content signal (portable: Google returns names too) and gets its own
branch in the encoder rather than being merged with tags -- concatenation lets
the fusion MLP weight name independently, which averaging into another branch
could not. Cheap to add: names are short, so this runs locally in seconds and
does not touch the expensive review encode.
"""

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

OUT = Path("data/processed")
MODEL = "thenlper/gte-base"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    d = torch.load(OUT / "features.pt", weights_only=False)
    names = d["names"]
    print(f"encoding {len(names):,} names on {device}")

    sbert = SentenceTransformer(MODEL, device=device)
    name_emb = sbert.encode(names, batch_size=128, show_progress_bar=True, convert_to_numpy=True)

    d["name_emb"] = torch.from_numpy(name_emb).float()
    torch.save(d, OUT / "features.pt")
    print(f"added name_emb {tuple(d['name_emb'].shape)} -> {OUT}/features.pt")


if __name__ == "__main__":
    main()
