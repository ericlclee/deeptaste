"""Content-based pretraining for RestaurantEncoder -- makes the embedding space
reflect restaurant similarity (shared cuisine tag, price tier, geo proximity)
BEFORE any user/collaborative signal touches the weights.

Why: the encoder currently only ever gets gradient from BPR over user
interactions. That signal is sparse (thousands of users, a few reviews each),
so a restaurant with few reviews barely shapes the space at all -- and there's
nothing enforcing that two restaurants that plainly look alike (same cuisine,
price, neighborhood) end up close together in embedding space; it's hoped for
as an emergent side effect, not guaranteed. Content similarity, by contrast,
is defined for EVERY restaurant regardless of review count, so this stage can
place a restaurant sensibly even with zero interaction history -- which is
what lets a brand-new restaurant (or a 1-review user's single data point) be
embedded meaningfully at serve time.

Training signal: same "same cuisine/price/geo cluster" candidates
build_hard_candidates() already computes for hard-negative sampling in
train.py, but used here as the ground truth for what "similar" means, not as
negatives -- there's no other source of restaurant-similarity labels
available. A BPR-style triplet loss pulls the anchor closer to a candidate
from that pool than to a uniformly random restaurant.

Trains the CONTENT path only (encoder.content_parameters()) -- the
per-restaurant id embedding and item bias are deliberately left at their
initialization. Those exist to memorize what content cannot explain, so
training them on a content-similarity objective would be self-defeating: it
would teach the memorization table to reproduce the content signal, and
"restaurants that look alike get identical vectors" is exactly the collapse the
id embedding is there to break.

Usage:
    python src/pretrain_encoder.py
    python src/train.py --init-checkpoint data/yelp_philadelphia/pretrained_encoder.pt ...
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import RestaurantEncoder
from train import build_hard_candidates

OUT = Path(os.environ.get("DEEP_TASTE_DATA", "data/yelp_philadelphia"))


class SimilarityDataset(Dataset):
    """One example = (anchor, similar, random). `similar` comes from
    build_hard_candidates(anchor); `random` is any other restaurant not in
    that pool (so the loss can't be satisfied by e.g. just embedding every
    restaurant identically)."""

    def __init__(self, hard_candidates: list[np.ndarray], n_restaurants: int):
        self.hard_candidates = hard_candidates
        self.n_restaurants = n_restaurants
        # restaurants with an empty candidate pool (shouldn't happen given
        # build_hard_candidates' geo-only fallback, but be defensive) can't
        # form a training example
        self.anchors = [i for i in range(n_restaurants) if len(hard_candidates[i]) > 0]

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        pool = self.hard_candidates[anchor]
        similar = int(np.random.choice(pool))

        exclude = set(pool.tolist())
        exclude.add(anchor)
        neg = np.random.randint(self.n_restaurants)
        while neg in exclude:
            neg = np.random.randint(self.n_restaurants)

        return anchor, similar, neg


def similarity_loss(anchor_emb, sim_emb, neg_emb):
    """Same BPR form as train.py's bpr_loss, applied to restaurant-restaurant
    similarity instead of user-restaurant preference."""
    pos_score = (anchor_emb * sim_emb).sum(dim=1)
    neg_score = (anchor_emb * neg_emb).sum(dim=1)
    return -F.logsigmoid(pos_score - neg_score).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--output-dims", type=int, default=128)
    p.add_argument("--hard-neg-k", type=int, default=30)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="where to write the pretrained encoder (default: <data>/pretrained_encoder.pt)",
    )
    args = p.parse_args()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else OUT / "pretrained_encoder.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")

    feats = torch.load(OUT / "features.pt", weights_only=False)
    n_restaurants = len(feats["business_ids"])
    print(f"{n_restaurants:,} restaurants")

    encoder = RestaurantEncoder(feats, output_dims=args.output_dims).to(device)

    print(f"building hard-candidate pools (k={args.hard_neg_k})...")
    hard_candidates = build_hard_candidates(feats, k=args.hard_neg_k)

    ds = SimilarityDataset(hard_candidates, n_restaurants)
    print(f"{len(ds):,} restaurants with a similarity pool "
          f"({n_restaurants - len(ds):,} skipped, no candidates)")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    content_params = list(encoder.content_parameters())
    opt = torch.optim.Adam(content_params, lr=args.lr)

    for epoch in range(args.epochs):
        encoder.train()
        total = 0.0
        for anchor, sim, neg in loader:
            anchor, sim, neg = anchor.to(device), sim.to(device), neg.to(device)

            # content_embedding, not forward: the id embedding must stay out of
            # this objective (see module docstring). Normalize so the loss sees
            # the same cosine geometry training will.
            anchor_emb = F.normalize(encoder.content_embedding(anchor), dim=1)
            sim_emb = F.normalize(encoder.content_embedding(sim), dim=1)
            neg_emb = F.normalize(encoder.content_embedding(neg), dim=1)

            loss = similarity_loss(anchor_emb, sim_emb, neg_emb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(content_params, max_norm=args.clip)
            opt.step()
            total += loss.item() * len(anchor)

        print(f"epoch {epoch + 1}/{args.epochs}  sim_loss {total / len(ds):.4f}")

    torch.save(encoder.state_dict(), ckpt_path)
    print(f"\nsaved pretrained encoder to {ckpt_path}")


if __name__ == "__main__":
    main()
