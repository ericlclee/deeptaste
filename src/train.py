"""BPR training loop for the two-tower recommender.

Plumbing (Dataset, user-history precompute, negative sampling, the loop) is done.
You fill in the two `Q:` functions -- aggregate_user_emb and bpr_loss -- which are
the heart of the method.

The user tower has no parameters of its own: user_emb is a signed-weighted pool of
the SAME encoder's outputs on the restaurants that user rated. So gradients flow
from the BPR loss, through the positive/negative embeddings AND through the user's
history embeddings, all into the one shared RestaurantEncoder. That is what makes
the two-stage design "effectively end-to-end".
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model import RestaurantEncoder

OUT = Path("data/processed")
POSITIVE_THRESHOLD = 4  # a rating >= this is a "positive" the user is said to prefer


# ---------------------------------------------------------------------------
# Precompute: map ids to indices and build padded per-user rating histories.
# ---------------------------------------------------------------------------
def build_training_data(max_history: int):
    feats = torch.load(OUT / "features.pt", weights_only=False)
    biz_to_idx = {b: i for i, b in enumerate(feats["business_ids"])}
    n_restaurants = len(feats["business_ids"])

    reviews = pd.read_parquet(OUT / "reviews_split.parquet")
    train = reviews[reviews.split == "train"].copy()
    train["r_idx"] = train.business_id.map(biz_to_idx)
    train = train.sort_values(["user_id", "date"])

    users = sorted(train.user_id.unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    n_users = len(users)

    # padded history tensors, one row per user (most-recent `max_history` kept)
    hist_items = np.zeros((n_users, max_history), dtype=np.int64)
    hist_ratings = np.zeros((n_users, max_history), dtype=np.float32)
    hist_mask = np.zeros((n_users, max_history), dtype=bool)

    # (user_idx, positive_r_idx) pairs = the BPR training examples
    positives = []

    for uid, grp in train.groupby("user_id"):
        u = user_to_idx[uid]
        items = grp.r_idx.to_numpy()[-max_history:]
        rats = grp.stars.to_numpy(dtype=np.float32)[-max_history:]
        k = len(items)
        hist_items[u, :k] = items
        hist_ratings[u, :k] = rats
        hist_mask[u, :k] = True
        for r_idx, stars in zip(grp.r_idx.to_numpy(), grp.stars.to_numpy()):
            if stars >= POSITIVE_THRESHOLD:
                positives.append((u, int(r_idx)))

    global_mean = float(train.stars.mean())
    print(f"{n_users:,} users | {n_restaurants:,} restaurants | {len(positives):,} positive interactions")

    return {
        "n_users": n_users,
        "n_restaurants": n_restaurants,
        "hist_items": torch.from_numpy(hist_items),
        "hist_ratings": torch.from_numpy(hist_ratings),
        "hist_mask": torch.from_numpy(hist_mask),
        "positives": positives,
        "global_mean": global_mean,
        "features": feats,
    }


class BPRDataset(Dataset):
    """One example = (user_idx, positive_idx, negative_idx). Negatives are sampled
    uniformly from restaurants the user has not interacted with (false-negative
    rate ~0.1% at this catalog size, per the earlier analysis)."""

    def __init__(self, data: dict):
        self.positives = data["positives"]
        self.n_restaurants = data["n_restaurants"]
        # set of interacted items per user, for negative rejection
        self.seen = [set() for _ in range(data["n_users"])]
        items, mask = data["hist_items"], data["hist_mask"]
        for u in range(data["n_users"]):
            self.seen[u] = set(items[u][mask[u]].tolist())

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, i):
        u, pos = self.positives[i]
        j = np.random.randint(self.n_restaurants)
        while j in self.seen[u]:
            j = np.random.randint(self.n_restaurants)
        return u, pos, j


# ---------------------------------------------------------------------------
# YOUR TWO FUNCTIONS
# ---------------------------------------------------------------------------
def aggregate_user_emb(
    encoder: RestaurantEncoder,
    user_idx: torch.Tensor,      # (B,)
    exclude: torch.Tensor,       # (B,) the positive r_idx to leave out of each history
    hist_items: torch.Tensor,    # (n_users, H) padded restaurant indices
    hist_ratings: torch.Tensor,  # (n_users, H) padded raw stars
    hist_mask: torch.Tensor,     # (n_users, H) bool
    global_mean: float,
) -> torch.Tensor:
    """Build (B, dim) L2-normalized user embeddings from rated-restaurant history.

    This is _pool_tags one level up: same padded-set masked pool, but the weights
    are SIGNED ratings, and the current positive must be left out (leave-one-out).

    Q: gather this batch's history rows: items, ratings, mask -> each (B, H).
    Q: leave-one-out -- zero the mask wherever items == exclude[:, None], so the
       positive can't leak into the profile that predicts it.
    Q: signed weights. Per-user mean-centering (rating - user_mean) is the spec,
       but it degenerates for the 7.2% zero-variance users (all weights 0 -> zero
       user_emb). Decide a fallback (e.g. center on global_mean for those users).
    Q: encode the history items through `encoder` (flatten to (B*H,), encode,
       reshape back to (B, H, dim)). Padded positions encode garbage but the mask
       zeroes them.
    Q: weighted masked pool over H: sum(weight * emb) / sum(|weight|)-ish. Think
       about what to divide by so a confident user and a lukewarm one are comparable.
    Q: L2-normalize the result so score = dot = cosine.
    """
    raise NotImplementedError


def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss.

    Q: -log sigmoid(pos_score - neg_score), averaged over the batch. Use
       F.logsigmoid for numerical stability (not log(sigmoid(...))).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Training loop (plumbing)
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--output-dims", type=int, default=128)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = build_training_data(args.max_history)

    encoder = RestaurantEncoder(data["features"], output_dims=args.output_dims).to(device)
    hist_items = data["hist_items"].to(device)
    hist_ratings = data["hist_ratings"].to(device)
    hist_mask = data["hist_mask"].to(device)
    global_mean = data["global_mean"]

    loader = DataLoader(BPRDataset(data), batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        encoder.train()
        total = 0.0
        for u, pos, neg in loader:
            u, pos, neg = u.to(device), pos.to(device), neg.to(device)

            user_emb = aggregate_user_emb(
                encoder, u, pos, hist_items, hist_ratings, hist_mask, global_mean
            )
            pos_emb = encoder(pos)
            neg_emb = encoder(neg)

            pos_score = (user_emb * pos_emb).sum(dim=1)   # dot = cosine (all unit)
            neg_score = (user_emb * neg_emb).sum(dim=1)
            loss = bpr_loss(pos_score, neg_score)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(u)

        print(f"epoch {epoch + 1}/{args.epochs}  bpr_loss {total / len(loader.dataset):.4f}")

    torch.save(encoder.state_dict(), OUT / "encoder.pt")
    print(f"saved {OUT}/encoder.pt")


if __name__ == "__main__":
    main()
