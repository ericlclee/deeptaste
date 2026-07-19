"""Leave-one-out ranking evaluation for the trained encoder.

Protocol (standard LOO for two-tower recsys, e.g. NCF / LightGCN):
  - Each user's most-recent review is the held-out test item (kept in the test split).
  - Build the user's profile from their TRAINING history (the test item is in a
    separate split, so it is naturally excluded; we also pass it as `exclude` so a
    prior repeat-visit to the same restaurant cannot leak into the profile).
  - Score ALL restaurants against the profile (full ranking, not sampled negatives
    -- sampled metrics are known to distort; Rendle 2020).
  - Exclude restaurants the user has already interacted with (train + val) from the
    candidate set, EXCEPT the test item itself, which must stay rankable even if the
    user visited it before.
  - Report where the test item lands: HR@k (= Recall@k for a single held-out item),
    NDCG@k, and MRR.

Only test items the user rated >= POSITIVE_THRESHOLD are evaluated ("can the model
rank a restaurant the user actually liked highly?").
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from model import RestaurantEncoder
from train import build_training_data, aggregate_user_emb, POSITIVE_THRESHOLD

OUT = Path("data/processed")


def hr_at_k(ranks: torch.Tensor, k: int) -> float:
    return (ranks <= k).float().mean().item()


def ndcg_at_k(ranks: torch.Tensor, k: int) -> float:
    # single relevant item: IDCG = 1, so NDCG = 1/log2(rank+1) when within top-k
    gain = 1.0 / torch.log2(ranks.float() + 1)
    return torch.where(ranks <= k, gain, torch.zeros_like(gain)).mean().item()


def mrr(ranks: torch.Tensor) -> float:
    return (1.0 / ranks.float()).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(OUT / "encoder.pt"))
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--output-dims", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    args = p.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")

    data = build_training_data(args.max_history)
    u2i, b2i = data["user_to_idx"], data["biz_to_idx"]
    n_users, n_rest = data["n_users"], data["n_restaurants"]

    enc = RestaurantEncoder(data["features"], output_dims=args.output_dims).to(device)
    enc.load_state_dict(torch.load(args.checkpoint, map_location=device))
    enc.eval()

    hist_items = data["hist_items"].to(device)
    hist_ratings = data["hist_ratings"].to(device)
    hist_mask = data["hist_mask"].to(device)
    global_mean = data["global_mean"]

    # ---- full seen set (train + val) per user, for candidate exclusion ----
    # (train history is capped at max_history for the profile, but candidate
    #  exclusion must use ALL interactions, so we rebuild it from the reviews.)
    reviews = pd.read_parquet(OUT / "reviews_split.parquet")
    reviews["u"] = reviews.user_id.map(u2i)
    reviews["r"] = reviews.business_id.map(b2i)
    seen = torch.zeros(n_users, n_rest, dtype=torch.bool)
    tv = reviews[reviews.split.isin(["train", "val"]) & reviews.u.notna()]
    seen[tv.u.to_numpy(dtype="int64"), tv.r.to_numpy(dtype="int64")] = True

    # ---- test set: each user's held-out positive ----
    test = reviews[(reviews.split == "test") & (reviews.stars >= POSITIVE_THRESHOLD)]
    test = test[test.u.notna() & test.r.notna()]
    test_users = torch.tensor(test.u.to_numpy(dtype="int64"))
    test_items = torch.tensor(test.r.to_numpy(dtype="int64"))
    print(f"evaluating {len(test_users):,} test users (held-out positive, full ranking over {n_rest:,})")

    # ---- precompute all restaurant embeddings once ----
    with torch.no_grad():
        R = enc(torch.arange(n_rest, device=device))  # (n_rest, dim)

    all_ranks = []
    with torch.no_grad():
        for s in range(0, len(test_users), args.batch_size):
            u = test_users[s : s + args.batch_size].to(device)
            it = test_items[s : s + args.batch_size].to(device)

            # profile with the test item left out of the aggregation
            ue = aggregate_user_emb(enc, u, it, hist_items, hist_ratings, hist_mask, global_mean)
            scores = ue @ R.T  # (B, n_rest)

            # save the test item's score, then mask out everything the user has seen
            b = torch.arange(len(u), device=device)
            test_score = scores[b, it].clone()
            seen_b = seen[u.cpu()].to(device)  # (B, n_rest)
            scores = scores.masked_fill(seen_b, float("-inf"))
            scores[b, it] = test_score  # keep the test item rankable

            # rank = 1 + (number of restaurants scoring strictly higher)
            ranks = (scores > test_score.unsqueeze(1)).sum(dim=1) + 1
            all_ranks.append(ranks.cpu())

    ranks = torch.cat(all_ranks)
    print(f"\nmedian rank: {int(ranks.median())} / {n_rest:,}")
    print(f"{'k':>6} {'HR@k':>10} {'NDCG@k':>10}")
    for k in args.ks:
        print(f"{k:>6} {hr_at_k(ranks, k):>10.4f} {ndcg_at_k(ranks, k):>10.4f}")
    print(f"MRR: {mrr(ranks):.4f}")


if __name__ == "__main__":
    main()
