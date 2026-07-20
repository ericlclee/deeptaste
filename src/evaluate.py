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

import pandas as pd
import torch

from model import RestaurantEncoder
from train import build_training_data, aggregate_user_emb, POSITIVE_THRESHOLD, OUT


def hr_at_k(ranks: torch.Tensor, k: int) -> float:
    return (ranks <= k).float().mean().item()


def ndcg_at_k(ranks: torch.Tensor, k: int) -> float:
    # single relevant item: IDCG = 1, so NDCG = 1/log2(rank+1) when within top-k
    gain = 1.0 / torch.log2(ranks.float() + 1)
    return torch.where(ranks <= k, gain, torch.zeros_like(gain)).mean().item()


def mrr(ranks: torch.Tensor) -> float:
    return (1.0 / ranks.float()).mean().item()


# ---------------------------------------------------------------------------
# Reusable pieces -- train.py calls these once per epoch for val monitoring.
# ---------------------------------------------------------------------------
def index_reviews(u2i: dict, b2i: dict) -> pd.DataFrame:
    """reviews_split.parquet with user/business ids mapped to model indices."""
    reviews = pd.read_parquet(OUT / "reviews_split.parquet")
    reviews["u"] = reviews.user_id.map(u2i)
    reviews["r"] = reviews.business_id.map(b2i)
    return reviews


def build_seen(reviews: pd.DataFrame, n_users: int, n_rest: int, splits) -> torch.Tensor:
    """(n_users, n_rest) bool mask of interactions to drop from the candidate set.

    Pass only the splits that precede the one being evaluated: ranking a val item
    against a candidate set that already excludes val is leakage.
    """
    m = torch.zeros(n_users, n_rest, dtype=torch.bool)
    s = reviews[reviews.split.isin(splits) & reviews.u.notna() & reviews.r.notna()]
    # torch.tensor(...) rather than raw numpy: indexing with a non-writable array
    # that pandas returns warns about undefined write behaviour.
    m[
        torch.tensor(s.u.to_numpy(dtype="int64")),
        torch.tensor(s.r.to_numpy(dtype="int64")),
    ] = True
    return m


def eval_pairs(reviews: pd.DataFrame, split: str):
    """(users, items) for one held-out positive per user in `split`."""
    s = reviews[(reviews.split == split) & (reviews.stars >= POSITIVE_THRESHOLD)]
    s = s[s.u.notna() & s.r.notna()]
    return (
        torch.tensor(s.u.to_numpy(dtype="int64")),
        torch.tensor(s.r.to_numpy(dtype="int64")),
    )


def rank_heldout(
    enc,
    eval_users: torch.Tensor,
    eval_items: torch.Tensor,
    seen: torch.Tensor,
    hist_items: torch.Tensor,
    hist_ratings: torch.Tensor,
    hist_mask: torch.Tensor,
    global_mean: float,
    n_rest: int,
    device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Full-catalog rank of each user's held-out item (1 = top). See module docstring."""
    was_training = enc.training
    enc.eval()
    all_ranks = []
    with torch.no_grad():
        R = enc(torch.arange(n_rest, device=device))  # (n_rest, dim)
        for s in range(0, len(eval_users), batch_size):
            u = eval_users[s : s + batch_size].to(device)
            it = eval_items[s : s + batch_size].to(device)

            # profile with the held-out item left out of the aggregation
            ue = aggregate_user_emb(
                enc, u, it, hist_items, hist_ratings, hist_mask, global_mean
            )
            scores = ue @ R.T  # (B, n_rest)

            # save the held-out item's score, then mask out everything already seen
            b = torch.arange(len(u), device=device)
            held_score = scores[b, it].clone()
            scores = scores.masked_fill(seen[u.cpu()].to(device), float("-inf"))
            scores[b, it] = held_score  # keep it rankable even on a repeat visit

            all_ranks.append(((scores > held_score.unsqueeze(1)).sum(dim=1) + 1).cpu())
    if was_training:
        enc.train()
    return torch.cat(all_ranks)


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

    # ---- candidate exclusion: all train+val interactions (not just the capped
    #      max_history used for the profile), rebuilt from the reviews table ----
    reviews = index_reviews(u2i, b2i)
    seen = build_seen(reviews, n_users, n_rest, ["train", "val"])

    # ---- test set: each user's held-out positive ----
    test_users, test_items = eval_pairs(reviews, "test")
    print(f"evaluating {len(test_users):,} test users (held-out positive, full ranking over {n_rest:,})")

    ranks = rank_heldout(
        enc, test_users, test_items, seen, hist_items, hist_ratings, hist_mask,
        global_mean, n_rest, device, args.batch_size,
    )
    print(f"\nmedian rank: {int(ranks.median())} / {n_rest:,}")
    print(f"{'k':>6} {'HR@k':>10} {'NDCG@k':>10}")
    for k in args.ks:
        print(f"{k:>6} {hr_at_k(ranks, k):>10.4f} {ndcg_at_k(ranks, k):>10.4f}")
    print(f"MRR: {mrr(ranks):.4f}")


if __name__ == "__main__":
    main()
