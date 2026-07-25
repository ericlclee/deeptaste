"""Leave-one-out ranking evaluation for the trained model, against baselines.

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

BASELINES are reported alongside the model, under the identical protocol and
candidate masking. An HR@10 number is meaningless on its own -- 6,176 candidates
makes random ~0.0016, but "recommend whatever is popular" and "recommend whatever
is nearby" are the bars a personalized model actually has to clear, and in
recommender systems they are notoriously hard to beat. A model that loses to
popularity is not a working recommender no matter how principled its architecture.
"""

import argparse

import pandas as pd
import torch

from model import RestaurantEncoder, UserProfile
from train import build_training_data, POSITIVE_THRESHOLD, OUT


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


def ranks_from_scores(
    scores: torch.Tensor, items: torch.Tensor, seen_rows: torch.Tensor
) -> torch.Tensor:
    """(B, n_rest) scores -> (B,) rank of each held-out item (1 = top).

    Shared by the model and every baseline so they cannot accidentally differ in
    candidate masking -- the one thing that would make the comparison a lie.

    Ties get the MIDRANK (average of the positions they span) rather than the
    optimistic "1 + count of strictly greater". Continuous model scores never
    tie, but a baseline like mean star rating has ~9 distinct values across the
    whole catalog, and optimistic ranking would score every 5-star restaurant as
    rank 1 -- handing the baseline an HR@5 of 0.02 that is pure tie-breaking
    luck. Optimistic ranking flatters exactly the baselines the model must beat.
    """
    b = torch.arange(len(items), device=scores.device)
    held = scores[b, items].clone()
    scores = scores.masked_fill(seen_rows, float("-inf"))
    scores[b, items] = held  # stays rankable even on a repeat visit
    greater = (scores > held.unsqueeze(1)).sum(dim=1)
    tied = (scores == held.unsqueeze(1)).sum(dim=1) - 1  # exclude the item itself
    return (greater + 1 + tied.float() / 2).cpu()


def rank_heldout(
    enc,
    user_profile,
    eval_users: torch.Tensor,
    eval_items: torch.Tensor,
    seen: torch.Tensor,
    hist_items: torch.Tensor,
    hist_ratings: torch.Tensor,
    hist_mask: torch.Tensor,
    n_rest: int,
    device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Full-catalog rank of each user's held-out item. See module docstring."""
    was_training = enc.training
    enc.eval()
    user_profile.eval()
    all_ranks = []
    with torch.no_grad():
        all_idx = torch.arange(n_rest, device=device)
        R = enc(all_idx)  # (n_rest, dim)
        item_bias = enc.bias(all_idx)  # (n_rest,)
        for s in range(0, len(eval_users), batch_size):
            u = eval_users[s : s + batch_size].to(device)
            it = eval_items[s : s + batch_size].to(device)

            # profile with the held-out item left out of the aggregation
            ue = user_profile(enc, u, it, hist_items, hist_ratings, hist_mask)
            scores = ue @ R.T + item_bias.unsqueeze(0)
            all_ranks.append(ranks_from_scores(scores, it, seen[u.cpu()].to(device)))
    if was_training:
        enc.train()
    return torch.cat(all_ranks)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def rank_item_scalar(
    item_scores: torch.Tensor,
    eval_users: torch.Tensor,
    eval_items: torch.Tensor,
    seen: torch.Tensor,
    device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Baselines that score every user identically (popularity, rating, random).

    Non-personalized by construction: the only per-user variation comes from the
    candidate mask.
    """
    item_scores = item_scores.to(device)
    all_ranks = []
    for s in range(0, len(eval_users), batch_size):
        u = eval_users[s : s + batch_size]
        it = eval_items[s : s + batch_size].to(device)
        scores = item_scores.unsqueeze(0).expand(len(u), -1).clone()
        all_ranks.append(ranks_from_scores(scores, it, seen[u].to(device)))
    return torch.cat(all_ranks)


def rank_geo(
    latlng: torch.Tensor,
    eval_users: torch.Tensor,
    eval_items: torch.Tensor,
    seen: torch.Tensor,
    hist_items: torch.Tensor,
    hist_mask: torch.Tensor,
    device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Personalized-but-trivial baseline: rank by proximity to the centroid of
    the user's previously-rated restaurants. This is the one to beat -- people
    eat near where they already eat, and any embedding model that can't clear it
    is being outperformed by two lines of arithmetic."""
    latlng = latlng.to(device)
    all_ranks = []
    for s in range(0, len(eval_users), batch_size):
        u = eval_users[s : s + batch_size].to(device)
        it = eval_items[s : s + batch_size].to(device)

        items = hist_items[u]
        mask = (hist_mask[u] & (items != it.unsqueeze(1))).float()  # (B, H)
        pts = latlng[items] * mask.unsqueeze(-1)  # (B, H, 2)
        centroid = pts.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)  # (B, 2)

        # Explicit difference-then-norm, NOT torch.cdist: cdist defaults to the
        # matmul expansion |a-b|^2 = |a|^2 - 2a.b + |b|^2, which catastrophically
        # cancels here. Coordinates are ~(40, -75) while the distances that
        # matter are ~0.001 degrees, so in float32 the |a|^2 terms (~7000) leave
        # absolute error larger than the signal -- it silently reported ~28% of
        # held-out items as the strict nearest restaurant, making this baseline
        # look 100x stronger than it is.
        dist = torch.linalg.norm(centroid[:, None, :] - latlng[None, :, :], dim=-1)
        all_ranks.append(ranks_from_scores(-dist, it, seen[u.cpu()].to(device)))
    return torch.cat(all_ranks)


def report(name: str, ranks: torch.Tensor, ks, n_rest: int) -> None:
    cells = "  ".join(
        f"{hr_at_k(ranks, k):>8.4f} {ndcg_at_k(ranks, k):>8.4f}" for k in ks
    )
    print(f"{name:<22} {int(ranks.median()):>7} {cells}  {mrr(ranks):>8.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(OUT / "encoder.pt"))
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")
    torch.manual_seed(args.seed)

    data = build_training_data(args.max_history)
    u2i, b2i = data["user_to_idx"], data["biz_to_idx"]
    n_users, n_rest = data["n_users"], data["n_restaurants"]
    feats = data["features"]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    enc = RestaurantEncoder(
        feats,
        output_dims=ckpt.get("output_dims", 128),
        use_id_emb=ckpt.get("use_id_emb", True),
    ).to(device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    user_profile = UserProfile(ckpt.get("global_mean", data["global_mean"])).to(device)
    user_profile.load_state_dict(ckpt["user_profile"])
    user_profile.eval()

    hist_items = data["hist_items"].to(device)
    hist_ratings = data["hist_ratings"].to(device)
    hist_mask = data["hist_mask"].to(device)

    # ---- candidate exclusion: all train+val interactions (not just the capped
    #      max_history used for the profile), rebuilt from the reviews table ----
    reviews = index_reviews(u2i, b2i)
    seen = build_seen(reviews, n_users, n_rest, ["train", "val"])

    # ---- test set: each user's held-out positive ----
    test_users, test_items = eval_pairs(reviews, "test")
    print(
        f"evaluating {len(test_users):,} test users "
        f"(held-out positive, full ranking over {n_rest:,})\n"
    )

    header = "  ".join(f"{'HR@' + str(k):>8} {'NDCG@' + str(k):>8}" for k in args.ks)
    print(f"{'':<22} {'medRank':>7} {header}  {'MRR':>8}")

    # numeric columns are z-scored, which is monotonic -- ranking by the z-score
    # is identical to ranking by the raw feature.
    report("random", rank_item_scalar(torch.rand(n_rest), test_users, test_items, seen, device, args.batch_size), args.ks, n_rest)
    report("popularity", rank_item_scalar(feats["numeric"][:, 1], test_users, test_items, seen, device, args.batch_size), args.ks, n_rest)
    report("mean rating", rank_item_scalar(feats["numeric"][:, 0], test_users, test_items, seen, device, args.batch_size), args.ks, n_rest)
    report(
        "geo proximity",
        rank_geo(feats["latlng"], test_users, test_items, seen, hist_items, hist_mask, device, args.batch_size),
        args.ks,
        n_rest,
    )
    report(
        "MODEL",
        rank_heldout(
            enc, user_profile, test_users, test_items, seen, hist_items,
            hist_ratings, hist_mask, n_rest, device, args.batch_size,
        ),
        args.ks,
        n_rest,
    )


if __name__ == "__main__":
    main()
