"""BPR training loop for the two-tower recommender.

user_emb is a signed-weighted pool of the SAME encoder's outputs on the
restaurants that user rated (see model.UserProfile). So gradients flow from the
BPR loss, through the positive/negative embeddings AND through the user's
history embeddings, all into the one shared RestaurantEncoder -- which is what
makes the two-stage design "effectively end-to-end".

Scoring is cos(user, item) + item_bias, not cosine alone: both towers are
normalized, so the cosine term carries "does this match your taste" and the bias
term carries "is this place generally any good". Collapsing both into one
normalized dot product means a model that ranks well for a typical user has to
fight its own geometry.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import RestaurantEncoder, UserProfile

# Default is repo-relative; DEEP_TASTE_DATA lets a scheduler point at scratch
# without depending on the job's working directory.
OUT = Path(os.environ.get("DEEP_TASTE_DATA", "data/yelp_philadelphia"))
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
        # A (user, positive) pair is only valid if the user has >=1 OTHER rated
        # restaurant: after leave-one-out removes the positive, an empty history
        # gives a zero user_emb, whose F.normalize gradient is ~1/norm -> explodes
        # to NaN. ~11 users rate a single restaurant repeatedly and hit this.
        distinct = set(grp.r_idx.to_numpy().tolist())
        for r_idx, stars in zip(grp.r_idx.to_numpy(), grp.stars.to_numpy()):
            if stars >= POSITIVE_THRESHOLD and len(distinct - {int(r_idx)}) > 0:
                positives.append((u, int(r_idx)))

    global_mean = float(train.stars.mean())
    print(
        f"{n_users:,} users | {n_restaurants:,} restaurants | {len(positives):,} positive interactions"
    )

    return {
        "n_users": n_users,
        "n_restaurants": n_restaurants,
        "hist_items": torch.from_numpy(hist_items),
        "hist_ratings": torch.from_numpy(hist_ratings),
        "hist_mask": torch.from_numpy(hist_mask),
        "positives": positives,
        "global_mean": global_mean,
        "features": feats,
        "user_to_idx": user_to_idx,
        "biz_to_idx": biz_to_idx,
    }


def build_hard_candidates(feats: dict, k: int = 30) -> list[np.ndarray]:
    """For each restaurant, up to `k` OTHER restaurants that share a price tier
    and at least one cuisine tag, restricted to its geographic neighborhood --
    the "same cuisine/price/geo cluster" hard-negative pool from the project
    spec. A restaurant with too few same-tag-and-price geo-neighbors (rare
    tags, missing price) falls back to geo+price, then to geo alone, so every
    restaurant still gets a same-*area* pool even if cuisine/price can't be
    matched.
    """
    from sklearn.neighbors import NearestNeighbors

    missing = [k for k in ("latlng", "price", "tag_ids") if k not in feats]
    if missing:
        raise KeyError(
            f"hard-negative sampling needs {missing}, which this data source does not "
            "provide (the TripAdvisor London set has no coordinates, price tier or "
            "cuisine categories). Run with --hard-neg-ratio 0."
        )

    latlng = feats["latlng"].numpy()
    price = feats["price"].numpy()
    price_mask = feats["price_mask"].numpy()
    tag_ids = feats["tag_ids"].numpy()  # (n, t_max), 0 = padding
    n = latlng.shape[0]

    # Wide geo pool (k*4) so the tag/price filters below have enough to work
    # with; +1 because a restaurant is always its own nearest neighbor.
    nn = NearestNeighbors(n_neighbors=min(k * 4 + 1, n)).fit(latlng)
    _, geo_idx = nn.kneighbors(latlng)

    tag_sets = [set(row[row > 0].tolist()) for row in tag_ids]

    def same_price(i, j):
        return not price_mask[i] or not price_mask[j] or price[i] == price[j]

    candidates = []
    for i in range(n):
        neighbors = geo_idx[i][geo_idx[i] != i]

        pool = [j for j in neighbors if tag_sets[j] & tag_sets[i] and same_price(i, j)]
        if len(pool) < 5:
            pool = [j for j in neighbors if same_price(i, j)]
        if len(pool) < 5:
            pool = neighbors.tolist()

        candidates.append(np.array(pool[:k], dtype=np.int64))
    return candidates


class BPRDataset(Dataset):
    """One example = (user_idx, positive_idx, negative_idx). Three negative
    sources, tried in priority order:

    1. Rated negative: a restaurant THIS user rated below POSITIVE_THRESHOLD
       (used with probability `rated_neg_ratio` when the user has any -- a
       below-4-star review, including a "meh" 3-star one, is real personal
       dislike, and BPR never otherwise gets to contrast it against something
       the user liked: it's excluded from the candidate set at eval time and
       (until now) from negative sampling too, so its only effect was a mild
       pull on the aggregated user_emb).
    2. Hard negative: with probability `hard_neg_ratio`, drawn from
       `hard_candidates[pos]` -- restaurants that look like the positive (same
       price tier, nearby, sharing a cuisine tag) but weren't chosen. Random
       negatives are trivially distinguishable (wrong cuisine, wrong side of
       town), so the model can satisfy BPR on them without learning much.
    3. Uniform random negative: the fallback whenever 1-2 don't apply or their
       pool is empty/already-seen.
    """

    def __init__(
        self,
        data: dict,
        hard_neg_ratio: float = 0.0,
        hard_neg_k: int = 30,
        rated_neg_ratio: float = 1.0,
    ):
        self.positives = data["positives"]
        self.n_restaurants = data["n_restaurants"]
        self.hard_neg_ratio = hard_neg_ratio
        self.rated_neg_ratio = rated_neg_ratio
        # set of interacted items per user, for negative rejection
        self.seen = [set() for _ in range(data["n_users"])]
        # subset of the above this user rated < POSITIVE_THRESHOLD (explicit
        # dislikes), for rated-negative sampling
        self.rated_neg = [None] * data["n_users"]
        items, ratings, mask = data["hist_items"], data["hist_ratings"], data["hist_mask"]
        for u in range(data["n_users"]):
            self.seen[u] = set(items[u][mask[u]].tolist())
            disliked_mask = mask[u] & (ratings[u] < POSITIVE_THRESHOLD)
            self.rated_neg[u] = items[u][disliked_mask].numpy()

        self.hard_candidates = (
            build_hard_candidates(data["features"], k=hard_neg_k)
            if hard_neg_ratio > 0
            else None
        )

    def __len__(self):
        return len(self.positives)

    def _random_negative(self, seen: set) -> int:
        j = np.random.randint(self.n_restaurants)
        while j in seen:
            j = np.random.randint(self.n_restaurants)
        return j

    def __getitem__(self, i):
        u, pos = self.positives[i]
        if np.random.rand() < self.rated_neg_ratio:
            # exclude pos itself: a repeat visitor can rate the SAME restaurant
            # both >=POSITIVE_THRESHOLD (today's positive) and below it (a
            # different visit), which would otherwise let pos == neg through.
            candidates = self.rated_neg[u][self.rated_neg[u] != pos]
            if len(candidates) > 0:
                return u, pos, int(np.random.choice(candidates))
        seen = self.seen[u]
        if self.hard_candidates is not None and np.random.rand() < self.hard_neg_ratio:
            for j in np.random.permutation(self.hard_candidates[pos]):
                if j != pos and j not in seen:
                    return u, pos, int(j)
            # pool empty or fully seen -- fall through to a random negative
        return u, pos, self._random_negative(seen)


def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss: -log sigmoid(pos - neg). F.logsigmoid
    rather than log(sigmoid(...)) for numerical stability."""
    return -F.logsigmoid(pos_score - neg_score).mean()


# ---------------------------------------------------------------------------
# Training loop (plumbing)
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)  # 1e-3 caused dying-ReLU collapse
    p.add_argument("--clip", type=float, default=1.0)  # grad-norm clip for stability
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--output-dims", type=int, default=128)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="explicit checkpoint path. Overrides --run-name.",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="write to runs/<data-dir-name>/<run-name>/encoder.pt instead of "
        "overwriting <data>/encoder.pt. Every unnamed run clobbers the previous "
        "one, which silently destroys the model an ablation is meant to be "
        "compared against; name your runs and they accumulate side by side.",
    )
    p.add_argument(
        "--init-checkpoint",
        default=None,
        help="warm-start the encoder from a checkpoint (e.g. pretrain_encoder.py's "
        "content-similarity pretraining) instead of random init",
    )
    p.add_argument("--eval-k", type=int, default=10)  # k for the per-epoch HR/NDCG
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument(
        "--hard-neg-ratio",
        type=float,
        default=0.0,
        help="fraction of negatives drawn from the positive's cuisine/price/geo "
        "cluster instead of uniformly at random (0 = pure random, the original "
        "behavior; try 0.5 as a starting point)",
    )
    p.add_argument(
        "--hard-neg-k",
        type=int,
        default=30,
        help="candidate pool size per restaurant for hard-negative sampling",
    )
    p.add_argument(
        "--rated-neg-ratio",
        type=float,
        default=1.0,
        help="probability of using a restaurant this user rated below "
        "POSITIVE_THRESHOLD as the negative, when they have one (highest-"
        "priority negative source, tried before hard/random); 0 disables it",
    )
    p.add_argument(
        "--no-id-emb",
        action="store_true",
        help="ablate the per-restaurant id embedding, leaving a content-only item "
        "tower. Use to measure how much of the model's ranking comes from "
        "memorization vs. generalizable content features.",
    )
    p.add_argument(
        "--fixed-agg",
        action="store_true",
        help="pin the user tower's per-star weights at (rating - global_mean) "
        "instead of learning them",
    )
    p.add_argument(
        "--eval-test",
        action="store_true",
        help="also print test metrics each epoch. Off by default: checkpoints are "
        "selected on val, and repeatedly reading test biases it into a second val set.",
    )
    args = p.parse_args()
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.run_name:
        ckpt_path = Path("runs") / OUT.name / args.run_name / "encoder.pt"
    else:
        ckpt_path = OUT / "encoder.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {ckpt_path}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")
    data = build_training_data(args.max_history)

    encoder = RestaurantEncoder(
        data["features"], output_dims=args.output_dims, use_id_emb=not args.no_id_emb
    ).to(device)
    if args.init_checkpoint:
        encoder.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"warm-started encoder from {args.init_checkpoint}")
    global_mean = data["global_mean"]
    user_profile = UserProfile(global_mean, learned=not args.fixed_agg).to(device)

    hist_items = data["hist_items"].to(device)
    hist_ratings = data["hist_ratings"].to(device)
    hist_mask = data["hist_mask"].to(device)

    print(
        f"hard-neg ratio: {args.hard_neg_ratio} (k={args.hard_neg_k})  |  "
        f"rated-neg ratio: {args.rated_neg_ratio}"
    )
    print(
        f"item tower: {'content + id embedding' if not args.no_id_emb else 'content only (id ablated)'}"
        f"  |  user tower: {'learned' if not args.fixed_agg else 'fixed'} per-star weights"
    )
    loader = DataLoader(
        BPRDataset(
            data,
            hard_neg_ratio=args.hard_neg_ratio,
            hard_neg_k=args.hard_neg_k,
            rated_neg_ratio=args.rated_neg_ratio,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(user_profile.parameters()), lr=args.lr
    )

    # Imported here, not at module scope: evaluate.py imports from train.py, so a
    # top-level import would be circular.
    from evaluate import index_reviews, build_seen, eval_pairs, rank_heldout, hr_at_k, ndcg_at_k

    n_rest = data["n_restaurants"]
    reviews = index_reviews(data["user_to_idx"], data["biz_to_idx"])

    # Val: rank the held-out val item against everything except the train history.
    # (Excluding val from the candidate set would remove the very item we score.)
    val_seen = build_seen(reviews, data["n_users"], n_rest, ["train"])
    val_users, val_items = eval_pairs(reviews, "val")
    print(f"val: {len(val_users):,} users with a held-out positive")

    test_seen = test_users = test_items = None
    if args.eval_test:
        test_seen = build_seen(reviews, data["n_users"], n_rest, ["train", "val"])
        test_users, test_items = eval_pairs(reviews, "test")
        print(
            f"test: {len(test_users):,} users -- REPORTED ONLY, not used for checkpoint "
            "selection. Tuning against these numbers biases them."
        )

    def eval_ndcg(users, items, seen):
        ranks = rank_heldout(
            encoder, user_profile, users, items, seen, hist_items, hist_ratings,
            hist_mask, n_rest, device, args.eval_batch_size,
        )
        return hr_at_k(ranks, args.eval_k), ndcg_at_k(ranks, args.eval_k)

    trainable = list(encoder.parameters()) + list(user_profile.parameters())
    best_ndcg, best_epoch = -1.0, 0
    for epoch in range(args.epochs):
        encoder.train()
        user_profile.train()
        total = 0.0
        for u, pos, neg in loader:
            u, pos, neg = u.to(device), pos.to(device), neg.to(device)

            user_emb = user_profile(
                encoder, u, pos, hist_items, hist_ratings, hist_mask
            )
            pos_emb = encoder(pos)
            neg_emb = encoder(neg)

            # cosine (both towers unit-norm) + per-item popularity bias
            pos_score = (user_emb * pos_emb).sum(dim=1) + encoder.bias(pos)
            neg_score = (user_emb * neg_emb).sum(dim=1) + encoder.bias(neg)
            loss = bpr_loss(pos_score, neg_score)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.clip)
            opt.step()
            total += loss.item() * len(u)

        # Train loss is a tripwire (NaN / divergence / flatline), not a progress
        # metric: its scale depends on the negative sampler, so it is not comparable
        # across configs. Val NDCG is what selects the checkpoint.
        line = (
            f"epoch {epoch + 1}/{args.epochs}  bpr_loss {total / len(loader.dataset):.4f}"
        )

        val_hr, val_ndcg = eval_ndcg(val_users, val_items, val_seen)
        line += f"  |  val HR@{args.eval_k} {val_hr:.4f}  NDCG@{args.eval_k} {val_ndcg:.4f}"

        if args.eval_test:
            test_hr, test_ndcg = eval_ndcg(test_users, test_items, test_seen)
            line += f"  |  test HR@{args.eval_k} {test_hr:.4f}  NDCG@{args.eval_k} {test_ndcg:.4f}"

        if val_ndcg > best_ndcg:
            best_ndcg, best_epoch = val_ndcg, epoch + 1
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "user_profile": user_profile.state_dict(),
                    "output_dims": args.output_dims,
                    "use_id_emb": not args.no_id_emb,
                    "global_mean": global_mean,
                    # Which content branches this model was built with. Without
                    # it, loading a London checkpoint (no tag/price/geo branches)
                    # against Yelp features fails as an opaque shape mismatch
                    # deep inside a Linear rather than saying what is wrong.
                    "modalities": {
                        "tags": encoder.has_tags,
                        "price": encoder.has_price,
                        "geo": encoder.has_geo,
                    },
                    "data_dir": str(OUT),
                    "epoch": epoch + 1,
                    "val_ndcg": val_ndcg,
                },
                ckpt_path,
            )
            line += "  *best"
        print(line)

    print(
        f"\nsaved {ckpt_path} from epoch {best_epoch} "
        f"(best val NDCG@{args.eval_k} {best_ndcg:.4f})"
    )
    # The learned per-star weights are the interpretable payoff of a trainable
    # user tower -- print them rather than leaving them buried in the checkpoint.
    w = user_profile.level_weight.detach().cpu()
    print(f"user tower, {'learned' if not args.fixed_agg else 'fixed'} weight per star rating:")
    print("  " + "  ".join(f"{s}star={w[s - 1]:+.3f}" for s in range(1, len(w) + 1)))


if __name__ == "__main__":
    main()
