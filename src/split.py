"""Per-user temporal split: each user's most recent reviews are test, the ones
before those are val, the rest train. Proportional by default (80/10/10), so a
heavy reviewer contributes more held-out interactions than a light one.

Split is per-user rather than global-by-date so every user has a training
history to build a profile from -- a global date cutoff would leave users who
joined late with no history at all.

Each held-out review is scored as its own ranking query, so a user with three
test reviews contributes three queries rather than one. Other held-out reviews
by the same user stay in the candidate set (only train+val are masked), which
is correct: at the moment of any one visit, the others were genuinely still
available choices.

Must run before features.py. Restaurant features are built from training
reviews only, so the split boundary has to exist before any of them are
computed.

Repeat visits are collapsed to the user's LATEST review of that restaurant before
anything else happens (~4% of (user, restaurant) pairs, ~8% of all reviews --
not a rare edge case). An outdated review would otherwise sit alongside the
current one everywhere downstream: double-counted (or outright contradicted) in
the profile aggregation, redundant/stale in the BPR positive set, and eligible
as a "disliked" rated-negative for a restaurant the user's later review says
they now like (or vice versa).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("DEEP_TASTE_DATA", "data/yelp_philadelphia"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--val-frac", type=float, default=0.10)
    args = p.parse_args()

    reviews = pd.read_parquet(OUT / "reviews.parquet")
    reviews["date"] = pd.to_datetime(reviews["date"])

    n_before = len(reviews)
    reviews = reviews.sort_values(["user_id", "business_id", "date"], kind="stable")
    reviews = reviews.drop_duplicates(subset=["user_id", "business_id"], keep="last")
    n_dropped = n_before - len(reviews)
    print(
        f"dropped {n_dropped:,} stale repeat-visit reviews "
        f"({n_dropped / n_before * 100:.2f}%), keeping each user's latest review per restaurant"
    )

    reviews = reviews.sort_values(["user_id", "date"], kind="stable")

    # 0 = most recent review for that user
    recency = reviews.groupby("user_id").cumcount(ascending=False)
    n_user = reviews.groupby("user_id").user_id.transform("size")

    # At least one held-out review each, so no user is evaluated on nothing.
    # Rounding rather than truncating: the median user has 16 reviews, so
    # floor(0.10 * 16) = 1 for most of the catalog and the realised split
    # drifts to 83/8/8. Rounding lands at 80/10/10 as asked. k-core >= 10
    # guarantees a training history survives either way.
    n_test = np.maximum(1, np.round(n_user * args.test_frac)).astype(int)
    n_val = np.maximum(1, np.round(n_user * args.val_frac)).astype(int)

    reviews["split"] = "train"
    reviews.loc[recency < n_test, "split"] = "test"
    reviews.loc[(recency >= n_test) & (recency < n_test + n_val), "split"] = "val"

    counts = reviews.split.value_counts()
    tot = len(reviews)
    print(
        f"train {counts.get('train', 0):,} ({counts.get('train', 0) / tot * 100:.1f}%) | "
        f"val {counts.get('val', 0):,} ({counts.get('val', 0) / tot * 100:.1f}%) | "
        f"test {counts.get('test', 0):,} ({counts.get('test', 0) / tot * 100:.1f}%)"
    )

    train = reviews[reviews.split == "train"]
    print(f"users with train history: {train.user_id.nunique():,} / {reviews.user_id.nunique():,}")
    print(f"restaurants seen in train: {train.business_id.nunique():,} / {reviews.business_id.nunique():,}")

    unseen = ~reviews[reviews.split == "test"].business_id.isin(train.business_id.unique())
    print(f"test reviews on restaurants never seen in train: {unseen.sum():,} ({unseen.mean() * 100:.2f}%)")

    print(f"train date range: {train.date.min().date()} -> {train.date.max().date()}")
    print(f"test  date range: {reviews[reviews.split == 'test'].date.min().date()} -> {reviews[reviews.split == 'test'].date.max().date()}")

    reviews.to_parquet(OUT / "reviews_split.parquet", index=False)
    print(f"wrote {OUT}/reviews_split.parquet")


if __name__ == "__main__":
    main()
