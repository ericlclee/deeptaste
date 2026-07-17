"""Per-user temporal split: each user's most recent review is test, next-most-recent is val.

Must run before features.py. Restaurant text embeddings are built from training
reviews only, so the split boundary has to exist before any text is encoded.
"""

import argparse
from pathlib import Path

import pandas as pd

OUT = Path("data/processed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-test", type=int, default=1)
    p.add_argument("--n-val", type=int, default=1)
    args = p.parse_args()

    reviews = pd.read_parquet(OUT / "reviews.parquet")
    reviews["date"] = pd.to_datetime(reviews["date"])
    reviews = reviews.sort_values(["user_id", "date"], kind="stable")

    # 0 = most recent review for that user
    recency = reviews.groupby("user_id").cumcount(ascending=False)

    reviews["split"] = "train"
    reviews.loc[recency < args.n_test, "split"] = "test"
    reviews.loc[
        (recency >= args.n_test) & (recency < args.n_test + args.n_val), "split"
    ] = "val"

    counts = reviews.split.value_counts()
    print(f"train {counts.get('train', 0):,} | val {counts.get('val', 0):,} | test {counts.get('test', 0):,}")

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
