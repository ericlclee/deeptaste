"""Filter the Yelp Open Dataset down to restaurant reviews in a single metro."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from common import k_core
from tqdm import tqdm

RAW = Path(os.environ.get("DEEP_TASTE_RAW", "data/raw"))
OUT = Path(os.environ.get("DEEP_TASTE_DATA", "data/yelp_philadelphia"))


def load_businesses(city: str | None) -> pd.DataFrame:
    rows = []
    with open(RAW / "yelp_academic_dataset_business.json") as f:
        for line in f:
            b = json.loads(line)
            cats = b.get("categories") or ""
            if "Restaurants" not in cats and "Food" not in cats:
                continue
            if city and b["city"].strip().lower() != city.lower():
                continue
            rows.append(
                {
                    "business_id": b["business_id"],
                    "name": b["name"],
                    "city": b["city"].strip(),
                    "state": b["state"],
                    "latitude": b["latitude"],
                    "longitude": b["longitude"],
                    "stars": b["stars"],
                    "review_count": b["review_count"],
                    "categories": cats,
                    "price": (b.get("attributes") or {}).get("RestaurantsPriceRange2"),
                }
            )
    return pd.DataFrame(rows)


def load_reviews(keep_ids: set[str]) -> pd.DataFrame:
    rows = []
    with open(RAW / "yelp_academic_dataset_review.json") as f:
        for line in tqdm(f, desc="scanning reviews", unit=" lines"):
            r = json.loads(line)
            if r["business_id"] not in keep_ids:
                continue
            rows.append(
                {
                    "user_id": r["user_id"],
                    "business_id": r["business_id"],
                    "stars": r["stars"],
                    "date": r["date"],
                    "text": r["text"],
                }
            )
    return pd.DataFrame(rows)




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="Philadelphia")
    p.add_argument("--k-user", type=int, default=5)
    p.add_argument("--k-item", type=int, default=5)
    p.add_argument("--survey", action="store_true", help="just print city counts and exit")
    args = p.parse_args()

    if args.survey:
        biz = load_businesses(None)
        print(biz.groupby(["city", "state"]).size().sort_values(ascending=False).head(15))
        return

    OUT.mkdir(parents=True, exist_ok=True)

    biz = load_businesses(args.city)
    print(f"{len(biz):,} restaurants in {args.city}")

    reviews = load_reviews(set(biz.business_id))
    print(f"{len(reviews):,} reviews before k-core")

    reviews = k_core(reviews, args.k_user, args.k_item)
    biz = biz[biz.business_id.isin(reviews.business_id.unique())]
    print(
        f"{len(reviews):,} reviews after k-core | "
        f"{reviews.user_id.nunique():,} users | {len(biz):,} restaurants | "
        f"density {len(reviews) / (reviews.user_id.nunique() * len(biz)) * 100:.3f}%"
    )

    biz.to_parquet(OUT / "businesses.parquet", index=False)
    reviews.to_parquet(OUT / "reviews.parquet", index=False)
    print(f"wrote {OUT}/businesses.parquet, {OUT}/reviews.parquet")


if __name__ == "__main__":
    main()
