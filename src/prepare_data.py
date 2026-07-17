"""Filter the Yelp Open Dataset down to restaurant reviews in a single metro."""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW = Path("data/raw")
OUT = Path("data/processed")


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


def k_core(reviews: pd.DataFrame, k_user: int, k_item: int) -> pd.DataFrame:
    """Iteratively drop users and items with too few interactions until stable."""
    while True:
        n = len(reviews)
        uc = reviews.user_id.value_counts()
        reviews = reviews[reviews.user_id.isin(uc[uc >= k_user].index)]
        bc = reviews.business_id.value_counts()
        reviews = reviews[reviews.business_id.isin(bc[bc >= k_item].index)]
        if len(reviews) == n:
            return reviews


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
