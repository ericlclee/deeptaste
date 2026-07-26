"""Convert the Zenodo TripAdvisor London reviews CSV into this project's
businesses/reviews contract, so the rest of the pipeline runs unchanged.

Source: "A TripAdvisor Dataset for Dyadic Context Analysis" (Zenodo 6583422),
CC-BY-NC 4.0 -- non-commercial use only. English-language reviews only, which
is a real selection bias in a city as multilingual as London: reviewers who
write in other languages are absent, so the user population skews toward
English speakers and tourists.

What carries over and what does not
-----------------------------------
The interaction side is complete -- user, restaurant, rating, date and full
review text -- which covers BPR training, the ABSA text branch, id embeddings
and the popularity bias.

Three CONTENT features have no equivalent in this source: cuisine categories,
price tier, and lat/lng. They are emitted as nulls rather than being guessed
at, and features.py drops the corresponding encoder branches. Two consequences
worth knowing before comparing London results against Philadelphia:

  - hard-negative sampling and content pretraining both key on cuisine/price/
    geo, so neither is available here.
  - the geo-proximity baseline cannot be computed, leaving three baselines
    rather than four.

Recovering them means joining OpenStreetMap (which has London restaurants with
cuisine tags and coordinates) on restaurant name -- messy fuzzy matching, and
deliberately out of scope here.

Restaurant identity comes from the numeric id embedded in the TripAdvisor URL
(".../Restaurant_Review-g186338-d9994333-Reviews-..." -> "d9994333"), not from
restaurant_name: names collide across chains and locations, which would merge
distinct restaurants into one item.

Usage:
    python src/prepare_london.py
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd

from prepare_data import k_core

RAW = Path(os.environ.get("DEEP_TASTE_RAW", "data/raw"))
# Deliberately NOT defaulting to data/processed: that directory holds the Yelp
# build, and this script would overwrite businesses.parquet/reviews.parquet in
# place. London is a different source, so it gets its own directory and the two
# never collide -- every downstream script already reads DEEP_TASTE_DATA.
OUT = Path(os.environ.get("DEEP_TASTE_DATA", "data/london"))

# Only the columns we actually consume. review_preview is a truncated copy of
# review_full, and title_review/sample/parse_count are unused -- skipping them
# roughly halves peak memory on a ~1GB CSV.
USECOLS = ["restaurant_name", "rating_review", "review_full", "date", "url_restaurant", "author_id"]

_ID_RE = re.compile(r"-(d\d+)-")


def extract_business_id(url) -> str | None:
    # Missing urls arrive as float NaN, which is truthy -- `url or ""` would
    # pass it straight into the regex and raise.
    if not isinstance(url, str):
        return None
    m = _ID_RE.search(url)
    return m.group(1) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(RAW / "London_reviews.csv"))
    p.add_argument("--k-user", type=int, default=5)
    p.add_argument("--k-item", type=int, default=5)
    p.add_argument("--city", default="London")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.csv}")
    df = pd.read_csv(args.csv, usecols=USECOLS)
    print(f"{len(df):,} raw rows")

    df["business_id"] = df.url_restaurant.map(extract_business_id)
    n_bad_id = int(df.business_id.isna().sum())
    if n_bad_id:
        print(f"dropping {n_bad_id:,} rows whose URL had no -dNNNN- restaurant id")
        df = df[df.business_id.notna()]

    # "September 23, 2020". Anything unparseable is dropped rather than imputed:
    # the train/val/test split is temporal, so a wrong date silently moves a
    # review into the wrong split.
    df["date"] = pd.to_datetime(df.date, format="%B %d, %Y", errors="coerce")
    n_bad_date = int(df.date.isna().sum())
    if n_bad_date:
        print(f"dropping {n_bad_date:,} rows with an unparseable date")
        df = df[df.date.notna()]

    df = df.rename(columns={"author_id": "user_id", "rating_review": "stars", "review_full": "text"})
    df["stars"] = pd.to_numeric(df.stars, errors="coerce")
    df = df[df.stars.notna() & df.text.notna()]

    reviews = df[["user_id", "business_id", "stars", "date", "text"]].copy()
    reviews["stars"] = reviews.stars.astype("float32")
    print(f"{len(reviews):,} reviews before k-core")

    reviews = k_core(reviews, args.k_user, args.k_item)
    n_users, n_rest = reviews.user_id.nunique(), reviews.business_id.nunique()
    print(
        f"{len(reviews):,} reviews after k-core | {n_users:,} users | "
        f"{n_rest:,} restaurants | density "
        f"{len(reviews) / (n_users * n_rest) * 100:.4f}%"
    )

    # Restaurant-level aggregates the Yelp source provides directly; here they
    # are computed from the reviews themselves, which is the same quantity.
    names = df.groupby("business_id").restaurant_name.first()
    agg = reviews.groupby("business_id").stars.agg(["mean", "count"])
    biz = pd.DataFrame(
        {
            "business_id": agg.index,
            "name": names.reindex(agg.index).str.replace("_", " ").values,
            "city": args.city,
            "state": "England",
            # absent from this source -- see module docstring
            "latitude": pd.NA,
            "longitude": pd.NA,
            "stars": agg["mean"].astype("float32").values,
            "review_count": agg["count"].astype("int64").values,
            "categories": pd.NA,
            "price": pd.NA,
        }
    )

    biz.to_parquet(OUT / "businesses.parquet", index=False)
    reviews.to_parquet(OUT / "reviews.parquet", index=False)
    print(f"wrote {OUT}/businesses.parquet ({len(biz):,}), {OUT}/reviews.parquet ({len(reviews):,})")
    print("\nNOTE: categories/price/latitude/longitude are null for this source.")
    print("features.py will drop the tag, price and geo branches accordingly.")


if __name__ == "__main__":
    main()
