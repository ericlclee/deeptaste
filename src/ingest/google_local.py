"""Ingest the UCSD Google Local dataset (McAuley lab) into the project contract.

    https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/

Two newline-delimited JSON files per US state:

    meta-<State>.json     name, address, gmap_id, latitude, longitude,
                          category[], avg_rating, num_of_reviews, price, ...
    review-<State>.json   user_id, name, time (unix ms), rating, text, gmap_id

Unlike the Yelp and TripAdvisor sources this one is too large to load and then
filter -- review-New_York.json alone is ~10.5 GB -- so reviews are streamed and
discarded as they are read, keeping only those belonging to an in-scope
restaurant.

Scope is a lat/lng bounding box rather than a city name, because Google Local
has no city field and the addresses are free text. The default box is NYC's
five boroughs.

    python src/ingest/google_local.py                    # NYC, 10-core
    python src/ingest/google_local.py --k-user 5 --k-item 5
    python src/ingest/google_local.py --survey           # counts only, no write
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from common import k_core, report_density

RAW = Path(os.environ.get("DEEP_TASTE_RAW", "data/raw/google_local"))
OUT = Path(os.environ.get("DEEP_TASTE_DATA", "data/newyork_google"))

# NYC five boroughs: Staten Island's southern tip to the north Bronx, and
# western Staten Island to eastern Queens.
NYC_BBOX = (40.4774, 40.9176, -74.2591, -73.7004)  # lat_min, lat_max, lng_min, lng_max

# Google's category taxonomy is granular and consistently suffixed -- "Pizza
# restaurant", "Chinese restaurant", "Fast food restaurant" -- so a substring
# match on the taxonomy does the work a hand-built cuisine list would, and
# picks up cuisines nobody thought to enumerate.
DEFAULT_CATEGORY_MATCH = "restaurant"

# price is a run of currency symbols, and which symbol depends on the
# restaurant's locale ("$$" but also "₩₩" on Korean places), so the tier is the
# COUNT of symbols, not the symbols themselves.
PRICE_RE = re.compile(r"^([^\w\s]){1,4}$")


def parse_price(raw) -> float | None:
    """-> tier 1-4, or None when absent/unparseable."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    return float(len(s)) if PRICE_RE.match(s) and len(set(s)) == 1 else None


# MISC groups worth keeping. Excluded: Payments, Accessibility, Health &
# safety and Service options -- near-universal or logistical, so they say
# little about who would choose the place. Amenities is dropped as mostly
# restroom/wifi noise.
MISC_GROUPS = {
    "Atmosphere",
    "Crowd",
    "Popular for",
    "Highlights",
    "Planning",
    "Dining options",
    "Offerings",
}


def misc_attributes(misc) -> list[str]:
    """Google's own structured attributes, flattened into category-style strings.

    These cover ground the review text cannot: "Solo dining", "Accepts
    reservations", "Late-night food" and "LGBTQ friendly" are facts about the
    restaurant, not sentiment about it. They are emitted alongside the
    categories so the existing tag pipeline tokenizes and embeds them with no
    separate branch -- "Late-night food" simply becomes the tokens late, night,
    food, and PMI decides whether late_night binds.

    Caveat worth remembering: these are partly owner-supplied, so coverage
    tracks how actively a business manages its listing, and they carry no
    timestamp -- unlike reviews they cannot be held to the training side of
    the split.
    """
    if not isinstance(misc, dict):
        return []
    out = []
    for group, items in misc.items():
        if group in MISC_GROUPS:
            out.extend(i for i in (items or []) if i)
    return out


def load_restaurants(bbox, category_match: str) -> pd.DataFrame:
    """Stream the metadata file, keeping in-box places whose category matches."""
    lat_min, lat_max, lng_min, lng_max = bbox
    rows, n_total, n_box = [], 0, 0
    with (RAW / "meta-New_York.json").open() as fh:
        for line in fh:
            d = json.loads(line)
            n_total += 1
            lat, lng = d.get("latitude"), d.get("longitude")
            if lat is None or lng is None:
                continue
            if not (lat_min <= lat <= lat_max and lng_min <= lng <= lng_max):
                continue
            n_box += 1
            cats = d.get("category") or []
            if not any(category_match in c.lower() for c in cats):
                continue
            rows.append(
                {
                    "business_id": d["gmap_id"],
                    "name": d.get("name") or "",
                    "categories": ", ".join(cats + misc_attributes(d.get("MISC"))),
                    "price": parse_price(d.get("price")),
                    "latitude": lat,
                    "longitude": lng,
                    "stars": d.get("avg_rating"),
                    "review_count": d.get("num_of_reviews"),
                }
            )
    print(f"{n_total:,} places in file | {n_box:,} in bbox | {len(rows):,} matching '{category_match}'")
    return pd.DataFrame(rows)


def stream_reviews(keep_ids: set[str]) -> pd.DataFrame:
    """Single streaming pass over the review file, keeping in-scope rows only."""
    users, biz, stars, times, texts = [], [], [], [], []
    n = 0
    with (RAW / "review-New_York.json").open() as fh:
        for line in fh:
            n += 1
            if n % 5_000_000 == 0:
                print(f"  scanned {n:,} reviews, kept {len(users):,}")
            d = json.loads(line)
            g = d.get("gmap_id")
            if g not in keep_ids:
                continue
            users.append(d["user_id"])
            biz.append(g)
            stars.append(d.get("rating"))
            times.append(d.get("time"))
            texts.append(d.get("text") or "")
    print(f"  scanned {n:,} reviews total, kept {len(users):,}")
    return pd.DataFrame(
        {
            "user_id": users,
            "business_id": biz,
            "stars": pd.array(stars, dtype="float32"),
            "date": pd.to_datetime(times, unit="ms"),
            "text": texts,
        }
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k-user", type=int, default=10)
    p.add_argument("--k-item", type=int, default=10)
    p.add_argument("--category-match", default=DEFAULT_CATEGORY_MATCH)
    p.add_argument("--bbox", type=float, nargs=4, default=list(NYC_BBOX),
                   metavar=("LAT_MIN", "LAT_MAX", "LNG_MIN", "LNG_MAX"))
    p.add_argument("--survey", action="store_true",
                   help="report counts at several k-core settings and exit without writing")
    args = p.parse_args()

    biz = load_restaurants(tuple(args.bbox), args.category_match)
    if biz.empty:
        raise SystemExit("no restaurants matched -- check --bbox and --category-match")

    print(f"streaming reviews for {len(biz):,} restaurants (this reads ~10 GB)")
    reviews = stream_reviews(set(biz.business_id))
    report_density(reviews, reviews.business_id.nunique(), "before k-core")

    if args.survey:
        # k-core is destructive and the right k is a judgement call, so make the
        # tradeoff visible rather than asking anyone to guess at it.
        print(f"\n{'k':>4} {'reviews':>12} {'users':>10} {'restaurants':>12} {'density':>9}")
        for k in (0, 5, 10, 15, 20):
            r = reviews if k == 0 else k_core(reviews, k, k)
            nb, nu = r.business_id.nunique(), r.user_id.nunique()
            d = len(r) / (nu * nb) * 100 if nu and nb else 0
            print(f"{k:>4} {len(r):>12,} {nu:>10,} {nb:>12,} {d:>8.3f}%")
        return

    reviews = k_core(reviews, args.k_user, args.k_item)
    biz = biz[biz.business_id.isin(reviews.business_id.unique())].reset_index(drop=True)
    report_density(reviews, len(biz))

    OUT.mkdir(parents=True, exist_ok=True)
    biz.to_parquet(OUT / "businesses.parquet", index=False)
    reviews.to_parquet(OUT / "reviews.parquet", index=False)
    print(f"wrote {OUT}/businesses.parquet, {OUT}/reviews.parquet")


if __name__ == "__main__":
    main()
