"""Shared helpers for turning a raw source into this project's ingest contract.

Every ingest script in this directory emits the same two files into its
dataset's data directory, and nothing downstream knows which source produced
them:

    businesses.parquet   business_id, name, categories, price, latitude,
                         longitude, stars, review_count
                         (categories/price/latitude/longitude may be absent --
                         features.py drops the corresponding branch)
    reviews.parquet      user_id, business_id, stars, date, text

Anything source-specific belongs in that source's own module, not here.
"""

import pandas as pd


def k_core(reviews: pd.DataFrame, k_user: int, k_item: int) -> pd.DataFrame:
    """Iteratively drop users and items with too few interactions until stable.

    Iterative because the two conditions interact: dropping a thin restaurant
    can push one of its reviewers below k_user, which can in turn thin another
    restaurant. The loop runs to a fixed point, so every remaining user has
    >= k_user reviews AND every remaining restaurant has >= k_item, both true
    simultaneously.

    Note this makes metrics look better without the model improving -- it
    removes exactly the sparse users and items that are hardest to predict.
    Standard practice in the recsys literature, but it means numbers are only
    comparable across datasets prepared with the SAME k.
    """
    while True:
        n = len(reviews)
        uc = reviews.user_id.value_counts()
        reviews = reviews[reviews.user_id.isin(uc[uc >= k_user].index)]
        bc = reviews.business_id.value_counts()
        reviews = reviews[reviews.business_id.isin(bc[bc >= k_item].index)]
        if len(reviews) == n:
            return reviews


def report_density(reviews: pd.DataFrame, n_biz: int, label: str = "after k-core") -> None:
    n_users = reviews.user_id.nunique()
    density = len(reviews) / (n_users * n_biz) * 100 if n_users and n_biz else 0.0
    print(
        f"{len(reviews):,} reviews {label} | {n_users:,} users | "
        f"{n_biz:,} restaurants | density {density:.3f}%"
    )
