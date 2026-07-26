"""Map Apify Google Maps scraper output into this project's feature contract.

The contract in features.py is deliberately the intersection of what any
restaurant source provides, so that a model trained on Yelp can score a
restaurant scraped from Google. This module is the test of that claim: it takes
a place record as returned by Apify's Google Maps actors and produces exactly
the tensors RestaurantEncoder.embed_new() consumes.

Field mapping (Apify `compass/crawler-google-places` + `compass/
google-maps-reviews-scraper`):

    ours            Apify                       notes
    ----            -----                       -----
    name            title                       direct
    tag_texts       categories[]                direct
    price           price                       "$$" or "$10-20"; parsed below
    lat, lng        location.lat/.lng           direct
    rating          totalScore                  direct
    n_reviews       reviewsCount                direct
    rating_std      reviewsDistribution         computed from the star histogram
    review text     reviews[].text              fed to the ABSA model
    review date     reviews[].publishedAtDate   recency weights
    (user identity) reviews[].reviewerId        NOT usable -- see below

Two things do not carry over, and both are properties of the data rather than
bugs to fix:

1. Google reviewer ids live in a different namespace from Yelp user ids with no
   overlap, so scraped reviews cannot extend the interaction graph the model
   trains on. Google data can add ITEMS (scored by content, cold-start) but not
   USERS. Building a collaborative signal from Google would mean scraping enough
   reviews to construct a separate user graph and retraining.

2. A scraped restaurant has no id embedding or popularity bias (it was not in
   the catalog at training time), so it is ranked on taste match alone. It is
   structurally at a disadvantage against catalog restaurants that have accrued
   a positive bias -- worth knowing before mixing the two in one ranked list.

Note on rating_std: features.py's docstring calls it a Yelp-only signal because
the Google Places *API* does not return a rating distribution. The scraper does
(reviewsDistribution), so the contract is portable after all -- the constraint
was the API, not the source.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import torch

# Google's price field appears either as repeated currency symbols ("$$") or,
# in newer scrapes, as a spend range ("$10-20", "$30–50"). Thresholds are the
# conventional casual/mid/upscale/fine-dining splits; they only need to be
# monotonic and stable, since the tier is consumed as a one-hot category.
_RANGE_BOUNDS = [(10, 1), (25, 2), (50, 3)]


def parse_price(raw) -> int:
    """-> tier 1-4, or 0 for 'unknown' (the same sentinel features.py uses)."""
    if not raw or not isinstance(raw, str):
        return 0
    symbols = re.fullmatch(r"\s*([$€£¥]{1,4})\s*", raw)
    if symbols:
        return len(symbols.group(1))
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", raw)]
    if not nums:
        return 0
    midpoint = sum(nums) / len(nums)
    for bound, tier in _RANGE_BOUNDS:
        if midpoint < bound:
            return tier
    return 4


def rating_std_from_distribution(dist: dict) -> float | None:
    """Sample std-dev of individual review stars, from Apify's star histogram.

    features.py computes this directly from per-review stars; the histogram is
    sufficient for the same quantity, so the two sources agree in meaning.
    Returns None when fewer than 2 reviews leave the std undefined -- callers
    should fall back to the population median exactly as features.py does.
    """
    if not dist:
        return None
    counts = [
        float(dist.get(k, 0) or 0)
        for k in ("oneStar", "twoStar", "threeStar", "fourStar", "fiveStar")
    ]
    n = sum(counts)
    if n < 2:
        return None
    mean = sum((i + 1) * c for i, c in enumerate(counts)) / n
    var = sum(c * ((i + 1) - mean) ** 2 for i, c in enumerate(counts)) / (n - 1)
    return math.sqrt(max(var, 0.0))


def _z(x: float, stats, key: str) -> float:
    mu, sd = stats[key]
    return (x - mu) / (sd if abs(sd) > 1e-8 else 1.0)


def geo_features(lat: float, lng: float, stats: dict) -> np.ndarray:
    """The same 5 geo columns features.py builds, reconstructed for a restaurant
    that was not present when KMeans was fit -- it is assigned to the nearest
    saved centroid instead of refitting."""
    centers = np.asarray(stats["geo_cluster_centers"], dtype=np.float64)
    sizes = np.asarray(stats["geo_cluster_sizes"], dtype=np.float64)
    pt = np.array([lat, lng], dtype=np.float64)

    d = np.linalg.norm(centers - pt, axis=1)
    nearest = int(d.argmin())

    return np.array(
        [
            _z(lat, stats, "lat"),
            _z(lng, stats, "lng"),
            _z(float(np.linalg.norm(pt - np.asarray(stats["city_center"]))), stats, "dist_center"),
            _z(float(d[nearest]), stats, "dist_cluster"),
            _z(float(np.log1p(sizes[nearest])), stats, "log_cluster_size"),
        ],
        dtype=np.float32,
    )


def pool_absa(
    per_review_scores: np.ndarray,  # (n_reviews, n_aspects * n_labels)
    age_years: np.ndarray,  # (n_reviews,)
    stats: dict,
) -> np.ndarray:
    """Recency-weight and shrink toward the catalog prior, matching features.py.

    Uses the tau/kappa/prior recorded in norm_stats.json rather than recomputing
    them, so a scraped restaurant is pooled on exactly the same terms as the
    training catalog. A place with two reviews lands near the prior, which is
    the intended behaviour, not a degenerate case.
    """
    prior = np.asarray(stats["absa_prior"], dtype=np.float32)
    kappa = float(stats.get("absa_kappa", 2.0))
    if len(per_review_scores) == 0:
        return prior
    w = np.exp(-age_years / float(stats["recency_tau_years"])).astype(np.float32)
    return ((per_review_scores * w[:, None]).sum(0) + kappa * prior) / (w.sum() + kappa)


def adapt_place(
    place: dict,
    stats: dict,
    name_emb: np.ndarray,
    tag_emb: np.ndarray,
    absa_per_review: np.ndarray,
    review_ages: np.ndarray,
    rating_std_fallback: float,
) -> dict:
    """One Apify place record -> the kwargs of RestaurantEncoder.embed_new().

    The three embedding inputs are passed in rather than computed here so this
    module stays free of model dependencies: name_emb/tag_emb come from the same
    sentence-transformer features.py uses, absa_per_review from the same ABSA
    model src/absa_tag_reviews.py uses.
    """
    loc = place.get("location") or {}
    lat, lng = float(loc.get("lat", 0.0)), float(loc.get("lng", 0.0))

    n_reviews = float(place.get("reviewsCount") or 0)
    rating = float(place.get("totalScore") or 0.0)
    n_tags = len(place.get("categories") or [])
    std = rating_std_from_distribution(place.get("reviewsDistribution"))

    numeric = np.array(
        [
            _z(rating, stats, "rating"),
            _z(float(np.log1p(n_reviews)), stats, "log_review_count"),
            _z(float(n_tags), stats, "tag_count"),
            _z(std if std is not None else rating_std_fallback, stats, "rating_std"),
        ],
        dtype=np.float32,
    )

    return {
        "name_emb": torch.from_numpy(np.atleast_2d(name_emb).astype(np.float32)),
        "tag_vec": torch.from_numpy(np.atleast_2d(tag_emb).astype(np.float32)),
        "absa": torch.from_numpy(
            np.atleast_2d(pool_absa(absa_per_review, review_ages, stats))
        ),
        "price": torch.tensor([parse_price(place.get("price"))], dtype=torch.long),
        "numeric": torch.from_numpy(numeric).unsqueeze(0),
        "geo": torch.from_numpy(geo_features(lat, lng, stats)).unsqueeze(0),
    }


def load_stats(path) -> dict:
    stats = json.loads(Path(path).read_text())
    missing = [
        k
        for k in ("geo_cluster_sizes", "absa_prior", "rating_std")
        if k not in stats
    ]
    if missing:
        raise KeyError(
            f"norm_stats.json is missing {missing}, so a restaurant outside the "
            "catalog cannot be featurized. Rebuild features.py to regenerate it."
        )
    return stats
