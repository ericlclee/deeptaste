"""Tests for the Google/Apify -> feature-contract adapter.

Self-contained: builds a small synthetic catalog and norm_stats rather than
requiring a real features.pt, so this runs anywhere without the dataset.

The point is not that the arithmetic is right (it is checked, but that is the
easy part) -- it is whether a restaurant described by a Google Maps scrape can
be pushed through the encoder at all, and whether it lands somewhere sensible
relative to the catalog. That is the claim features.py's docstring makes about
being source-agnostic, and it has never actually been exercised.

    python src/test_adapt_google.py
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from adapt_google import (
    adapt_place,
    geo_features,
    load_stats,
    parse_price,
    pool_absa,
    rating_std_from_distribution,
)
from model import RestaurantEncoder

SBERT_DIMS, ABSA_DIMS, N_CATALOG = 32, 12, 40
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(name)


# --- a real Apify record, trimmed to the fields the adapter reads -------------
APIFY_PLACE = {
    "title": "Kensington Ramen House",
    "categories": ["Ramen restaurant", "Japanese restaurant", "Noodle shop"],
    "price": "$10–20",
    "location": {"lat": 39.9781, "lng": -75.1393},
    "totalScore": 4.4,
    "reviewsCount": 312,
    "reviewsDistribution": {
        "oneStar": 8,
        "twoStar": 11,
        "threeStar": 30,
        "fourStar": 96,
        "fiveStar": 167,
    },
    "reviews": [
        {"stars": 5, "text": "Broth is incredible.", "publishedAtDate": "2025-06-01T00:00:00Z"},
        {"stars": 4, "text": "Great noodles, slow service.", "publishedAtDate": "2024-01-15T00:00:00Z"},
    ],
}


def synthetic_stats() -> dict:
    rng = np.random.default_rng(0)
    centers = rng.normal([39.97, -75.15], [0.05, 0.05], size=(5, 2))
    return {
        "source": "yelp",
        "rating": [3.61, 0.78],
        "log_review_count": [3.90, 1.22],
        "tag_count": [3.34, 2.23],
        "rating_std": [1.05, 0.32],
        "lat": [39.973, 0.047],
        "lng": [-75.155, 0.050],
        "dist_center": [0.053, 0.043],
        "dist_cluster": [0.008, 0.006],
        "log_cluster_size": [5.83, 0.78],
        "n_geo_clusters": 5,
        "city_center": [39.9734, -75.1545],
        "geo_cluster_centers": centers.tolist(),
        "geo_cluster_sizes": [120, 340, 55, 890, 210],
        "absa_prior": (np.ones(ABSA_DIMS, dtype=np.float32) / 3).tolist(),
        "absa_kappa": 2.0,
        "recency_tau_years": 2.0,
        "sbert_dim": SBERT_DIMS,
    }


def synthetic_encoder() -> RestaurantEncoder:
    g = torch.Generator().manual_seed(0)
    feats = {
        "business_ids": [f"b{i}" for i in range(N_CATALOG)],
        "tag_vecs": torch.randn(9, SBERT_DIMS, generator=g),
        "tag_ids": torch.randint(0, 9, (N_CATALOG, 4), generator=g),
        "tag_mask": torch.ones(N_CATALOG, 4, dtype=torch.bool),
        "absa_scores": torch.rand(N_CATALOG, ABSA_DIMS, generator=g),
        "name_emb": torch.randn(N_CATALOG, SBERT_DIMS, generator=g),
        "price": torch.randint(0, 5, (N_CATALOG,), generator=g).float(),
        "numeric": torch.randn(N_CATALOG, 4, generator=g),
        "geo": torch.randn(N_CATALOG, 5, generator=g),
    }
    return RestaurantEncoder(feats, output_dims=16, branch_dims=8, hidden_dims=16).eval()


def test_price_parsing() -> None:
    print("\nprice parsing (Google ships two different formats)")
    for raw, want in [
        ("$", 1), ("$$", 2), ("$$$", 3), ("$$$$", 4),
        ("$10–20", 2),      # en-dash, as Google actually emits it
        ("$5-10", 1), ("$30–50", 3), ("$100+", 4),
        (None, 0), ("", 0), ("Moderate", 0),
    ]:
        got = parse_price(raw)
        check(f"parse_price({raw!r}) == {want}", got == want, f"got {got}")


def test_rating_std() -> None:
    print("\nrating_std from the star histogram")
    # all five stars -> zero disagreement
    check(
        "uniform 5-star distribution -> std 0",
        rating_std_from_distribution(
            {"oneStar": 0, "twoStar": 0, "threeStar": 0, "fourStar": 0, "fiveStar": 50}
        ) == 0.0,
    )
    # verify against the same quantity computed from raw per-review stars, which
    # is what features.py does -- the two paths must agree or the Google-sourced
    # column means something different from the Yelp-sourced one.
    dist = APIFY_PLACE["reviewsDistribution"]
    expanded = np.repeat([1, 2, 3, 4, 5], [dist[k] for k in
                ("oneStar", "twoStar", "threeStar", "fourStar", "fiveStar")])
    from_hist = rating_std_from_distribution(dist)
    from_raw = float(np.std(expanded, ddof=1))
    check(
        "histogram std == std of expanded per-review stars",
        math.isclose(from_hist, from_raw, rel_tol=1e-9),
        f"{from_hist:.6f} vs {from_raw:.6f}",
    )
    check("<2 reviews -> None (undefined, caller imputes)",
          rating_std_from_distribution({"fiveStar": 1}) is None)
    check("empty -> None", rating_std_from_distribution({}) is None)


def test_absa_pooling() -> None:
    print("\nABSA pooling matches features.py's shrinkage")
    stats = synthetic_stats()
    prior = np.asarray(stats["absa_prior"], dtype=np.float32)
    check(
        "no reviews -> exactly the catalog prior",
        np.allclose(pool_absa(np.zeros((0, ABSA_DIMS), np.float32), np.zeros(0, np.float32), stats), prior),
    )
    # one brand-new review, kappa=2 -> weight 1 vs prior weight 2, so the result
    # sits 1/3 of the way from prior to the observation
    obs = np.ones((1, ABSA_DIMS), dtype=np.float32)
    got = pool_absa(obs, np.zeros(1, np.float32), stats)
    check(
        "1 fresh review shrinks 1/3 toward the observation",
        np.allclose(got, prior + (obs[0] - prior) / 3, atol=1e-6),
    )
    # an old review should move the result less than a fresh one
    old = pool_absa(obs, np.array([8.0], np.float32), stats)
    check("an 8-year-old review moves it less than a fresh one",
          float(np.abs(old - prior).sum()) < float(np.abs(got - prior).sum()))


def test_geo_reconstruction() -> None:
    print("\ngeo features for a restaurant absent when KMeans was fit")
    stats = synthetic_stats()
    g = geo_features(39.9781, -75.1393, stats)
    check("returns the 5 columns the encoder expects", g.shape == (5,), str(g.shape))
    check("all finite", bool(np.isfinite(g).all()))
    # a point sitting exactly on a centroid must have dist_cluster == 0 pre-z-scoring
    c = stats["geo_cluster_centers"][3]
    on_centroid = geo_features(c[0], c[1], stats)
    expected = (0.0 - stats["dist_cluster"][0]) / stats["dist_cluster"][1]
    check("a point on a centroid gets dist_cluster == 0",
          math.isclose(float(on_centroid[3]), expected, rel_tol=1e-5))


def test_missing_stats_is_actionable() -> None:
    print("\nnorm_stats.json completeness")
    stats = synthetic_stats()
    del stats["geo_cluster_sizes"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(stats, f)
        p = f.name
    try:
        load_stats(p)
        check("missing key raises", False, "no error raised")
    except KeyError as e:
        check("missing key raises a message naming the gap", "geo_cluster_sizes" in str(e))
    finally:
        Path(p).unlink()


def test_end_to_end() -> None:
    print("\nend-to-end: Apify record -> encoder embedding")
    stats = synthetic_stats()
    enc = synthetic_encoder()
    rng = np.random.default_rng(1)

    kwargs = adapt_place(
        APIFY_PLACE,
        stats,
        name_emb=rng.normal(size=SBERT_DIMS),
        tag_emb=rng.normal(size=SBERT_DIMS),
        absa_per_review=rng.random((2, ABSA_DIMS)).astype(np.float32),
        review_ages=np.array([0.5, 2.0], dtype=np.float32),
        rating_std_fallback=1.0,
    )

    with torch.no_grad():
        emb = enc.embed_new(**kwargs)

    check("embedding has the catalog's output width", emb.shape == (1, 16), str(tuple(emb.shape)))
    check("L2-normalized like every catalog embedding",
          torch.allclose(emb.norm(dim=1), torch.ones(1), atol=1e-5))
    check("no NaN/Inf", bool(torch.isfinite(emb).all()))

    # It must be scoreable against the catalog -- the actual point of the exercise.
    with torch.no_grad():
        catalog = enc(torch.arange(N_CATALOG))
        sims = (emb @ catalog.T).squeeze(0)
    check("scores against every catalog restaurant", sims.shape == (N_CATALOG,), str(tuple(sims.shape)))
    check("similarities are valid cosines", bool((sims.abs() <= 1.0 + 1e-5).all()))
    check("does not collapse to one point", float(sims.std()) > 1e-4, f"std {sims.std():.4f}")

    # A scraped restaurant has no id embedding and no popularity bias.
    check("no popularity bias exists for an unseen restaurant",
          not hasattr(enc, "_new_bias"))
    print(f"    similarity to catalog: min {sims.min():+.3f}  max {sims.max():+.3f}  std {sims.std():.3f}")


def test_content_path_is_shared() -> None:
    """embed_new must use the SAME weights as the catalog path -- if it drifted
    into a separate code path, a scraped restaurant would be embedded by a
    different function than the one that was trained."""
    print("\nembed_new shares the trained content path")
    enc = synthetic_encoder()
    idx = torch.tensor([7])
    with torch.no_grad():
        from_catalog = enc.content_embedding(idx)
        from_features = enc.fuse(
            name_emb=enc.name_emb[idx],
            absa=enc.absa_scores[idx],
            tag_vec=enc._pool_tags(idx),
            price=enc.price[idx],
            numeric=enc.numeric[idx],
            geo=enc.geo[idx],
        )
    check("same weights, same inputs -> identical output",
          torch.allclose(from_catalog, from_features, atol=1e-6))


if __name__ == "__main__":
    test_price_parsing()
    test_rating_std()
    test_absa_pooling()
    test_geo_reconstruction()
    test_missing_stats_is_actionable()
    test_content_path_is_shared()
    test_end_to_end()

    print(f"\n{'-' * 60}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all checks passed")
