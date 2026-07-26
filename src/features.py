"""Build source-agnostic restaurant features.

Feature contract — every field here must be derivable from any restaurant source
(Yelp today, Google Places later), so this is deliberately the *intersection* of
what sources provide, not everything Yelp happens to offer:

    tag_texts     list[str]   category names, encoded as text (not vocab indices)
    price         1-4         Yelp RestaurantsPriceRange2 / Google price_level
    lat, lng      float
    rating        1-5         z-scored with per-source stats
    n_reviews     int         log1p then z-scored, per-source stats
    absa_scores   float       per-review food/service/price/ambience sentiment
                              (src/absa_tag_reviews.py), pooled with exponential
                              recency weights

rating_std (per-restaurant std-dev of individual review stars, computed from
train reviews only -- see "numerics" below) was previously documented here as
a Yelp-only exception, on the grounds that Google Places doesn't return a
rating distribution. That is true of the Places *API* but not of the source:
Apify's Google Maps scraper returns reviewsDistribution (a 1-5 star
histogram), which yields the identical quantity. So the contract is
portable after all; the constraint was the API, not Google.

Tags are encoded by running the tag *name* through the sentence encoder rather
than a learned nn.Embedding, so an unseen vocabulary ("chinese_restaurant")
lands near a known one ("Szechuan") with no retraining. tag_ids/tag_vecs below
are a compression device, not vocab lock-in: any new tag can be encoded at
serve time.

Review-derived features (absa_scores) are recency-weighted (w = exp(-age_years
/ tau)) rather than depth-capped. Effective pool depth then self-adjusts -- a
busy restaurant is described by its recent reviews, a quiet one reaches
further back -- which matches what a live source returns without committing
to a fixed depth.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

OUT = Path(os.environ.get("DEEP_TASTE_DATA", "data/yelp_philadelphia"))
# 512-token ctx fits 97.4% of reviews whole (MiniLM's 256 fit 83.4%). Swap via
# DEEP_TASTE_SBERT to try another encoder without editing source; the chosen name
# is recorded in norm_stats.json so features can be traced back to their model.
MODEL = os.environ.get("DEEP_TASTE_SBERT", "thenlper/gte-base")


def parse_tags(cats: str) -> list[str]:
    if not cats:
        return []
    drop = {"Restaurants", "Food"}
    return [t.strip() for t in cats.split(",") if t.strip() and t.strip() not in drop]


def zscore(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu, sd = float(x.mean()), float(x.std())
    sd = sd if sd > 1e-8 else 1.0
    return (x - mu) / sd, mu, sd


def main():
    global OUT

    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, default=2.0, help="recency half-life in years")
    p.add_argument(
        "--absa-kappa",
        type=float,
        default=2.0,
        help="shrinkage strength for pooled ABSA scores, in 'reviews worth of "
        "prior'. Roughly the median effective pool depth, so a typical "
        "restaurant is a 50/50 blend of its own reviews and the catalog mean; "
        "0 disables shrinkage",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--n-geo-clusters",
        type=int,
        default=25,
        help="KMeans neighborhood clusters over restaurant lat/lng, for the "
        "dist_cluster/log_cluster_size geo features",
    )
    p.add_argument("--data-dir", default=str(OUT))
    p.add_argument("--source", default="yelp", help="recorded in norm_stats.json for provenance")
    p.add_argument("--fp16", action="store_true", help="half precision; ~2x on CUDA, unsupported on MPS")
    args = p.parse_args()

    OUT = Path(args.data_dir)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")

    biz = pd.read_parquet(OUT / "businesses.parquet")
    reviews = pd.read_parquet(OUT / "reviews_split.parquet")
    train = reviews[reviews.split == "train"].reset_index(drop=True)

    biz = biz[biz.business_id.isin(reviews.business_id.unique())].reset_index(drop=True)
    n = len(biz)
    idx = {b: i for i, b in enumerate(biz.business_id)}
    print(f"{n:,} restaurants | {len(train):,} train reviews")

    sbert = SentenceTransformer(MODEL, device=device)
    if args.fp16:
        sbert = sbert.half()
    dim = sbert.get_sentence_embedding_dimension()

    # Which content modalities this SOURCE provides. Yelp gives all three;
    # the TripAdvisor London set (src/ingest/london.py) has none of them, so
    # the corresponding branches are omitted from features.pt entirely and
    # RestaurantEncoder builds without them. Emitting placeholder columns
    # instead would hand the model constant inputs to overfit through.
    has_tags = "categories" in biz and biz.categories.notna().any()
    has_price = "price" in biz and biz.price.notna().any()
    has_geo = (
        "latitude" in biz and biz.latitude.notna().any() and biz.longitude.notna().any()
    )
    print(f"modalities present: tags={has_tags} price={has_price} geo={has_geo}")

    # ---- tags
    if has_tags:
        tag_lists = [parse_tags(c) for c in biz.categories]
        vocab = sorted({t for tags in tag_lists for t in tags})
        tag_to_id = {t: i + 1 for i, t in enumerate(vocab)}  # 0 = padding
        print(f"{len(vocab):,} unique tags")

        tag_vecs = np.zeros((len(vocab) + 1, dim), dtype=np.float32)
        tag_vecs[1:] = sbert.encode(vocab, batch_size=args.batch_size, show_progress_bar=True)

        t_max = max((len(t) for t in tag_lists), default=1)
        tag_ids = np.zeros((n, t_max), dtype=np.int64)
        tag_mask = np.zeros((n, t_max), dtype=bool)
        for i, tags in enumerate(tag_lists):
            for j, t in enumerate(tags):
                tag_ids[i, j] = tag_to_id[t]
                tag_mask[i, j] = True
        print(f"tag matrix: {tag_ids.shape}")

    # ---- name: same encoder as tags/reviews, one vector per restaurant name --
    # gives the encoder a name-based prior (e.g. "Taco" in the name) independent
    # of the categories the source assigned it. Available from every source, and
    # the only wide content branch London has besides the aspect scores.
    name_emb = sbert.encode(
        list(biz.name), batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)

    # ---- reviews: load precomputed per-review ABSA aspect scores
    # (src/absa_tag_reviews.py, run separately on a GPU node -- scores every
    # review against food/service/price/ambience with a pretrained ABSA
    # model) and pool them with the same exponential recency weights used
    # everywhere else in this file, instead of re-encoding review text with
    # a sentence-transformer here. Replaces the old text_emb branch.
    absa_path = OUT / "absa_scores.pt"
    if not absa_path.exists():
        raise SystemExit(
            f"{absa_path} not found. Per-review ABSA scoring has to run before "
            "features are built:\n"
            f"    DEEP_TASTE_DATA={OUT} sbatch -p ice-gpu scripts/run_absa.sh\n"
            "Pipeline order is prepare -> split -> run_absa -> features -> train."
        )
    absa = torch.load(absa_path, weights_only=False)
    assert list(absa["business_id"]) == list(reviews.business_id), (
        "absa_scores.pt review order doesn't match reviews_split.parquet -- "
        "re-run src/absa_tag_reviews.py against the current reviews_split.parquet"
    )
    absa_aspects = absa["aspects"]
    absa_labels = absa["labels"]
    train_mask = (reviews.split == "train").to_numpy()
    absa_train = absa["scores"].float()[train_mask]  # (n_train, n_aspects, n_labels)

    now = reviews.date.max()
    age_years = ((now - train.date).dt.total_seconds() / (365.25 * 86400)).to_numpy(dtype=np.float32)
    w = np.exp(-age_years / args.tau)
    owner = train.business_id.map(idx).to_numpy(dtype=np.int64)

    W = torch.from_numpy(w)
    O = torch.from_numpy(owner)
    num_absa = torch.zeros(n, len(absa_aspects), len(absa_labels)).index_add_(0, O, absa_train * W[:, None, None])
    den = torch.zeros(n).index_add_(0, O, W)

    # Shrink toward the catalog-wide profile. Median effective depth is only ~2
    # reviews, so without this, half the catalog's aspect scores are decided by
    # one or two recent reviews -- variance dressed up as signal. kappa is "how
    # many reviews' worth of prior to blend in": sum(w) >> kappa keeps a
    # restaurant's own profile, sum(w) << kappa falls back toward the average
    # restaurant. Also subsumes the no-train-reviews case, which lands exactly
    # on the prior instead of needing a separate imputation branch.
    prior = (absa_train * W[:, None, None]).sum(0) / W.sum().clamp(min=1e-8)
    absa_pooled = (num_absa + args.absa_kappa * prior) / (den + args.absa_kappa)[:, None, None]

    n_missing = int((den.numpy() <= 1e-8).sum())
    absa_scores = absa_pooled.reshape(n, -1).numpy()  # flatten (n_aspects, n_labels) -> one vector per restaurant
    print(f"pooled with tau={args.tau}y kappa={args.absa_kappa} | effective depth (sum w): median {np.median(den.numpy()):.1f}")
    print(f"absa scores: {absa_scores.shape} ({absa_aspects} x {absa_labels}) | {n_missing} fell back to the prior (no train reviews)")

    # Redundancy check. Aspect sentiment co-moves hard with overall review
    # valence, and mean star rating is ALREADY a feature (numeric[:, 0]) -- so
    # if these 12 dims are just a re-encoding of the rating, the ABSA branch is
    # spending parameters to say something the model already knows. Cheap to
    # print, and the answer decides whether the aspect signal is worth building
    # on or needs a different extraction.
    stars_np = biz.stars.to_numpy(dtype=np.float32)
    r = np.abs([np.corrcoef(absa_scores[:, j], stars_np)[0, 1] for j in range(absa_scores.shape[1])])
    sv = np.linalg.svd(absa_scores - absa_scores.mean(0), compute_uv=False)
    n90 = int(np.searchsorted((sv**2 / (sv**2).sum()).cumsum(), 0.90) + 1)
    print(f"absa vs. mean rating |r|: max {r.max():.3f} median {np.median(r):.3f} "
          f"| {n90}/{absa_scores.shape[1]} dims carry 90% of variance")

    # ---- price
    if has_price:
        price_raw = pd.to_numeric(biz.price, errors="coerce").to_numpy(dtype=np.float32)
        price_mask = ~np.isnan(price_raw)
        price = np.nan_to_num(price_raw, nan=0.0)
        print(f"price present for {price_mask.sum():,}/{n:,}")

    # ---- numerics
    rating_z, r_mu, r_sd = zscore(biz.stars.to_numpy(dtype=np.float32))
    count_z, c_mu, c_sd = zscore(np.log1p(biz.review_count.to_numpy(dtype=np.float32)))
    if has_tags:
        tag_count_z, t_mu, t_sd = zscore(np.array([len(t) for t in tag_lists], dtype=np.float32))

    # rating_std: within-restaurant disagreement, from TRAIN reviews only (same
    # leakage boundary as the ABSA pooling above). A restaurant with <2 train reviews
    # has an undefined std -- fill with the population median (neutral), not 0,
    # which would falsely assert "perfectly consistent" for a restaurant we
    # simply have no variance evidence for.
    rating_std_raw = train.groupby("business_id").stars.std().reindex(biz.business_id)
    n_missing_std = int(rating_std_raw.isna().sum())
    rating_std_raw = rating_std_raw.fillna(rating_std_raw.median()).to_numpy(dtype=np.float32)
    print(f"rating_std: {n_missing_std} restaurants median-filled (fewer than 2 train reviews)")
    rating_std_z, rs_mu, rs_sd = zscore(rating_std_raw)

    if has_geo:
        lat = biz.latitude.to_numpy(dtype=np.float32)
        lng = biz.longitude.to_numpy(dtype=np.float32)
        lat_z, lat_mu, lat_sd = zscore(lat)
        lng_z, lng_mu, lng_sd = zscore(lng)

        # ---- geo clusters: cheap "which neighborhood" signal on top of raw lat/lng.
        # KMeans on a single metro's coordinates (src/ingest/yelp.py already filters to
        # one city) -- plain Euclidean in degree-space, not haversine, consistent
        # with lat/lng elsewhere in this file; fine at metro scale.
        latlng_raw = np.stack([lat, lng], 1)
        kmeans = KMeans(n_clusters=args.n_geo_clusters, random_state=0, n_init=10)
        cluster_id = kmeans.fit_predict(latlng_raw)
        centers = kmeans.cluster_centers_

        city_center = latlng_raw.mean(axis=0)
        dist_center = np.linalg.norm(latlng_raw - city_center, axis=1).astype(np.float32)
        dist_cluster = np.linalg.norm(latlng_raw - centers[cluster_id], axis=1).astype(np.float32)
        cluster_counts = np.bincount(cluster_id, minlength=args.n_geo_clusters)
        log_cluster_size = np.log1p(cluster_counts[cluster_id]).astype(np.float32)

        dist_center_z, dc_mu, dc_sd = zscore(dist_center)
        dist_cluster_z, dk_mu, dk_sd = zscore(dist_cluster)
        log_cluster_size_z, ls_mu, ls_sd = zscore(log_cluster_size)

    stats = {
        "source": args.source,
        "modalities": {"tags": has_tags, "price": has_price, "geo": has_geo},
        "rating": [r_mu, r_sd],
        "log_review_count": [c_mu, c_sd],
        "rating_std": [rs_mu, rs_sd],
        # catalog-wide ABSA profile: the shrinkage target. A newly-added
        # restaurant with 3 reviews needs this to be pooled the same way the
        # training catalog was, so it must travel with the stats, not be
        # recomputed from whatever reviews happen to be on hand.
        "absa_prior": prior.reshape(-1).tolist(),
        "absa_kappa": args.absa_kappa,
        "recency_tau_years": args.tau,
        "reference_date": str(now.date()),
        "sbert_model": MODEL,
        "sbert_dim": dim,
        "absa_aspects": absa_aspects,
        "absa_labels": absa_labels,
    }
    if has_tags:
        stats["tag_count"] = [t_mu, t_sd]
    if has_geo:
        stats.update(
            {
                "lat": [lat_mu, lat_sd],
                "lng": [lng_mu, lng_sd],
                "dist_center": [dc_mu, dc_sd],
                "dist_cluster": [dk_mu, dk_sd],
                "log_cluster_size": [ls_mu, ls_sd],
                "n_geo_clusters": args.n_geo_clusters,
                "city_center": city_center.tolist(),
                # cluster centers, so a new restaurant at serve time is assigned to
                # the nearest existing cluster rather than refitting KMeans on one
                # point -- and their sizes, without which log_cluster_size is not
                # reconstructible for a restaurant absent at build time.
                "geo_cluster_centers": centers.tolist(),
                "geo_cluster_sizes": cluster_counts.tolist(),
            }
        )

    numeric_cols = [rating_z, count_z] + ([tag_count_z] if has_tags else []) + [rating_std_z]
    out = {
        "business_ids": list(biz.business_id),
        "names": list(biz.name),
        "absa_scores": torch.from_numpy(absa_scores),
        "name_emb": torch.from_numpy(name_emb),
        "numeric": torch.from_numpy(np.stack(numeric_cols, 1)),
    }
    if has_tags:
        out.update(
            {
                "tag_vocab": vocab,
                "tag_vecs": torch.from_numpy(tag_vecs),
                "tag_ids": torch.from_numpy(tag_ids),
                "tag_mask": torch.from_numpy(tag_mask),
            }
        )
    if has_price:
        out.update(
            {"price": torch.from_numpy(price), "price_mask": torch.from_numpy(price_mask)}
        )
    if has_geo:
        out.update(
            {
                "geo": torch.from_numpy(
                    np.stack([lat_z, lng_z, dist_center_z, dist_cluster_z, log_cluster_size_z], 1)
                ),
                "latlng": torch.from_numpy(np.stack([lat, lng], 1)),
            }
        )

    torch.save(out, OUT / "features.pt")
    (OUT / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {OUT}/features.pt, {OUT}/norm_stats.json")


if __name__ == "__main__":
    main()
