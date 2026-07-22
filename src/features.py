"""Build source-agnostic restaurant features.

Feature contract — every field here must be derivable from any restaurant source
(Yelp today, Google Places later), so this is deliberately the *intersection* of
what sources provide, not everything Yelp happens to offer:

    tag_texts     list[str]   category names, encoded as text (not vocab indices)
    price         1-4         Yelp RestaurantsPriceRange2 / Google price_level
    lat, lng      float
    rating        1-5         z-scored with per-source stats
    n_reviews     int         log1p then z-scored, per-source stats
    review_texts  list[str]   pooled with exponential recency weights

Exception to the source-agnostic rule: rating_std (per-restaurant std-dev of
individual review stars, computed from train reviews only -- see "numerics"
below). Google Places doesn't expose a rating distribution, so this is a
Yelp-only signal for now; per the project spec Google enrichment is an
optional later phase, so this is an accepted tradeoff, not an oversight. If
Google Places support is ever added, this column needs a fallback (impute
the population mean, or drop the column) for restaurants from that source.

Tags are encoded by running the tag *name* through the sentence encoder rather
than a learned nn.Embedding, so an unseen vocabulary ("chinese_restaurant")
lands near a known one ("Szechuan") with no retraining. tag_ids/tag_vecs below
are a compression device, not vocab lock-in: any new tag can be encoded at
serve time.

Review pooling is recency-weighted (w = exp(-age_years / tau)) rather than
depth-capped. Effective pool depth then self-adjusts -- a busy restaurant is
described by its recent reviews, a quiet one reaches further back -- which
matches what a live source returns without committing to a fixed depth.
Per-review vectors are retained so the fixed weights can later be swapped for
learned attention.
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

OUT = Path(os.environ.get("DEEP_TASTE_DATA", "data/processed"))
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--n-geo-clusters",
        type=int,
        default=25,
        help="KMeans neighborhood clusters over restaurant lat/lng, for the "
        "dist_cluster/log_cluster_size geo features",
    )
    p.add_argument("--data-dir", default=str(OUT))
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

    # ---- tags
    tag_lists = [parse_tags(c) for c in biz.categories]
    vocab = sorted({t for tags in tag_lists for t in tags})
    tag_to_id = {t: i + 1 for i, t in enumerate(vocab)}  # 0 = padding
    print(f"{len(vocab):,} unique tags")

    tag_vecs = np.zeros((len(vocab) + 1, dim), dtype=np.float32)
    tag_vecs[1:] = sbert.encode(vocab, batch_size=args.batch_size, show_progress_bar=True)

    # ---- name: same encoder as tags/reviews, one vector per restaurant name --
    # gives the encoder a name-based prior (e.g. "Taco" in the name) independent
    # of the categories Yelp assigned it.
    name_emb = sbert.encode(
        list(biz.name), batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)

    t_max = max((len(t) for t in tag_lists), default=1)
    tag_ids = np.zeros((n, t_max), dtype=np.int64)
    tag_mask = np.zeros((n, t_max), dtype=bool)
    for i, tags in enumerate(tag_lists):
        for j, t in enumerate(tags):
            tag_ids[i, j] = tag_to_id[t]
            tag_mask[i, j] = True
    print(f"tag matrix: {tag_ids.shape}")

    # ---- reviews: encode all, pool with recency weights
    print(f"encoding {len(train):,} reviews (this is the slow one)")
    vecs = sbert.encode(
        list(train.text), batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)

    now = reviews.date.max()
    age_years = ((now - train.date).dt.total_seconds() / (365.25 * 86400)).to_numpy(dtype=np.float32)
    w = np.exp(-age_years / args.tau)
    owner = train.business_id.map(idx).to_numpy(dtype=np.int64)

    V = torch.from_numpy(vecs)
    W = torch.from_numpy(w)
    O = torch.from_numpy(owner)
    num = torch.zeros(n, dim).index_add_(0, O, V * W[:, None])
    den = torch.zeros(n).index_add_(0, O, W)
    text_emb = (num / den.clamp(min=1e-8)[:, None]).numpy()

    text_mask = den.numpy() > 1e-8
    n_missing = int((~text_mask).sum())
    if n_missing:
        text_emb[~text_mask] = text_emb[text_mask].mean(0)
    print(f"pooled with tau={args.tau}y | effective depth (sum w): median {np.median(den.numpy()):.1f}")
    print(f"text embeddings: {text_emb.shape} | {n_missing} imputed (no train reviews)")

    # per-review archive for the learned-attention upgrade
    torch.save(
        {
            "review_vecs": torch.from_numpy(vecs).half(),
            "business_idx": O.to(torch.int32),
            "age_years": torch.from_numpy(age_years),
            "stars": torch.from_numpy(train.stars.to_numpy(dtype=np.float32)),
            "user_id": list(train.user_id),
        },
        OUT / "review_vecs.pt",
    )

    # ---- price
    price_raw = pd.to_numeric(biz.price, errors="coerce").to_numpy(dtype=np.float32)
    price_mask = ~np.isnan(price_raw)
    price = np.nan_to_num(price_raw, nan=0.0)
    print(f"price present for {price_mask.sum():,}/{n:,}")

    # ---- numerics
    rating_z, r_mu, r_sd = zscore(biz.stars.to_numpy(dtype=np.float32))
    count_z, c_mu, c_sd = zscore(np.log1p(biz.review_count.to_numpy(dtype=np.float32)))
    tag_count_z, t_mu, t_sd = zscore(np.array([len(t) for t in tag_lists], dtype=np.float32))

    # rating_std: within-restaurant disagreement, from TRAIN reviews only (same
    # leakage boundary as text_emb above). A restaurant with <2 train reviews
    # has an undefined std -- fill with the population median (neutral), not 0,
    # which would falsely assert "perfectly consistent" for a restaurant we
    # simply have no variance evidence for.
    rating_std_raw = train.groupby("business_id").stars.std().reindex(biz.business_id)
    n_missing_std = int(rating_std_raw.isna().sum())
    rating_std_raw = rating_std_raw.fillna(rating_std_raw.median()).to_numpy(dtype=np.float32)
    print(f"rating_std: {n_missing_std} restaurants median-filled (fewer than 2 train reviews)")
    rating_std_z, rs_mu, rs_sd = zscore(rating_std_raw)

    lat = biz.latitude.to_numpy(dtype=np.float32)
    lng = biz.longitude.to_numpy(dtype=np.float32)
    lat_z, lat_mu, lat_sd = zscore(lat)
    lng_z, lng_mu, lng_sd = zscore(lng)

    # ---- geo clusters: cheap "which neighborhood" signal on top of raw lat/lng.
    # KMeans on a single metro's coordinates (prepare_data.py already filters to
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
        "source": "yelp",
        "rating": [r_mu, r_sd],
        "log_review_count": [c_mu, c_sd],
        "tag_count": [t_mu, t_sd],
        "rating_std": [rs_mu, rs_sd],
        "lat": [lat_mu, lat_sd],
        "lng": [lng_mu, lng_sd],
        "dist_center": [dc_mu, dc_sd],
        "dist_cluster": [dk_mu, dk_sd],
        "log_cluster_size": [ls_mu, ls_sd],
        "n_geo_clusters": args.n_geo_clusters,
        "city_center": city_center.tolist(),
        # cluster centers, so a new restaurant at serve time is assigned to the
        # nearest existing cluster rather than refitting KMeans on one point.
        "geo_cluster_centers": centers.tolist(),
        "recency_tau_years": args.tau,
        "reference_date": str(now.date()),
        "sbert_model": MODEL,
        "sbert_dim": dim,
    }

    torch.save(
        {
            "business_ids": list(biz.business_id),
            "names": list(biz.name),
            "tag_vocab": vocab,
            "tag_vecs": torch.from_numpy(tag_vecs),
            "tag_ids": torch.from_numpy(tag_ids),
            "tag_mask": torch.from_numpy(tag_mask),
            "text_emb": torch.from_numpy(text_emb),
            "name_emb": torch.from_numpy(name_emb),
            "price": torch.from_numpy(price),
            "price_mask": torch.from_numpy(price_mask),
            "numeric": torch.from_numpy(
                np.stack([rating_z, count_z, tag_count_z, rating_std_z], 1)
            ),
            "geo": torch.from_numpy(
                np.stack([lat_z, lng_z, dist_center_z, dist_cluster_z, log_cluster_size_z], 1)
            ),
            "latlng": torch.from_numpy(np.stack([lat, lng], 1)),
        },
        OUT / "features.pt",
    )
    (OUT / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {OUT}/features.pt, {OUT}/review_vecs.pt, {OUT}/norm_stats.json")


if __name__ == "__main__":
    main()
