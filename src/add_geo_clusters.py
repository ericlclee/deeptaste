"""Replace the 2-dim geo features with a 5-dim cluster-aware encoding.

k-means (k=40, ~neighborhood granularity for Philadelphia) over restaurant
locations projected to km. Per restaurant:

    [lat_z, lng_z, dist_center_z, dist_cluster_z, log_cluster_size_z]

- lat/lng keep the raw directional axes (north-vs-south still means something)
- dist_center: km from the catalog centroid (downtown-vs-suburb axis)
- dist_cluster: km from the nearest cluster centroid ("how far off a hub")
- log_cluster_size: how big that hub is (Center City vs a corner cluster)

Portable: centroids and normalization stats are frozen into norm_stats.json, so
any new restaurant's features are computable from lat/lng alone. Cluster sizes
describe the Yelp catalog's density, which serves as a fixed map of the city.
Degrees are projected to km before clustering (at 40N, 1 deg lng is ~85 km vs
111 km per deg lat -- raw-degree k-means would stretch clusters east-west).
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

OUT = Path("data/processed")
K = 40


def main():
    d = torch.load(OUT / "features.pt", weights_only=False)
    if d["geo"].shape[1] == 5:
        print("geo already (N, 5); nothing to do")
        return

    ll = d["latlng"].numpy().astype(np.float64)
    lat0, lng0 = float(ll[:, 0].mean()), float(ll[:, 1].mean())
    kx = 111.32
    ky = 111.32 * np.cos(np.radians(lat0))
    xy = np.stack([(ll[:, 0] - lat0) * kx, (ll[:, 1] - lng0) * ky], 1)

    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(xy)
    sizes = np.bincount(km.labels_)

    dist_center = np.linalg.norm(xy, axis=1)
    dist_cluster = np.linalg.norm(xy - km.cluster_centers_[km.labels_], axis=1)
    log_size = np.log1p(sizes[km.labels_].astype(np.float64))

    cols, stats = [], {}
    for name, x in [
        ("lat", ll[:, 0]), ("lng", ll[:, 1]),
        ("dist_center_km", dist_center),
        ("dist_cluster_km", dist_cluster),
        ("log_cluster_size", log_size),
    ]:
        mu, sd = float(x.mean()), float(x.std())
        cols.append((x - mu) / sd)
        stats[name] = [round(mu, 5), round(sd, 5)]

    d["geo"] = torch.tensor(np.stack(cols, 1), dtype=torch.float32)
    torch.save(d, OUT / "features.pt")

    ns = json.loads((OUT / "norm_stats.json").read_text())
    ns["geo"] = stats
    ns["geo_clusters"] = {
        "k": K,
        "projection": {"lat0": lat0, "lng0": lng0, "km_per_deg_lat": kx, "km_per_deg_lng": round(ky, 5)},
        "centroids_km": [[round(a, 4), round(b, 4)] for a, b in km.cluster_centers_],
        "sizes": sizes.tolist(),
    }
    (OUT / "norm_stats.json").write_text(json.dumps(ns, indent=2))

    print(f"geo now {tuple(d['geo'].shape)}")
    print(f"dist_center km: median {np.median(dist_center):.2f}  p90 {np.percentile(dist_center, 90):.2f}")
    print(f"dist_cluster km: median {np.median(dist_cluster):.2f}  p90 {np.percentile(dist_cluster, 90):.2f}")
    print(f"cluster sizes: min {sizes.min()}  median {int(np.median(sizes))}  max {sizes.max()}")


if __name__ == "__main__":
    main()
