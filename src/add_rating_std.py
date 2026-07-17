"""Append rating_std to the numeric features in features.pt.

Per-restaurant std of individual review stars (train split only), z-scored across
restaurants. A compact proxy for the rating histogram we dropped: distinguishes
polarizing (bimodal 1s-and-5s) from consistently mediocre (peaked at 3) at the
same mean. Portable: derivable from any source that exposes per-review stars
(Apify-scraped Google does; the official Places API does not).

Restaurants with <2 train reviews have undefined std; they get the population
median (neutral) rather than 0, which would falsely assert "perfectly consistent".
"""

import json
from pathlib import Path

import pandas as pd
import torch

OUT = Path("data/processed")


def main():
    d = torch.load(OUT / "features.pt", weights_only=False)
    if d["numeric"].shape[1] == 4:
        print("rating_std already present; numeric is (N, 4). nothing to do")
        return

    r = pd.read_parquet(OUT / "reviews_split.parquet", columns=["business_id", "stars", "split"])
    std = r[r.split == "train"].groupby("business_id").stars.std()

    per_biz = std.reindex(d["business_ids"])
    n_missing = int(per_biz.isna().sum())
    per_biz = per_biz.fillna(per_biz.median())

    mu, sd = float(per_biz.mean()), float(per_biz.std())
    std_z = torch.tensor(((per_biz - mu) / sd).to_numpy(), dtype=torch.float32)

    d["numeric"] = torch.cat([d["numeric"], std_z[:, None]], dim=1)
    torch.save(d, OUT / "features.pt")

    stats = json.loads((OUT / "norm_stats.json").read_text())
    stats["rating_std"] = [round(mu, 4), round(sd, 4)]
    (OUT / "norm_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"numeric now {tuple(d['numeric'].shape)} | rating_std pop: {mu:.3f} ± {sd:.3f} "
          f"| {n_missing} median-filled (fewer than 2 train reviews)")


if __name__ == "__main__":
    main()
