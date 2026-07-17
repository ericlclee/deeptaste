"""Restaurant encoder (the trainable tower).

Consumes the precomputed feature contract in data/processed/features.pt and maps
each restaurant to a single L2-normalized embedding. The frozen inputs (SBERT
vectors, z-scored numerics) live as buffers; the only trainable parts are the
per-branch projections and the fusion MLP.

Fill in every `Q:` — those are the modeling decisions. The plumbing (loading,
buffers, the forward's gather step) is done.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path("data/processed")


class RestaurantEncoder(nn.Module):
    tag_vecs: torch.Tensor
    tag_ids: torch.Tensor
    tag_mask: torch.Tensor
    text_emb: torch.Tensor
    name_emb: torch.Tensor
    price: torch.Tensor
    numeric: torch.Tensor
    geo: torch.Tensor

    def __init__(
        self,
        features: dict,
        output_dims: int = 128,
        branch_dims: int = 128,
        hidden_dims: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        # --- frozen inputs, registered as buffers so .to(device) moves them and
        #     they are saved with the model but never receive gradients ---
        self.register_buffer("tag_vecs", features["tag_vecs"])
        self.register_buffer("tag_ids", features["tag_ids"])  # (N, T_max) int
        self.register_buffer("tag_mask", features["tag_mask"])  # (N, T_max) bool
        self.register_buffer("text_emb", features["text_emb"])  # (N, 768)
        self.register_buffer("name_emb", features["name_emb"])  # (N, 768)
        self.register_buffer(
            "price", features["price"].long()
        )  # (N,) tiers 0..4, 0 = missing
        self.register_buffer(
            "numeric", features["numeric"]
        )  # (N, 4) z-scored: rating, log_count, tag_count, rating_std
        self.register_buffer(
            "geo", features["geo"]
        )  # (N, 5) z-scored: lat, lng, dist_center, dist_cluster, log_cluster_size
        self.n_restaurants = self.text_emb.shape[0]
        self.output_dims = output_dims
        self.branch_dims = branch_dims
        self.hidden_dims = hidden_dims

        # --- per-branch projections (the "don't pre-average, project then concat"
        #     principle) ---
        # Q: name, text, and pooled-tag vectors are each (·, 768). Do they share
        #    one projection or get three separate Linears? (Which choice lets the
        #    model weight them independently — the reason we concatenate?)
        # Q: what output width should each branch project to? (Anything; the
        #    fusion MLP consumes the concatenation.)
        sbert_dims = self.text_emb.shape[1]
        self.name_proj = nn.Linear(sbert_dims, branch_dims)
        self.text_proj = nn.Linear(sbert_dims, branch_dims)
        self.tag_proj = nn.Linear(sbert_dims, branch_dims)

        # --- fusion ---
        # Q: what is the width of the concatenated vector that feeds the MLP?
        #    Sum the branch outputs + price one-hot (5) + numeric + geo. Read the
        #    numeric/geo widths from the buffers (their shapes have grown twice
        #    today: numeric 3->4 with rating_std, geo 2->5 with clusters) --
        #    hardcoded literals go stale, self.numeric.shape[1] never does.
        # Q: build the fusion MLP. The spec calls for 2-3 Linear layers with ReLU
        #    and dropout, ending in `dim`. Sketch it as an nn.Sequential.

        self.price_dims = int(self.price.max()) + 1  # 4 prices + no price category
        mlp_input_dims = (
            branch_dims * 3  # tags + reviews + name
            + self.price_dims
            + self.numeric.shape[1]
            + self.geo.shape[1]
        )

        self.fusion = nn.Sequential(
            nn.Linear(mlp_input_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, output_dims),
        )

    def _pool_tags(self, idx: torch.Tensor) -> torch.Tensor:
        """Masked mean of a restaurant's tag vectors. Returns (B, 768)."""
        ids = self.tag_ids[idx]  # (B, T_max)
        mask = self.tag_mask[idx]  # (B, T_max)
        vecs = self.tag_vecs[ids]  # (B, T_max, 768)

        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        vecs = vecs.sum(dim=1) / count
        return vecs

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B,) restaurant indices. Returns (B, dim) L2-normalized embeddings."""
        name = self.name_emb[idx]  # (B, 768)
        text = self.text_emb[idx]  # (B, 768)
        tag = self._pool_tags(idx)  # (B, 768)
        price = self.price[idx]  # (B,)
        numeric = self.numeric[idx]  # (B, 4)
        geo = self.geo[idx]  # (B, 5)

        price_oh = F.one_hot(price, num_classes=self.price_dims).float()

        name = self.name_proj(name)
        text = self.text_proj(text)
        tag = self.tag_proj(tag)
        x = torch.cat([name, text, tag, price_oh, numeric, geo], dim=1)
        z = self.fusion(x)
        z = F.normalize(z, dim=1)

        return z


def load_encoder(dim: int = 128, device: str | None = None) -> RestaurantEncoder:
    """Convenience: build an encoder from the on-disk features."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    features = torch.load(OUT / "features.pt", weights_only=False)
    return RestaurantEncoder(features).to(device)


if __name__ == "__main__":
    enc = load_encoder()
    idx = torch.arange(4, device=next(enc.parameters()).device)
    out = enc(idx)
    print(out.shape)
    print(out.norm(dim=1))
