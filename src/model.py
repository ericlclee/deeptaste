"""The two towers.

RestaurantEncoder (item tower) maps each restaurant to an L2-normalized
embedding. UserProfile (user tower) pools a user's rated restaurants into an
embedding in the same space. Score = cos(user, item) + item_bias.

Item tower is HYBRID, not content-only:

    embedding = normalize( MLP(content features) + id_embedding )

The content path generalizes (a brand-new restaurant with no interactions
still gets a sensible vector from its tags/ABSA/price/geo). The id_embedding
is a free per-restaurant parameter that absorbs whatever collaborative signal
the content features cannot express -- without it, two restaurants with the
same cuisine, price, neighborhood and aspect scores are forced to identical
vectors no matter how differently users actually treat them. It is initialized
near zero so the model starts content-driven and only memorizes where the data
demands it; `--no-id-emb` ablates it to measure the split.

item_bias is a scalar per restaurant, added to the score OUTSIDE the cosine.
Both towers are L2-normalized, so without it the model has no way to express
"this place is just generally good" -- popularity/quality is a strong baseline
signal and normalization otherwise deletes it.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path("data/processed")

# Below this, a user profile is treated as directionless rather than
# normalized into a random unit vector (see UserProfile.forward).
MIN_PROFILE_NORM = 1e-6


class RestaurantEncoder(nn.Module):
    tag_vecs: torch.Tensor
    tag_ids: torch.Tensor
    tag_mask: torch.Tensor
    absa_scores: torch.Tensor
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
        use_id_emb: bool = True,
        id_init_std: float = 0.01,
    ):
        super().__init__()

        if "absa_scores" not in features:
            raise KeyError(
                "features.pt has no 'absa_scores' -- it predates the ABSA review "
                "branch (it probably still has the old 'text_emb'). Rebuild it:\n"
                "    sbatch -p ice-gpu scripts/features.sbatch\n"
                "which needs data/processed/absa_scores.pt from scripts/run_absa.sh."
            )

        # --- frozen inputs, registered as buffers so .to(device) moves them and
        #     they are saved with the model but never receive gradients ---
        # Which content modalities this source actually provides. Yelp gives all
        # of them; the TripAdvisor London source has no cuisine categories,
        # price tier or coordinates, so those branches are simply not built
        # rather than being fed placeholder values -- a constant input would
        # still consume fusion width and let the model fit noise through it.
        self.has_tags = "tag_vecs" in features
        self.has_price = "price" in features
        self.has_geo = "geo" in features

        if self.has_tags:
            self.register_buffer("tag_vecs", features["tag_vecs"])
            self.register_buffer("tag_ids", features["tag_ids"])  # (N, T_max) int
            self.register_buffer("tag_mask", features["tag_mask"])  # (N, T_max) bool
        # (N, n_aspects * n_labels) -- recency-weighted food/service/price/ambience
        # sentiment from src/absa_tag_reviews.py, flattened.
        self.register_buffer("absa_scores", features["absa_scores"])
        self.register_buffer("name_emb", features["name_emb"])  # (N, 768)
        if self.has_price:
            self.register_buffer(
                "price", features["price"].long()
            )  # (N,) tiers 0..4, 0 = missing
        self.register_buffer(
            "numeric", features["numeric"]
        )  # (N, 3-4) z-scored: rating, log_count, [tag_count], rating_std
        if self.has_geo:
            self.register_buffer(
                "geo", features["geo"]
            )  # (N, 5) z-scored: lat, lng, dist_center, dist_cluster, log_cluster_size
        self.n_restaurants = self.absa_scores.shape[0]
        self.output_dims = output_dims
        self.use_id_emb = use_id_emb

        # --- per-branch projections: each modality gets its own Linear so the
        #     fusion MLP can weight them independently (the reason we concat
        #     rather than pre-average) ---
        sbert_dims = self.name_emb.shape[1]
        self.price_dims = (int(self.price.max()) + 1) if self.has_price else 0
        self.name_proj = nn.Linear(sbert_dims, branch_dims)
        self.absa_proj = nn.Linear(self.absa_scores.shape[1], branch_dims)
        if self.has_tags:
            self.tag_proj = nn.Linear(sbert_dims, branch_dims)
        # Price/numeric/geo are only ~14 raw dims against the wide 128-dim
        # branches, so at init they contribute ~4% of the concatenated width --
        # despite containing star rating and distance, plausibly the two most
        # predictive features for restaurant choice. Project them up so the
        # fusion MLP sees them on comparable footing, not as a rounding error.
        dense_dims = self.price_dims + self.numeric.shape[1] + (self.geo.shape[1] if self.has_geo else 0)
        self.dense_proj = nn.Linear(dense_dims, branch_dims // 2)

        n_wide = 3 if self.has_tags else 2  # name + absa [+ tags]
        mlp_input_dims = branch_dims * n_wide + branch_dims // 2
        self.fusion = nn.Sequential(
            nn.Linear(mlp_input_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, output_dims),
        )

        # --- collaborative residual + popularity bias ---
        # Small init so the content path dominates early; the model has to earn
        # its way into memorizing individual restaurants.
        self.id_emb = nn.Embedding(self.n_restaurants, output_dims)
        nn.init.normal_(self.id_emb.weight, std=id_init_std)
        self.item_bias = nn.Embedding(self.n_restaurants, 1)
        nn.init.zeros_(self.item_bias.weight)

    def _pool_tags(self, idx: torch.Tensor) -> torch.Tensor:
        """Masked mean of a restaurant's tag vectors. Returns (B, 768)."""
        ids = self.tag_ids[idx]  # (B, T_max)
        mask = self.tag_mask[idx]  # (B, T_max)
        vecs = self.tag_vecs[ids]  # (B, T_max, 768)

        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        vecs = vecs.sum(dim=1) / count
        return vecs

    def fuse(
        self,
        name_emb: torch.Tensor,  # (B, sbert_dims)
        absa: torch.Tensor,  # (B, n_aspects * n_labels)
        numeric: torch.Tensor,  # (B, 3-4) z-scored
        tag_vec: torch.Tensor | None = None,  # (B, sbert_dims) already pooled
        price: torch.Tensor | None = None,  # (B,) long, tier 0..4
        geo: torch.Tensor | None = None,  # (B, 5) z-scored
    ) -> torch.Tensor:
        """The content path, over RAW feature tensors rather than catalog indices.

        Split out from content_embedding so a restaurant that was never in the
        training catalog can still be embedded -- see embed_new(). Indexing
        buffers by position works only for restaurants that existed at build
        time, which would make live enrichment from another source impossible.

        tag_vec/price/geo are optional because not every source provides them
        (see the has_* flags in __init__); passing one the encoder was not
        built with is an error rather than being silently ignored.
        """
        for name, value, present in [
            ("tag_vec", tag_vec, self.has_tags),
            ("price", price, self.has_price),
            ("geo", geo, self.has_geo),
        ]:
            if (value is None) == present:
                raise ValueError(
                    f"{name} was {'omitted' if value is None else 'supplied'} but this "
                    f"encoder was built with has_{name.split('_')[0]}="
                    f"{present}. The feature set must match the one it was trained on."
                )

        dense = [numeric]
        if self.has_price:
            dense.insert(0, F.one_hot(price, num_classes=self.price_dims).float())
        if self.has_geo:
            dense.append(geo)

        wide = [self.name_proj(name_emb), self.absa_proj(absa)]
        if self.has_tags:
            wide.append(self.tag_proj(tag_vec))
        wide.append(self.dense_proj(torch.cat(dense, dim=1)))
        return self.fusion(torch.cat(wide, dim=1))

    def content_embedding(self, idx: torch.Tensor) -> torch.Tensor:
        """Content-only embedding, pre-normalization and without the id term.

        This is the path that generalizes to unseen restaurants, so it is also
        what content pretraining (src/pretrain_encoder.py) trains.
        """
        return self.fuse(
            name_emb=self.name_emb[idx],
            absa=self.absa_scores[idx],
            numeric=self.numeric[idx],
            tag_vec=self._pool_tags(idx) if self.has_tags else None,
            price=self.price[idx] if self.has_price else None,
            geo=self.geo[idx] if self.has_geo else None,
        )

    def embed_new(self, **features: torch.Tensor) -> torch.Tensor:
        """Embed a restaurant that is NOT in the training catalog (e.g. one just
        scraped from Google Maps -- see src/adapt_google.py).

        Content path only: there is no id embedding or bias for a restaurant the
        model has never seen, which is the correct cold-start behaviour rather
        than a limitation. Scores for it come purely from taste matching, with
        no popularity offset, until it accumulates real interactions.
        """
        return F.normalize(self.fuse(**features), dim=1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B,) restaurant indices. Returns (B, dim) L2-normalized embeddings."""
        z = self.content_embedding(idx)
        if self.use_id_emb:
            z = z + self.id_emb(idx)
        return F.normalize(z, dim=1)

    def bias(self, idx: torch.Tensor) -> torch.Tensor:
        """(B,) per-restaurant score offset. Added outside the cosine."""
        return self.item_bias(idx).squeeze(-1)

    def content_parameters(self):
        """Everything except the per-restaurant id/bias tables -- the parameters
        that transfer to a restaurant the model has never seen."""
        excluded = {id(self.id_emb.weight), id(self.item_bias.weight)}
        return (p for p in self.parameters() if id(p) not in excluded)


class UserProfile(nn.Module):
    """The user tower: a signed-weighted pool of the item tower's outputs over
    the restaurants a user rated.

    Weights are learned per star level rather than fixed at (rating -
    global_mean). They are INITIALIZED to exactly that rule, so training starts
    equivalent to the hand-set heuristic and can only depart from it if the data
    says so -- and the learned values are directly readable afterwards ("what
    does a 3-star review actually mean to this model?"). `--fixed-agg` pins them.

    Note on magnitude: the pooled vector is L2-normalized, so any scalar divisor
    (review count, sum of weights) cancels exactly and cannot make a confident
    user comparable to a lukewarm one. Confidence is not expressible here by
    construction; it lives in item_bias and the score scale instead.
    """

    def __init__(self, global_mean: float, learned: bool = True, n_levels: int = 5):
        super().__init__()
        self.global_mean = global_mean
        self.learned = learned
        # level i == a rating of (i + 1) stars
        init = torch.arange(1.0, n_levels + 1.0) - global_mean
        if learned:
            self.level_weight = nn.Parameter(init)
        else:
            self.register_buffer("level_weight", init)

    def rating_weights(self, ratings: torch.Tensor) -> torch.Tensor:
        """(…) raw stars -> (…) signed weights."""
        level = (ratings.round().long() - 1).clamp(0, len(self.level_weight) - 1)
        return self.level_weight[level]

    def forward(
        self,
        encoder: RestaurantEncoder,
        user_idx: torch.Tensor,  # (B,)
        exclude: torch.Tensor,  # (B,) held-out r_idx to leave out of each history
        hist_items: torch.Tensor,  # (n_users, H) padded restaurant indices
        hist_ratings: torch.Tensor,  # (n_users, H) padded raw stars
        hist_mask: torch.Tensor,  # (n_users, H) bool
    ) -> torch.Tensor:
        """(B, dim) L2-normalized user embeddings."""
        items = hist_items[user_idx]
        ratings = hist_ratings[user_idx]
        mask = hist_mask[user_idx]

        # leave-one-out: the item being predicted must not shape its own profile
        loo_mask = mask & (items != exclude.unsqueeze(1))

        N, H = items.shape
        w = self.rating_weights(ratings) * loo_mask  # (B, H), zero where masked

        # Encode DISTINCT restaurants only. items is (B, H) with H=max_history,
        # so a batch of 2048 is 100k+ lookups of which the overwhelming majority
        # are duplicates -- every padded slot is index 0, and popular
        # restaurants recur across users. Encoding the flattened tensor directly
        # materializes a (B*H, T_max, 768) tag gather, ~10GB at batch 2048;
        # deduping caps it at the catalog size and is exact, not an
        # approximation.
        uniq, inverse = torch.unique(items.reshape(-1), return_inverse=True)
        emb = encoder(uniq)[inverse].reshape(N, H, -1)
        pooled = (emb * w.unsqueeze(-1)).sum(dim=1)  # (B, dim)

        # A user whose signed weights cancel (balanced likes/dislikes, or an
        # empty post-LOO history) pools to ~0. Normalizing that amplifies
        # floating-point noise into an arbitrary unit vector AND sends a ~1/norm
        # gradient back through the encoder. Leave it at zero instead: a zero
        # profile scores every item equally, which is the honest answer.
        norm = pooled.norm(dim=1, keepdim=True)
        return torch.where(norm > MIN_PROFILE_NORM, pooled / norm.clamp(min=MIN_PROFILE_NORM), pooled)


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
