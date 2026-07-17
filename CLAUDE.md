# Local Restaurant Recommender — Project Spec

## Purpose

Graph/embedding-based restaurant recommendation system. Primary goal: resume
differentiator for ML/data science roles coming out of a master's program.
Scope is chosen to demonstrate genuine understanding of modern recsys
techniques (two-tower embeddings, GNN-style aggregation, ranking losses,
online profile updates, exploration/diversity) without requiring
production-scale infrastructure.

## Data

**Primary dataset:** [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
- `business.json` — restaurant metadata: name, categories/cuisine, location
  (lat/long), price tier, attributes, average rating, review count.
- `review.json` — the core interaction data: user_id, business_id, stars,
  review text, date. This is the (user, item, rating) triple set used for
  training.
- `user.json` — user_id, review_count, yelping_since, friends (array of
  user_ids — social graph, optional extension), average_stars, fan/vote
  counts. No demographic data.

Live enrichment (optional, later phase): Google Places API or Yelp Fusion
API for "restaurants near me" freshness on top of the offline-trained model.

## Architecture Overview

Two-tower / dual-encoder design. Both towers output vectors in the same
embedding space; recommendation score = similarity(user_emb, restaurant_emb).

```
Restaurant features ──> Restaurant Encoder ──> restaurant_emb ─┐
                                                                 ├─> score = cos_sim(user_emb, restaurant_emb)
User's rated restaurants ──> Aggregation (weighted sum) ──> user_emb ─┘
```

This is intentionally framed as a **shallow, single-hop, one-directional
graph neural network**: the bipartite user-restaurant graph is real, and the
aggregation step (weighted sum of neighbor embeddings) is the core GNN
message-passing primitive. It is *not* a full multi-layer GNN — no
restaurant-side aggregation from users, no multi-hop propagation, no learned
transformation on the aggregation. Document this explicitly as a scoping
decision. Stretch goal: extend to a real 2-layer bidirectional GNN
(GraphSAGE/LightGCN-style) if time allows.

## Component 1: Restaurant Encoder

**Inputs (per restaurant), by modality:**
- Categorical: cuisine tags (multi-label), price tier → `nn.Embedding`
  lookup, averaged across multi-label cuisines.
- Text: review text → pretrained sentence embeddings (Sentence-BERT /
  all-MiniLM, frozen initially), mean-pooled across a restaurant's reviews.
- Geographic: lat/long, normalized or geohash-bucketed + embedded.
- Numerical: average rating, review count, rating distribution (5-dim
  histogram of 1-5 star %), all z-score normalized.

**Fusion:** concatenate all modality vectors → small MLP (2-3 dense layers,
ReLU, dropout) → final embedding (start with dim 64-128). L2-normalize
output.

**This MLP's weights are the trainable part of the restaurant tower.**

## Component 2: User Profile (Aggregation)

`user_emb = L2_normalize( Σ_i  w(rating_i) · restaurant_emb_i )`

where `w(rating)` is a **signed** weight (e.g. `rating - 3`, or
`rating - user's_average_rating`) — critical detail, using raw 1-5 star
value directly makes everything additive and breaks the "negative reviews
should subtract" intent.

Starting version: fixed weighted sum, no separate trainable aggregation
network (two-stage training, see below).
Stretch: learned aggregation (attention over rated restaurants — SASRec-style
self-attention weighting by recency/relevance) instead of fixed weights.

## Training

**Two-stage approach (recommended starting scope — easier to implement/debug
than full end-to-end):**
1. Train restaurant encoder + user aggregation jointly against a ranking loss
   (see below). This *is* effectively end-to-end since the aggregation has no
   independent params yet — gradients flow from the loss through user_emb's
   constituent restaurant embeddings and directly into the encoder MLP.
2. (Stretch) Once basic pipeline works, add a learned transformation/attention
   on the aggregation step and retrain.

**Loss: BPR (Bayesian Personalized Ranking)**

```
For each (user u, positive restaurant i, negative restaurant j):
loss = -log( sigmoid( score(u, i) - score(u, j) ) )
```

- Positive: a restaurant the user rated highly.
- Negative: sampled restaurant the user didn't rate (or rated low).
- Start with random negative sampling to validate the pipeline.
- Upgrade to **hard-negative sampling** (same cuisine/price/geo cluster but
  not chosen by the user) once basic loop works — meaningfully improves
  embedding quality, forces finer-grained distinctions.
- Prefer BPR over MSE-on-raw-rating: recommendation quality depends on
  relative ranking, not absolute score calibration.

## Online User Profile Updates

Goal: new reviews should genuinely shift recommendations, without full
retraining, and without simply overwriting history.

Progression (build in this order):
1. **Exponential moving average update** (baseline):
   `user_emb ← user_emb + η · signed_rating · (restaurant_emb - user_emb)`
   — η controls responsiveness vs. memory retention. Simple, tunable, good
   first implementation.
2. **Error-driven / bandit-style update** (the differentiator — prioritize
   this over jumping to a sequence model): update magnitude driven by
   *prediction error*, not raw rating —
   `error = actual_rating - predicted_score(u, i)` (predicted_score computed
   *before* the user visited, i.e. at recommendation time).
   A confirmed prediction barely moves the profile; a surprising outcome
   moves it a lot. This is the contextual-bandit framing (recommendation =
   action, rating = reward) and is the same framework that naturally unifies
   with the exploration mechanism below.
3. (Stretch) Recurrent/sequential profile (GRU4Rec-style hidden state, or
   SASRec-style self-attention over interaction sequence) — learned version
   of step 1/2 instead of a hand-set η.

## Exploration / Diversity (avoid recommendation convergence)

Separate concern from profile updating — solved at the re-ranking stage, not
by the encoder or aggregation:
- **MMR (Maximal Marginal Relevance) re-ranking** — simplest to implement.
  Take top-K by similarity, re-rank to penalize redundancy among selections.
  Build this first.
- (Stretch) Contextual bandit (LinUCB-style) — ties directly into the
  error-driven profile update above; same framework handles both
  exploration and adaptive updating jointly. This is the most defensible
  "advanced" feature if time allows — flag as the last major differentiator
  to add after the core pipeline works.

## Serving / Inference

- Restaurant embeddings: precomputed in batch, stored in a vector index
  (FAISS, or in-memory matrix — dataset size doesn't require anything
  heavier for a project of this scope).
- User embeddings: recomputed live on each new review (cheap — one node's
  local aggregation, not a graph-wide recompute).
- Recommendation = ANN lookup of user_emb against restaurant index, then MMR
  re-ranking pass.
- Full model retraining (encoder weights): scheduled/batch, not online.
  Explicitly scope "why online weight updates are a harder unsolved problem"
  as a documented design decision rather than attempting full dynamic/
  temporal GNN training (TGN, DySAT) — out of scope for this project.

## Suggested Build Order

1. Data pipeline: load Yelp Open Dataset, filter to a manageable
   metro area, build (user, business, rating, review_text) table.
2. Restaurant encoder: feature extraction + fusion MLP, sanity-check
   embeddings (nearest-neighbor restaurants should look sensible).
3. User aggregation (fixed weighted sum) + BPR training loop with random
   negatives. Get end-to-end training working.
4. Evaluation: precision@k / recall@k / NDCG on held-out reviews.
5. Hard-negative sampling upgrade.
6. Online EMA profile update + simple demo (rate a restaurant, watch
   recommendations shift).
7. MMR re-ranking for diversity.
8. Error-driven (bandit-style) profile update.
9. Stretch: attention-based aggregation, second GNN hop, social graph
   (friends) as additional signal, live API enrichment.

## Tech Stack

- PyTorch (core model)
- sentence-transformers (frozen text embeddings)
- pandas / polars (data pipeline)
- FAISS (vector index, once past prototyping with plain matrix ops)
- PyTorch Geometric or DGL — only needed if/when extending to a real
  multi-hop GNN (stretch goal, not required for core scope)
- Streamlit or FastAPI + simple frontend for the interactive demo

## Open Scoping Decisions (revisit as needed)

- Two-stage vs. fully joint training — start two-stage-equivalent (see
  Training section), revisit if time allows.
- How far to push the "GNN" framing — current design is a legitimate but
  shallow instance; be precise in writeup about what was/wasn't implemented
  vs. full LightGCN/PinSage.
- Whether to incorporate the Yelp `friends` field as a social signal —
  additive, not required for core recommendation mechanism to work.
