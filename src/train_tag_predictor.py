"""Train a small classifier on top of the EXISTING text_emb (already computed
in features.py) to predict the LLM-extracted aspect tags (restaurant_tags.json).

This is the diagnostic from earlier: does the current recency-weighted pooled
review embedding already contain enough signal for food-quality/service/
atmosphere/etc. tags, without building a separate per-review attention
mechanism? No new pooling architecture, no re-encoding reviews -- text_emb is
frozen, only the small head on top trains.

Architecture is a bottleneck MLP specifically so the head is removable:

    text_emb (768) -> Linear+ReLU+Dropout -> hidden (128) -> Linear -> tag logits
                                                 ^ this is the reusable
                                                   "feature-rich embedding" --
                                                   load_embedder() below loads
                                                   a trained model and exposes
                                                   only this half, head cut off.

Usage:
    python src/train_tag_predictor.py
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from validate_tags import VALID_TAGS

OUT = Path(os.environ.get("DEEP_TASTE_DATA", "data/processed"))

# Flat, fixed-order tag vocabulary derived from validate_tags.py's taxonomy --
# single source of truth, no re-typing the tag list a third time.
TAG_VOCAB = [t for cat, tags in VALID_TAGS.items() for t in sorted(tags)]
TAG_TO_COL = {t: i for i, t in enumerate(TAG_VOCAB)}


class TagPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_tags: int, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, num_tags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))

    def embed_only(self, x: torch.Tensor) -> torch.Tensor:
        """Head-removed forward -- the reusable embedding this whole
        experiment is for. No head weights involved."""
        return self.embed(x)


def load_embedder(checkpoint_path=None, hidden_dim=128, input_dim=768, device=None) -> TagPredictor:
    """Load a trained TagPredictor for embedding extraction. Use
    model.embed_only(x), not model(x), to get the head-removed representation."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt_path = Path(checkpoint_path) if checkpoint_path else OUT / "tag_predictor.pt"
    model = TagPredictor(input_dim, hidden_dim, len(TAG_VOCAB)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


class TagDataset(Dataset):
    def __init__(self, text_emb: torch.Tensor, labels: torch.Tensor):
        self.text_emb = text_emb
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.text_emb[i], self.labels[i]


def build_dataset():
    """Join restaurant_tags.json (LLM-extracted labels) against features.pt's
    text_emb (frozen input). Restaurants flagged as non-restaurants (no
    category keys, just name/notes) are excluded -- they're a different kind
    of entity, not restaurants that happen to have no notable attributes."""
    tags = json.loads((OUT / "restaurant_tags.json").read_text())
    feats = torch.load(OUT / "features.pt", weights_only=False)
    biz_to_idx = {b: i for i, b in enumerate(feats["business_ids"])}
    text_emb = feats["text_emb"]

    rows, labels = [], []
    n_non_restaurant = n_missing = 0
    for biz_id, entry in tags.items():
        cats_present = set(entry.keys()) & set(VALID_TAGS)
        if not cats_present:
            n_non_restaurant += 1
            continue
        if biz_id not in biz_to_idx:
            n_missing += 1
            continue
        label = torch.zeros(len(TAG_VOCAB))
        for cat in cats_present:
            for t in entry[cat]:
                label[TAG_TO_COL[t]] = 1.0
        rows.append(biz_to_idx[biz_id])
        labels.append(label)

    print(
        f"{len(rows):,} labeled restaurants "
        f"({n_non_restaurant} non-restaurants excluded, {n_missing} missing from features.pt)"
    )
    return text_emb[rows], torch.stack(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else OUT / "tag_predictor.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")

    X, Y = build_dataset()
    n = len(X)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = int(n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    print(f"train: {len(X_train):,} | val: {len(X_val):,} | tags: {len(TAG_VOCAB)}")

    train_loader = DataLoader(
        TagDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True
    )

    model = TagPredictor(X.shape[1], args.hidden_dim, len(TAG_VOCAB), args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_dev, Y_train_dev = X_train.to(device), Y_train.to(device)
    X_val_dev, Y_val_dev = X_val.to(device), Y_val.to(device)

    # trivial baseline: always predict each tag's majority class (usually
    # "absent", since most tags are sparse) -- val accuracy needs to clear
    # this by a real margin to mean anything, given how imbalanced tags are.
    baseline_acc = torch.max(Y_val_dev.mean(dim=0), 1 - Y_val_dev.mean(dim=0)).mean().item()
    print(f"majority-class baseline val accuracy: {baseline_acc:.4f}\n")

    best_val_acc, best_epoch = -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_acc = ((torch.sigmoid(model(X_val_dev)) > 0.5).float() == Y_val_dev).float().mean().item()
            train_acc = ((torch.sigmoid(model(X_train_dev)) > 0.5).float() == Y_train_dev).float().mean().item()

        line = (
            f"epoch {epoch + 1}/{args.epochs}  loss {total_loss / len(X_train):.4f}  "
            f"train_acc {train_acc:.4f}  val_acc {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
            torch.save(model.state_dict(), ckpt_path)
            line += "  *best"
        print(line)

    print(
        f"\nsaved {ckpt_path} from epoch {best_epoch} "
        f"(best val accuracy {best_val_acc:.4f} vs. {baseline_acc:.4f} baseline)"
    )

    # Per-tag breakdown at the best checkpoint -- aggregate accuracy hides a
    # model that just learned to always predict 0 for rare tags.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    with torch.no_grad():
        val_preds = (torch.sigmoid(model(X_val_dev)) > 0.5).float().cpu()
    print(f"\n{'tag':30s} {'pos_rate':>9s} {'accuracy':>9s}")
    for i, tag in enumerate(TAG_VOCAB):
        pos_rate = Y_val[:, i].mean().item()
        acc = (val_preds[:, i] == Y_val[:, i]).float().mean().item()
        print(f"{tag:30s} {pos_rate:9.3f} {acc:9.3f}")


if __name__ == "__main__":
    main()
