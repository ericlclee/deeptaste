"""Score every review against a fixed set of aspect categories (food, service,
price, ambience -- the standard SemEval-2014 restaurant categories, minus
anecdotes/miscellaneous which isn't a queryable aspect term) using a
pretrained aspect-based sentiment analysis model from Hugging Face, instead
of building/training our own tagger.

Model: yangheng/deberta-v3-base-absa-v1.1 (PyABSA, ~1M downloads). Given
(review_text, aspect_term), it returns Negative/Neutral/Positive probabilities
for how the review talks about that aspect. We run every review against every
aspect and store the full probability distribution -- thresholding into
discrete tags is a separate downstream decision, not baked in here.

Usage:
    python src/absa_tag_reviews.py --limit 20          # smoke test
    python src/absa_tag_reviews.py                     # full run (all reviews x all aspects)
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

OUT = Path(os.environ.get("DEEP_TASTE_DATA", "data/processed"))
MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
ASPECTS = ["food", "service", "price", "ambience"]

# This model's tokenizer is old-format sentencepiece with no shipped
# tokenizer.json. Recent transformers versions mis-parse its spm.model
# while trying to auto-convert it to the fast format (reads it as a
# tiktoken BPE file and crashes) -- happens on both the newer transformers
# used here and the version pinned in environment.yml for PACE. Loading a
# pre-converted tokenizer.json sidesteps the conversion entirely and needs
# no extra packages (no sentencepiece/protobuf) at runtime. Regenerate it
# with: AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(TOKENIZER_DIR)
# from an environment where the conversion succeeds (transformers<=4.44 on
# Python<=3.12, with protobuf installed).
TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "models" / "absa-tokenizer"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_aspect(model, tokenizer, texts, aspect, device, batch_size, max_length) -> torch.Tensor:
    probs = torch.empty(len(texts), model.config.num_labels)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            [aspect] * len(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs[start : start + len(batch)] = torch.softmax(logits, dim=-1).cpu()
        if (start // batch_size) % 50 == 0:
            print(f"  {aspect}: {start + len(batch):,}/{len(texts):,}")
    return probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--aspects", default=",".join(ASPECTS))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="only score the first N reviews (smoke test)")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    aspects = args.aspects.split(",")
    out_path = Path(args.output) if args.output else OUT / "absa_scores.pt"

    device = get_device()
    print(f"device: {device}")
    print(f"loading {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"labels (column order in output): {labels}")

    reviews = pd.read_parquet(OUT / "reviews_split.parquet")
    if args.limit:
        reviews = reviews.iloc[: args.limit]
    texts = list(reviews.text)
    print(f"scoring {len(texts):,} reviews x {len(aspects)} aspects = {len(texts) * len(aspects):,} inferences")

    all_scores = torch.empty(len(texts), len(aspects), model.config.num_labels)
    for a_idx, aspect in enumerate(aspects):
        print(f"aspect: {aspect}")
        all_scores[:, a_idx, :] = score_aspect(
            model, tokenizer, texts, aspect, device, args.batch_size, args.max_length
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scores": all_scores.half(),  # (n_reviews, n_aspects, n_labels)
            "labels": labels,
            "aspects": aspects,
            "business_id": list(reviews.business_id),
            "user_id": list(reviews.user_id),
            "stars": torch.from_numpy(reviews.stars.to_numpy(dtype="float32")),
            "date": list(reviews.date.astype(str)),
            "split": list(reviews.split),
        },
        out_path,
    )
    print(f"\nwrote {out_path}  scores shape {tuple(all_scores.shape)}")


if __name__ == "__main__":
    main()
