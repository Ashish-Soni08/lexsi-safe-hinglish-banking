"""Publish the final model, the baseline SFT model and the three datasets, plus an HF collection.

Repos are created PRIVATE unless --public is passed (flip to public once the owner approves).
  python scripts/publish.py --runs /tmp/runs [--public]
Needs: /tmp/final (merged final model) and /tmp/sft (SFT model), see jobs/stage_publish.sh.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from huggingface_hub import HfApi

ROOT = pathlib.Path(__file__).resolve().parents[1]
USER = "ashish-soni08"
FINAL, SFT = f"{USER}/tiny-aya-fire-banking-safe", f"{USER}/tiny-aya-fire-banking-sft"
D_EVAL, D_BITEXT, D_SAFETY = (f"{USER}/hinglish-banking-safety-eval", f"{USER}/bitext-banking-trilingual",
                              f"{USER}/banking-safety-mix-trilingual")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    api, private, runs = HfApi(), not args.public, pathlib.Path(args.runs)
    cards = ROOT / "cards"

    def repo(rid, kind):
        api.create_repo(rid, repo_type=kind, private=private, exist_ok=True)
        api.update_repo_settings(rid, repo_type=kind, private=private)

    # models
    for rid, folder, card in ((FINAL, "/tmp/final", "model_safe.md"), (SFT, "/tmp/sft", "model_sft.md")):
        repo(rid, "model")
        if pathlib.Path(folder).exists():
            api.upload_folder(repo_id=rid, folder_path=folder, commit_message="Upload model",
                              ignore_patterns=[".cache/*", "README.md"])
        api.upload_file(path_or_fileobj=str(cards / card), path_in_repo="README.md", repo_id=rid)
        for f in ("RESULTS.md", "SIGNIFICANCE.md", "summary.json"):
            api.upload_file(path_or_fileobj=str(ROOT / "results" / f), path_in_repo=f"eval/{f}", repo_id=rid)

    # datasets
    uploads = {
        D_EVAL: [(ROOT / "data" / "banking_safety_slice.jsonl", "data/test.jsonl")],
        D_BITEXT: [(runs / "fixinst/outputs/sft/train_trilingual.jsonl", "data/train.jsonl"),
                   (runs / "fixinst/outputs/sft/test_trilingual.jsonl", "data/test.jsonl")],
        D_SAFETY: [(runs / "harden2/outputs/harden/safety_mix.jsonl", "data/train.jsonl")],
    }
    for (rid, files), card in zip(uploads.items(), ("dataset_eval.md", "dataset_bitext.md", "dataset_safety.md")):
        repo(rid, "dataset")
        for src, dst in files:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=rid, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(cards / card), path_in_repo="README.md", repo_id=rid, repo_type="dataset")

    # collection (the token may lack collection scope; everything else is already published)
    try:
        coll = api.create_collection(title="Safe Hinglish Banking Assistant (Tiny Aya)", namespace=USER, private=private,
                                 exists_ok=True,
                                     description="Lexsi hackathon: banking SFT on Tiny Aya Fire, safety drift audit and repair.")
        for rid, kind in ((FINAL, "model"), (SFT, "model"), (D_EVAL, "dataset"), (D_BITEXT, "dataset"), (D_SAFETY, "dataset")):
            api.add_collection_item(coll.slug, item_id=rid, item_type=kind, exists_ok=True)
        if not private:
            api.update_collection_metadata(coll.slug, private=False)
        print(json.dumps({"collection": f"https://huggingface.co/collections/{coll.slug}", "private": private}))
    except Exception as e:  # noqa: BLE001
        print("COLLECTION FAILED:", type(e).__name__, str(e)[:300])


if __name__ == "__main__":
    main()
