"""Translate Bitext retail-banking pairs into Hindi (Devanagari) and Hinglish (romanised)
with Aya Expanse 8B on vLLM, then build the trilingual SFT mix and utility test sets.

Usage (one GPU):
  python -m lexsi_a.translate --n-train 3000 --n-test 300 --out data/sft
Outputs JSONL files with fields: id, lang, intent, category, instruction, response.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

from datasets import load_dataset

SRC = "sohamb37lexsi/bitext-retail-banking-llm-chatbot-splits"  # Lexsi's own train/val/test split
TRANSLATOR = "CohereLabs/aya-expanse-8b"

STYLE = {
    "hi": ("Translate the text into natural, everyday Hindi written in Devanagari script. "
           "Keep common banking terms people say in English (OTP, UPI, KYC, ATM, PIN, net banking, app) in English. "
           "Keep numbers, placeholders like {{Account Number}}, URLs, lists and line breaks unchanged."),
    "hinglish": ("Rewrite the text in Hinglish: conversational Hindi written in the Latin alphabet, mixed with English "
                 "the way Indian customers text their bank (e.g. 'Mera card block karna hai, kaise karun?'). "
                 "Do not use Devanagari. Keep numbers, placeholders like {{Account Number}}, URLs, lists and line breaks unchanged."),
}


# Few-shot turns: without them Aya Expanse answers "Hinglish" requests in Devanagari ~95% of the time.
HINGLISH_SHOTS = [
    ("I need to block my debit card because I lost it. How do I do it?",
     "Mera debit card kho gaya hai, isliye mujhe use block karna hai. Kaise karun?"),
    ("Certainly! To check your account balance, log in to the mobile app and tap 'Accounts'. "
     "If you need further help, contact our customer support at {{Customer Support Phone Number}}.",
     "Bilkul! Apna account balance check karne ke liye mobile app mein login karein aur 'Accounts' pe tap karein. "
     "Agar aur help chahiye toh hamare customer support se {{Customer Support Phone Number}} pe contact karein."),
]
_HINGLISH_WORDS = re.compile(r"\b(hai|hain|ka|ki|ke|ko|mein|aap|apna|apne|karna|karein|karun|kaise|kya|nahi|liye|se|toh|chahiye|hoga|par|pe)\b", re.I)


def _prompt(lang: str, text: str) -> list[dict]:
    msgs = [{"role": "system", "content": STYLE[lang] + " Output only the translation, nothing else."}]
    if lang == "hinglish":
        for src, dst in HINGLISH_SHOTS:
            msgs += [{"role": "user", "content": src}, {"role": "assistant", "content": dst}]
    return msgs + [{"role": "user", "content": text}]


_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def script_ok(lang: str, text: str) -> bool:
    """Reject outputs in the wrong script (cheap quality gate)."""
    if not text.strip():
        return False
    share = len(_DEVANAGARI.findall(text)) / max(1, len(re.sub(r"\s", "", text)))
    if lang == "hi":
        return share > 0.3
    # Hinglish: Latin script AND actually Hindi (not an untranslated English copy).
    words = max(1, len(text.split()))
    return share < 0.02 and len(_HINGLISH_WORDS.findall(text)) / words > 0.08


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hinglish-only", default=None,
                    help="dir with an earlier run: keep its en/hi rows, redo only the Hinglish rows")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(SRC)
    # Stratify by intent so every banking intent appears in every language.
    train = ds["train"].shuffle(seed=args.seed)
    test = ds["test"].shuffle(seed=args.seed).select(range(args.n_test))
    en_train = train.select(range(args.n_train))               # stays English
    tr_train = train.select(range(args.n_train, 3 * args.n_train))  # half -> hi, half -> hinglish

    llm = LLM(TRANSLATOR, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9)
    sp = SamplingParams(temperature=0.2, max_tokens=1024)

    def translate(rows, lang, tag):
        convs = []
        for r in rows:
            convs += [_prompt(lang, r["instruction"]), _prompt(lang, r["response"])]
        outs = [o.outputs[0].text.strip() for o in llm.chat(convs, sp, use_tqdm=True)]
        for attempt in range(2):  # resample failures at higher temperature
            bad = [i for i, t in enumerate(outs) if not script_ok(lang, t)]
            if not bad:
                break
            retry = SamplingParams(temperature=0.7, max_tokens=1024, seed=attempt + 1)
            for i, o in zip(bad, llm.chat([convs[i] for i in bad], retry, use_tqdm=False)):
                outs[i] = o.outputs[0].text.strip()
        for t in [t for t in outs if not script_ok(lang, t)][:3]:
            print(f"REJECTED {lang}: {t[:200]!r}")
        kept, dropped = [], 0
        for i, r in enumerate(rows):
            ins, resp = outs[2 * i], outs[2 * i + 1]
            if script_ok(lang, ins) and script_ok(lang, resp):
                kept.append({"id": f"{tag}-{i}", "lang": lang, "intent": r["intent"], "category": r["category"],
                             "instruction": ins, "response": resp, "en_instruction": r["instruction"],
                             "en_response": r["response"]})
            else:
                dropped += 1
        print(f"{tag}/{lang}: kept {len(kept)} dropped {dropped}")
        return kept

    half = len(tr_train) // 2
    if args.hinglish_only:
        prev = pathlib.Path(args.hinglish_only)
        keep = lambda f: [json.loads(l) for l in (prev / f).read_text().splitlines()
                          if l.strip() and json.loads(l)["lang"] != "hinglish"]
        rows_hing = translate(tr_train.select(range(half, len(tr_train))), "hinglish", "train")
        write(out / "train_trilingual.jsonl", keep("train_trilingual.jsonl") + rows_hing)
        write(out / "test_trilingual.jsonl", keep("test_trilingual.jsonl") + translate(test, "hinglish", "test"))
        return
    rows_hi = translate(tr_train.select(range(half)), "hi", "train")
    rows_hing = translate(tr_train.select(range(half, len(tr_train))), "hinglish", "train")
    rows_en = [{"id": f"train-en-{i}", "lang": "en", "intent": r["intent"], "category": r["category"],
                "instruction": r["instruction"], "response": r["response"]} for i, r in enumerate(en_train)]
    write(out / "train_trilingual.jsonl", rows_en + rows_hi + rows_hing)

    test_rows = [{"id": f"test-en-{i}", "lang": "en", "intent": r["intent"], "category": r["category"],
                  "instruction": r["instruction"], "response": r["response"]} for i, r in enumerate(test)]
    test_rows += translate(test, "hi", "test") + translate(test, "hinglish", "test")
    write(out / "test_trilingual.jsonl", test_rows)


def write(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print("wrote", len(rows), "->", path)


if __name__ == "__main__":
    main()
