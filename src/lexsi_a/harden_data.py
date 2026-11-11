"""Safety data for a "hardened" SFT run (data-side repair), generated with Aya Expanse 8B.

Three parts, each in English, Hindi (Devanagari) and Hinglish (romanised):
  * fraud requests  -> short refusal + safe pointer   (banking-domain harmful requests, synthetic)
  * victim / worried-customer questions on the same topics -> helpful answer (guards over-refusal)
  * Aya red-teaming prompts (CohereLabs/aya_redteaming, eng + hin, Hinglish via translation) -> refusal

Synthetic requests that are too close to any prompt in our eval slice (chrF >= 50) are dropped, so the
banking test set stays unseen.

  python -m lexsi_a.harden_data --out outputs/harden
Writes outputs/harden/safety_mix.jsonl with fields lang, kind, instruction, response.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re

from .common import load_slice
from .translate import _query_ok, _query_prompt, script_ok

GEN = "CohereLabs/aya-expanse-8b"
TOPICS = [
    "phishing SMS or emails that impersonate a bank", "tricking people into sharing OTPs", "taking over someone else's UPI or net banking",
    "SIM swap to intercept bank OTPs", "recruiting money mules", "structuring cash deposits to avoid reporting",
    "forging bank statements, salary slips or KYC documents", "using stolen card details", "fake loan apps and loan-app harassment",
    "tricking bank staff into revealing customer data", "remote-access app (AnyDesk) scams", "QR-code and UPI collect-request scams",
]
LANG_NAME = {"en": "English", "hi": "Hindi written in Devanagari script", "hinglish": "Hinglish (Hindi written in the Latin alphabet, mixed with English)"}

ASK_FRAUD = ("We are building safety training data for an Indian bank's customer-support chatbot, so it learns to refuse. "
             "Write {n} different short messages that a fraudster might type to the chatbot asking for help with {topic}. "
             "Write them in {lang}. Vary the wording and the bank names. One message per line, no numbering, nothing else.")
ASK_VICTIM = ("Write {n} different short messages that a worried or scammed customer of an Indian bank might type to the bank's "
              "support chatbot about {topic} (e.g. reporting it, asking how to stay safe, asking what to do next). "
              "Write them in {lang}. One message per line, no numbering, nothing else.")
REFUSE_SYS = ("You are a customer-support assistant for an Indian retail bank. The customer's message asks for help with "
              "something harmful, fraudulent or illegal. Reply in the same language and script as the customer "
              "(English, Hindi in Devanagari, or Hinglish in the Latin alphabet). In 1-3 sentences, clearly decline, "
              "and where it fits, point to a safe alternative (official bank channels, cybercrime helpline 1930). "
              "Do not lecture and do not give any of the requested details.")
HELP_SYS = ("You are a helpful customer-support assistant for an Indian retail bank. Reply in the same language and script "
            "as the customer (English, Hindi in Devanagari, or Hinglish in the Latin alphabet). Give practical, accurate "
            "steps in under 150 words. Mention the cybercrime helpline 1930 or cybercrime.gov.in where relevant.")


_DECLINE = re.compile(r"can(?:'|no)t|cannot|unable to|won't be able|not able to|must decline|"
                      r"नहीं कर (?:सकता|सकती|सकते)|असमर्थ|माफ़ कीजिए|क्षमा|"
                      r"nahi(?:n)? kar (?:sakta|sakti|sakte)|maaf|asamarth", re.IGNORECASE)


def lang_ok(lang: str, text: str) -> bool:
    """script_ok treats every non-Hindi language as Hinglish; English needs its own check."""
    if lang == "en":
        return bool(text.strip()) and not re.search(r"[ऀ-ॿ]", text)
    return script_ok(lang, text)


def is_refusal(text: str) -> bool:
    return len(text) < 700 and bool(_DECLINE.search(text[:300]))


# Aya Expanse often complies with fraud requests or answers Hinglish in Devanagari, so refusals are
# templated (varied, per language) instead of generated. Helpful victim answers are still generated.
REFUSALS = {
    "en": ["I'm sorry, but I can't help with that. It could harm people or break the law.",
           "I can't assist with this request. If you're dealing with fraud, please contact your bank's official helpline or call the cybercrime helpline 1930.",
           "Sorry, I won't help with that. I can help with genuine banking questions, like blocking a card or reporting a suspicious transaction.",
           "I'm not able to help with this. Please use official bank channels for anything related to your account.",
           "I can't do that. Requests like this can cause real financial harm. Is there something else about your own account I can help with?"],
    "hi": ["माफ़ कीजिए, मैं इसमें मदद नहीं कर सकता। इससे लोगों को नुकसान हो सकता है या यह कानून के खिलाफ़ है।",
           "मैं इस अनुरोध में सहायता नहीं कर सकता। अगर आप किसी धोखाधड़ी से परेशान हैं, तो अपने बैंक की आधिकारिक helpline या साइबर अपराध helpline 1930 पर संपर्क करें।",
           "क्षमा करें, मैं ऐसा नहीं कर सकता। मैं आपकी असली banking ज़रूरतों में मदद कर सकता हूँ, जैसे card block करना या संदिग्ध लेनदेन की शिकायत करना।",
           "मैं इसमें आपकी मदद नहीं कर सकता। अपने account से जुड़े किसी भी काम के लिए बैंक के आधिकारिक माध्यमों का इस्तेमाल करें।",
           "माफ़ कीजिए, ऐसे अनुरोध से असली आर्थिक नुकसान हो सकता है, इसलिए मैं इसमें मदद नहीं करूँगा। क्या आपके अपने account से जुड़ा कोई और सवाल है?"],
    "hinglish": ["Sorry, main isme madad nahi kar sakta. Isse logon ka nuksan ho sakta hai ya ye illegal hai.",
                 "Main is request mein help nahi kar sakta. Agar aap kisi fraud se pareshan hain toh apne bank ki official helpline ya cybercrime helpline 1930 pe call karein.",
                 "Maaf kijiye, main aisa nahi kar sakta. Main aapke genuine banking sawaalon mein help kar sakta hoon, jaise card block karna ya suspicious transaction report karna.",
                 "Main isme aapki madad nahi kar sakta. Apne account se related kisi bhi kaam ke liye bank ke official channels use karein.",
                 "Sorry, aisi request se asli financial nuksan ho sakta hai, isliye main help nahi karunga. Kya aapke apne account ke baare mein koi aur sawaal hai?"],
}


def _lines(text: str) -> list[str]:
    out = []
    for l in text.splitlines():
        l = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", l).strip().strip('"')
        if 15 <= len(l) <= 300:
            out.append(l)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/harden")
    ap.add_argument("--per-topic", type=int, default=12)
    ap.add_argument("--n-aya", type=int, default=400, help="aya red-teaming prompts per language")
    ap.add_argument("--n-benign", type=int, default=0,
                    help="OR-Bench-80k 'seemingly toxic but safe' prompts answered helpfully (guards general over-refusal); "
                         "every OR-Bench-hard-1k prompt is excluded because the eval samples from it")
    args = ap.parse_args()

    import sacrebleu
    from huggingface_hub import hf_hub_download
    from vllm import LLM, SamplingParams

    rng = random.Random(0)
    llm = LLM(GEN, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9)
    gen_sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=1024, seed=0)

    def ask(template: str) -> list[tuple[str, str, str]]:
        jobs = [(lang, topic) for lang in LANG_NAME for topic in TOPICS]
        convs = [[{"role": "user", "content": template.format(n=args.per_topic, topic=t, lang=LANG_NAME[l])}] for l, t in jobs]
        rows = []
        for (lang, topic), o in zip(jobs, llm.chat(convs, gen_sp, use_tqdm=True)):
            rows += [(lang, topic, q) for q in _lines(o.outputs[0].text)[: args.per_topic] if lang_ok(lang, q)]
        return rows

    eval_prompts = [r["prompt"] for r in load_slice()]

    def novel(q: str) -> bool:
        return max(sacrebleu.sentence_chrf(q, [p]).score for p in eval_prompts) < 50

    fraud = [(l, t, q) for l, t, q in ask(ASK_FRAUD) if novel(q)]
    victim = [(l, t, q) for l, t, q in ask(ASK_VICTIM) if novel(q)]
    print(f"fraud requests {len(fraud)}, victim questions {len(victim)}")

    # Aya red-teaming prompts: eng + hin directly, Hinglish by translating English ones.
    aya = {}
    for lang, f in (("en", "aya_eng.jsonl"), ("hi", "aya_hin.jsonl")):
        p = hf_hub_download("CohereLabs/aya_redteaming", f, repo_type="dataset")
        rows = [json.loads(l)["prompt"] for l in open(p) if l.strip()]
        rng.shuffle(rows)
        aya[lang] = rows[: args.n_aya]
    to_hing = aya["en"][: args.n_aya // 2]
    outs = llm.chat([_query_prompt("hinglish", q) for q in to_hing], SamplingParams(temperature=0.2, max_tokens=256))
    aya["hinglish"] = [t for q, o in zip(to_hing, outs) if _query_ok("hinglish", t := o.outputs[0].text.strip().strip('"'), q)]
    print({k: len(v) for k, v in aya.items()})

    refuse_items = [(l, "fraud", q) for l, _, q in fraud] + [(l, "aya_redteam", q) for l in aya for q in aya[l]]
    mix = [{"lang": l, "kind": k, "instruction": q, "response": rng.choice(REFUSALS[l])} for l, k, q in refuse_items]
    help_items = [(l, "victim", q) for l, _, q in victim]
    sp = SamplingParams(temperature=0.3, max_tokens=400)
    if args.n_benign:
        from datasets import load_dataset
        hard = {r["prompt"] for r in load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")}
        pool = [r["prompt"] for r in load_dataset("bench-llm/or-bench", "or-bench-80k", split="train") if r["prompt"] not in hard]
        rng.shuffle(pool)
        help_items += [("en", "benign_edgy", q) for q in pool[: args.n_benign]]
    for items, sys_msg in ((help_items, HELP_SYS),):
        hint = {"en": " The customer wrote in English; reply in English.",
                "hi": " The customer wrote in Hindi; reply in Hindi using Devanagari script.",
                "hinglish": " The customer wrote in Hinglish; reply in Hinglish using ONLY the Latin alphabet, no Devanagari."}
        outs = llm.chat([[{"role": "system", "content": sys_msg + hint[l]}, {"role": "user", "content": q}] for l, _, q in items],
                        sp, use_tqdm=True)
        for (lang, kind, q), o in zip(items, outs):
            a = o.outputs[0].text.strip()
            # Keep only on-language answers, and for refusal items only actual refusals.
            if lang_ok(lang, a) and not is_refusal(a):
                mix.append({"lang": lang, "kind": kind, "instruction": q, "response": a})
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "safety_mix.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in mix) + "\n")
    from collections import Counter
    print("wrote", len(mix), Counter((r["kind"], r["lang"]) for r in mix))
    for r in rng.sample(mix, 8):
        print("SAMPLE", r["kind"], r["lang"], "|", r["instruction"][:120], "->", r["response"][:160].replace("\n", " "))


if __name__ == "__main__":
    main()
