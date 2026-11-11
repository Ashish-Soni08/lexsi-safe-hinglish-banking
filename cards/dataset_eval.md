---
license: cc-by-4.0
language: [en, hi]
tags: [safety, banking, hinglish, over-refusal, red-teaming, lexsi-hackathon]
task_categories: [text-generation]
size_categories: [n<1K]
---

# Hinglish Banking Safety Eval

270 hand-written retail-banking prompts for testing a bank chatbot's safety in **English, Devanagari Hindi and
romanised Hinglish**: 45 harmful requests (should be refused: phishing, OTP scams, account takeover, SIM swap,
mule accounts, structuring, forgery, card fraud, predatory loan apps, social engineering, insider fraud) and
45 benign requests (should be answered, and deliberately full of scary words like OTP, CVV, fraud, block,
scam), each in all three forms.

Fields: `id`, `label` (harmful/benign), `expected` (refuse/comply), `category`, `lang` (en/hi/hinglish), `prompt`.
Items `000–024` were used to tune repair strengths (dev), `025–044` only for reporting (test).

Part of https://github.com/Ashish-Soni08/lexsi-safe-hinglish-banking. Harmful prompts are requests only;
no harmful completions are included.
