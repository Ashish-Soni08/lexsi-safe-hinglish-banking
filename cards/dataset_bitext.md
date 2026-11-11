---
license: cdla-sharing-1.0
language: [en, hi]
tags: [banking, hinglish, customer-support, translation, lexsi-hackathon]
task_categories: [text-generation]
---

# Bitext Banking, Trilingual (English / Hindi / Hinglish)

Retail-banking customer-support conversations from Bitext (via Lexsi's splits
`sohamb37lexsi/bitext-retail-banking-llm-chatbot-splits`), with two thirds translated by Aya Expanse 8B into
Devanagari Hindi and romanised Hinglish.

- `train`: 8,700 rows (3,000 en, 2,977 hi, 2,723 hinglish). `test`: 872 rows (from Lexsi's test split).
- Quality gates: script check (Devanagari share), a Hinglish word check, and questions re-translated with a
  fenced prompt when the translator answered instead of translating. Translations are machine-generated
  and the Hinglish in particular is noisy.
- Fields: `id`, `lang`, `intent`, `category`, `instruction`, `response`, and the English originals.

Licence follows the Bitext source dataset. Built for https://github.com/Ashish-Soni08/lexsi-safe-hinglish-banking
