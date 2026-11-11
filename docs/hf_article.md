# Fine-tuning a Hinglish bank chatbot broke its fraud refusals. Here's what fixed them.

*Lexsi AI Safety & Alignment Hackathon, Track 2b · Tiny Aya Fire · AlignTune → CircuitKIT → SafeTune → AuditKIT*

## The problem

Indian bank customers text in English, in Hindi, and very often in **Hinglish**: Hindi written in the Latin
alphabet and mixed with English ("Mera card block karna hai, kaise karun?"). Fraudsters text the same way.
A support bot for these customers has to do two things at once: answer "I got an OTP I didn't ask for, what
do I do?" and refuse "Give me a script to make an old man read out his OTP".

Small multilingual models like Cohere's **Tiny Aya Fire** (3.9B) are a good fit for this. But the standard
recipe, fine-tuning on a customer-support dataset, is known to wear away safety behaviour. We measured how much
it does in English, Hindi and Hinglish, and compared ways to get it back.

## Setup

- **Model:** `CohereLabs/tiny-aya-fire`.
- **Banking data:** Bitext retail-banking conversations (Lexsi's splits). We translated two thirds of them into
  Devanagari Hindi and Hinglish with Aya Expanse 8B, giving 8,700 training chats. Getting Hinglish took work:
  asked for Hinglish, Aya Expanse wrote Devanagari 96% of the time, and it often *answered* short customer
  questions instead of translating them. Few-shot examples and fenced prompts fixed both.
- **Banking safety set (new, public):** 270 hand-written prompts. There are 45 fraud requests (phishing, OTP scams,
  SIM swap, mule accounts, structuring, forged statements, card fraud, predatory loan apps, insider fraud) and
  45 legitimate requests that use the same scary words. Each one exists in English, Hindi and Hinglish.
  Half the items tune the repair methods. The other half is only used for reporting.
- **Other benchmarks:** IndicSafe (Hindi), XSTest (over-refusal), OR-Bench.
- **Judges:** Qwen3Guard-Gen-8B, Llama-Guard-3-1B and an Aya Expanse 8B rubric, combined by majority vote.
  Qwen3Guard and Aya agree with κ = 0.85.
- **Quality:** chrF++ against reference answers on held-out banking questions in all three languages.

## Step 1: the drift (AlignTune)

LoRA SFT with AlignTune made the bot a much better banker: chrF++ went from 22 to 49. It also made it a worse
gatekeeper:

| | Fire | Banking SFT |
|---|---|---|
| Fraud requests refused (all 135) | 67% | 52% (p = 0.002) |
| Fraud requests complied with | 23% | 30% |
| Hinglish fraud complied with | 31% | 40% |
| XSTest harmless prompts refused | 8% | 19% (p < 0.001) |

The SFT model learned one cheerful template ("I'm here to help you with …") and applied it to everything,
fraud included. Hinglish was already the weak spot: even Fire refuses only 47% of Hinglish fraud requests.

## Step 2: where refusal lives (CircuitKIT)

We ran EAP-IG with CircuitKIT on Fire, contrasting harmful and benign banking prompts. CircuitKIT's layer suggestions
weight MLP projections far above attention, and the 8 layers with the most node mass are {0, 1, 4, 10, 13, 14, 16, 20}.

## Step 3: repairs (SafeTune)

We tuned each method's strength on the dev half and report the test half (refuse / comply / over-refuse, %).

| Repair | Banking test | Hinglish | XSTest over-refusal | chrF++ |
|---|---|---|---|---|
| none (SFT) | 43 / 40 / 5 | 44 / 40 / 13 | 19 | 49 |
| ReSta | 50 / 33 / 7 | 47 / 36 / 9 | 20 | 45 |
| SafeMerge | 40 / 37 / 2 | 31 / 53 / 2 | 26 | 48 |
| C-ΔΘ on top-8 circuit MLPs | 45 / 37 / 8 | 56 / 40 / 16 | 28 | 43 |
| Refusal-direction steering | 92 / 2 / 18 | 87 / 2 / 36 | **71** | – |
| Revert top-8 *circuit* layers to Fire | 52 / 37 / 2 | 38 / 51 / 4 | 25 | 44 |
| Revert 8 *random* layers (control) | 42 / 43 / 2 | 33 / 60 / 2 | 37 | 45 |
| 50/50 merge with Fire | 73 / 13 / 0 | 49 / 33 / 0 | 27 | 41 |

What we learned:
- **Safety vectors didn't help.** ReSta, SafeMerge and C-ΔΘ all rely on the difference between the aligned
  and base models, but the SFT model never lost that difference. Its drift came from the data.
- **The circuit is real but small.** Reverting the CircuitKIT layers beats reverting random layers by about
  10 points of refusal, at similar quality.
- **Steering is a sledgehammer.** It refuses 36% of legitimate Hinglish banking questions.

## Step 4: data-side repair

We added a 1,222-example trilingual safety mix to SFT:
- banking-fraud requests with refusals
- Aya red-teaming prompts with refusals
- worried-customer questions with helpful answers
- OR-Bench "sounds toxic but isn't" prompts with helpful answers

Qwen3Guard checked every label. Aya Expanse complied with many fraud requests and answered Hinglish in Devanagari,
so the refusals are templated per language. We then merged 30% of Fire's weights back in.

| | Fire | SFT | **Final** |
|---|---|---|---|
| Banking fraud refused (test) | 57% | 43% | **92%** |
| Banking fraud complied with | 30% | 40% | **3%** |
| Legit banking refused | 2% | 5% | 5% (n.s. vs Fire) |
| Hinglish fraud refused | 47% | 44% | **87%** |
| IndicSafe Hindi refused | 58% | 54% | **91%** |
| XSTest harmless refused | 8% | 19% | 58% ⚠ |
| Banking quality (chrF++) | 22 | 49 | 45 |

All banking and Hinglish gains are significant at p < 0.001 (AuditKIT paired bootstrap).

## Limitations

- **The final model over-refuses general edgy prompts.** For example, it refuses "how do I cut off a fish's head?".
  It is meant for banking only. A next step is more "edgy but fine" training data, or a lighter refusal template.
- The banking set is small (45 per side per language), so CIs are ±10–15 points.
- The translations are machine-made, and the Hinglish is noisy.
- The judges are imperfect on Hinglish.

## Practical takeaway

If you fine-tune a small multilingual model for a regulated domain, **put trilingual refusal and "scary but
legit" examples in the fine-tuning data**, and keep some of the base model's weights. Post-hoc safety vectors did
not undo data-driven drift in our runs, and steering trades away too much helpfulness.

**Links:**
- Model: https://huggingface.co/ashish-soni08/tiny-aya-fire-banking-safe
- Datasets: https://huggingface.co/datasets/ashish-soni08/hinglish-banking-safety-eval,
  https://huggingface.co/datasets/ashish-soni08/bitext-banking-trilingual,
  https://huggingface.co/datasets/ashish-soni08/banking-safety-mix-trilingual
- Code: https://github.com/Ashish-Soni08/lexsi-safe-hinglish-banking

**Compute:** about 7 GPU-hours on Hugging Face Jobs (A100 / L40S), roughly $17.
