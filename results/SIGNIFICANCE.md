| Baseline → candidate | Suite | Metric | Baseline | Candidate | Δ | p |
|---|---|---|---|---|---|---|
| fire → sft | banking_test | refusal_on_harmful | 57% | 43% | -13 | 0.137 |
| fire → sft | banking_test | harmful_compliance | 30% | 40% | +10 | 0.292 |
| fire → sft | banking_test | over_refusal | 2% | 5% | +3 | 0.429 |
| fire → sft | banking | refusal_on_harmful | 67% | 52% | -16 | 0.002 ** |
| fire → sft | banking | harmful_compliance | 23% | 30% | +7 | 0.162 |
| fire → sft | banking | over_refusal | 1% | 4% | +3 | 0.142 |
| fire → sft | indicsafe_hi | refusal_on_harmful | 58% | 54% | -3 | 0.248 |
| fire → sft | indicsafe_hi | harmful_compliance | 3% | 4% | +1 | 0.507 |
| fire → sft | indicsafe_hi | over_refusal | 0% | 3% | +3 | 0.068 |
| fire → sft | xstest | refusal_on_harmful | 82% | 82% | +0 | 1.000 |
| fire → sft | xstest | harmful_compliance | 2% | 3% | +1 | 0.584 |
| fire → sft | xstest | over_refusal | 8% | 19% | +11 | 0.000 ** |
| fire → sft | orbench | refusal_on_harmful | 94% | 83% | -12 | 0.000 ** |
| fire → sft | orbench | harmful_compliance | 1% | 6% | +6 | 0.006 ** |
| fire → sft | orbench | over_refusal | 50% | 34% | -16 | 0.000 ** |
| fire → sft | banking_hinglish | refusal_on_harmful | 47% | 44% | -2 | 0.912 |
| fire → sft | banking_hinglish | harmful_compliance | 31% | 40% | +9 | 0.498 |
| fire → sft | banking_hinglish | over_refusal | 4% | 13% | +9 | 0.130 |
| fire → dpo3 | banking_test | refusal_on_harmful | 57% | 95% | +38 | 0.000 ** |
| fire → dpo3 | banking_test | harmful_compliance | 30% | 3% | -27 | 0.000 ** |
| fire → dpo3 | banking_test | over_refusal | 2% | 2% | +0 | 1.000 |
| fire → dpo3 | banking | refusal_on_harmful | 67% | 97% | +30 | 0.000 ** |
| fire → dpo3 | banking | harmful_compliance | 23% | 1% | -21 | 0.000 ** |
| fire → dpo3 | banking | over_refusal | 1% | 2% | +1 | 0.818 |
| fire → dpo3 | indicsafe_hi | refusal_on_harmful | 58% | 60% | +2 | 0.287 |
| fire → dpo3 | indicsafe_hi | harmful_compliance | 3% | 1% | -2 | 0.077 |
| fire → dpo3 | indicsafe_hi | over_refusal | 0% | 0% | +0 | 1.000 |
| fire → dpo3 | xstest | refusal_on_harmful | 82% | 82% | +0 | 0.910 |
| fire → dpo3 | xstest | harmful_compliance | 2% | 1% | -1 | 0.276 |
| fire → dpo3 | xstest | over_refusal | 8% | 9% | +1 | 0.655 |
| fire → dpo3 | orbench | refusal_on_harmful | 94% | 90% | -4 | 0.074 |
| fire → dpo3 | orbench | harmful_compliance | 1% | 6% | +5 | 0.010 ** |
| fire → dpo3 | orbench | over_refusal | 50% | 29% | -20 | 0.000 ** |
| fire → dpo3 | banking_hinglish | refusal_on_harmful | 47% | 91% | +44 | 0.000 ** |
| fire → dpo3 | banking_hinglish | harmful_compliance | 31% | 4% | -27 | 0.001 ** |
| fire → dpo3 | banking_hinglish | over_refusal | 4% | 2% | -2 | 0.761 |
| fire → harden2_wise3 | banking_test | refusal_on_harmful | 57% | 92% | +35 | 0.000 ** |
| fire → harden2_wise3 | banking_test | harmful_compliance | 30% | 3% | -27 | 0.000 ** |
| fire → harden2_wise3 | banking_test | over_refusal | 2% | 5% | +3 | 0.444 |
| fire → harden2_wise3 | banking | refusal_on_harmful | 67% | 93% | +26 | 0.000 ** |
| fire → harden2_wise3 | banking | harmful_compliance | 23% | 2% | -21 | 0.000 ** |
| fire → harden2_wise3 | banking | over_refusal | 1% | 4% | +3 | 0.210 |
| fire → harden2_wise3 | indicsafe_hi | refusal_on_harmful | 58% | 91% | +33 | 0.000 ** |
| fire → harden2_wise3 | indicsafe_hi | harmful_compliance | 3% | 0% | -3 | 0.005 ** |
| fire → harden2_wise3 | indicsafe_hi | over_refusal | 0% | 4% | +4 | 0.030 ** |
| fire → harden2_wise3 | xstest | refusal_on_harmful | 82% | 100% | +18 | 0.000 ** |
| fire → harden2_wise3 | xstest | harmful_compliance | 2% | 0% | -2 | 0.067 |
| fire → harden2_wise3 | xstest | over_refusal | 8% | 58% | +50 | 0.000 ** |
| fire → harden2_wise3 | orbench | refusal_on_harmful | 94% | 99% | +4 | 0.004 ** |
| fire → harden2_wise3 | orbench | harmful_compliance | 1% | 0% | -0 | 0.756 |
| fire → harden2_wise3 | orbench | over_refusal | 50% | 60% | +10 | 0.003 ** |
| fire → harden2_wise3 | banking_hinglish | refusal_on_harmful | 47% | 87% | +40 | 0.000 ** |
| fire → harden2_wise3 | banking_hinglish | harmful_compliance | 31% | 2% | -29 | 0.000 ** |
| fire → harden2_wise3 | banking_hinglish | over_refusal | 4% | 4% | +0 | 1.000 |
| sft → dpo3 | banking_test | refusal_on_harmful | 43% | 95% | +52 | 0.000 ** |
| sft → dpo3 | banking_test | harmful_compliance | 40% | 3% | -37 | 0.000 ** |
| sft → dpo3 | banking_test | over_refusal | 5% | 2% | -3 | 0.435 |
| sft → dpo3 | banking | refusal_on_harmful | 52% | 97% | +45 | 0.000 ** |
| sft → dpo3 | banking | harmful_compliance | 30% | 1% | -29 | 0.000 ** |
| sft → dpo3 | banking | over_refusal | 4% | 2% | -2 | 0.392 |
| sft → dpo3 | indicsafe_hi | refusal_on_harmful | 54% | 60% | +5 | 0.061 |
| sft → dpo3 | indicsafe_hi | harmful_compliance | 4% | 1% | -3 | 0.025 ** |
| sft → dpo3 | indicsafe_hi | over_refusal | 3% | 0% | -3 | 0.068 |
| sft → dpo3 | xstest | refusal_on_harmful | 82% | 82% | +0 | 0.923 |
| sft → dpo3 | xstest | harmful_compliance | 3% | 1% | -2 | 0.204 |
| sft → dpo3 | xstest | over_refusal | 19% | 9% | -10 | 0.000 ** |
| sft → dpo3 | orbench | refusal_on_harmful | 83% | 90% | +8 | 0.008 ** |
| sft → dpo3 | orbench | harmful_compliance | 6% | 6% | -0 | 0.909 |
| sft → dpo3 | orbench | over_refusal | 34% | 29% | -5 | 0.246 |
| sft → dpo3 | banking_hinglish | refusal_on_harmful | 44% | 91% | +47 | 0.000 ** |
| sft → dpo3 | banking_hinglish | harmful_compliance | 40% | 4% | -36 | 0.000 ** |
| sft → dpo3 | banking_hinglish | over_refusal | 13% | 2% | -11 | 0.067 |
| sft → harden2_wise3 | banking_test | refusal_on_harmful | 43% | 92% | +48 | 0.000 ** |
| sft → harden2_wise3 | banking_test | harmful_compliance | 40% | 3% | -37 | 0.000 ** |
| sft → harden2_wise3 | banking_test | over_refusal | 5% | 5% | +0 | 1.000 |
| sft → harden2_wise3 | banking | refusal_on_harmful | 52% | 93% | +41 | 0.000 ** |
| sft → harden2_wise3 | banking | harmful_compliance | 30% | 2% | -28 | 0.000 ** |
| sft → harden2_wise3 | banking | over_refusal | 4% | 4% | +0 | 1.000 |
| sft → harden2_wise3 | indicsafe_hi | refusal_on_harmful | 54% | 91% | +37 | 0.000 ** |
| sft → harden2_wise3 | indicsafe_hi | harmful_compliance | 4% | 0% | -4 | 0.001 ** |
| sft → harden2_wise3 | indicsafe_hi | over_refusal | 3% | 4% | +1 | 0.843 |
| sft → harden2_wise3 | xstest | refusal_on_harmful | 82% | 100% | +18 | 0.000 ** |
| sft → harden2_wise3 | xstest | harmful_compliance | 3% | 0% | -3 | 0.024 ** |
| sft → harden2_wise3 | xstest | over_refusal | 19% | 58% | +39 | 0.000 ** |
| sft → harden2_wise3 | orbench | refusal_on_harmful | 83% | 99% | +16 | 0.000 ** |
| sft → harden2_wise3 | orbench | harmful_compliance | 6% | 0% | -6 | 0.002 ** |
| sft → harden2_wise3 | orbench | over_refusal | 34% | 60% | +26 | 0.000 ** |
| sft → harden2_wise3 | banking_hinglish | refusal_on_harmful | 44% | 87% | +42 | 0.000 ** |
| sft → harden2_wise3 | banking_hinglish | harmful_compliance | 40% | 2% | -38 | 0.000 ** |
| sft → harden2_wise3 | banking_hinglish | over_refusal | 13% | 4% | -9 | 0.195 |
