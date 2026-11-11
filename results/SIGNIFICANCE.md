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
| fire → revert_all | banking_test | refusal_on_harmful | 57% | 73% | +17 | 0.007 ** |
| fire → revert_all | banking_test | harmful_compliance | 30% | 13% | -17 | 0.018 ** |
| fire → revert_all | banking_test | over_refusal | 2% | 0% | -2 | 0.621 |
| fire → revert_all | banking | refusal_on_harmful | 67% | 74% | +7 | 0.109 |
| fire → revert_all | banking | harmful_compliance | 23% | 14% | -9 | 0.048 ** |
| fire → revert_all | banking | over_refusal | 1% | 1% | -1 | 0.766 |
| fire → revert_all | indicsafe_hi | refusal_on_harmful | 58% | 66% | +9 | 0.000 ** |
| fire → revert_all | indicsafe_hi | harmful_compliance | 3% | 1% | -2 | 0.029 ** |
| fire → revert_all | indicsafe_hi | over_refusal | 0% | 0% | +0 | 1.000 |
| fire → revert_all | xstest | refusal_on_harmful | 82% | 88% | +6 | 0.015 ** |
| fire → revert_all | xstest | harmful_compliance | 2% | 1% | -1 | 0.441 |
| fire → revert_all | xstest | over_refusal | 8% | 27% | +19 | 0.000 ** |
| fire → revert_all | orbench | refusal_on_harmful | 94% | 94% | -1 | 0.659 |
| fire → revert_all | orbench | harmful_compliance | 1% | 4% | +2 | 0.129 |
| fire → revert_all | orbench | over_refusal | 50% | 57% | +8 | 0.022 ** |
| fire → revert_all | banking_hinglish | refusal_on_harmful | 47% | 49% | +2 | 0.904 |
| fire → revert_all | banking_hinglish | harmful_compliance | 31% | 33% | +2 | 0.910 |
| fire → revert_all | banking_hinglish | over_refusal | 4% | 0% | -4 | 0.274 |
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
| sft → revert_all | banking_test | refusal_on_harmful | 43% | 73% | +30 | 0.000 ** |
| sft → revert_all | banking_test | harmful_compliance | 40% | 13% | -27 | 0.000 ** |
| sft → revert_all | banking_test | over_refusal | 5% | 0% | -5 | 0.114 |
| sft → revert_all | banking | refusal_on_harmful | 52% | 74% | +22 | 0.000 ** |
| sft → revert_all | banking | harmful_compliance | 30% | 14% | -16 | 0.001 ** |
| sft → revert_all | banking | over_refusal | 4% | 1% | -4 | 0.075 |
| sft → revert_all | indicsafe_hi | refusal_on_harmful | 54% | 66% | +12 | 0.000 ** |
| sft → revert_all | indicsafe_hi | harmful_compliance | 4% | 1% | -3 | 0.006 ** |
| sft → revert_all | indicsafe_hi | over_refusal | 3% | 0% | -3 | 0.068 |
| sft → revert_all | xstest | refusal_on_harmful | 82% | 88% | +6 | 0.005 ** |
| sft → revert_all | xstest | harmful_compliance | 3% | 1% | -2 | 0.204 |
| sft → revert_all | xstest | over_refusal | 19% | 27% | +8 | 0.002 ** |
| sft → revert_all | orbench | refusal_on_harmful | 83% | 94% | +10 | 0.000 ** |
| sft → revert_all | orbench | harmful_compliance | 6% | 4% | -3 | 0.197 |
| sft → revert_all | orbench | over_refusal | 34% | 57% | +23 | 0.000 ** |
| sft → revert_all | banking_hinglish | refusal_on_harmful | 44% | 49% | +4 | 0.632 |
| sft → revert_all | banking_hinglish | harmful_compliance | 40% | 33% | -7 | 0.522 |
| sft → revert_all | banking_hinglish | over_refusal | 13% | 0% | -13 | 0.016 ** |
