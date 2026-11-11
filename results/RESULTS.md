| Model | Banking test refuse↑ / comply↓ / over-refuse↓ | Banking Hinglish refuse↑ / comply↓ / over-refuse↓ | IndicSafe Hindi refuse↑ / comply↓ / over-refuse↓ | XSTest refuse↑ / comply↓ / over-refuse↓ | OR-Bench refuse↑ / comply↓ / over-refuse↓ | Banking chrF++ |
|---|---|---|---|---|---|---|
| `fire` Tiny Aya Fire (aligned starting point) | 57 / 30 / 2 | 47 / 31 / 4 | 58 / 3 / 0 | 82 / 2 / 8 | 94 / 1 / 50 | 22.19 |
| `sft` LoRA SFT on trilingual Bitext banking data | 43 / 40 / 5 | 44 / 40 / 13 | 54 / 4 / 3 | 82 / 3 / 19 | 83 / 6 / 34 | 48.84 |
| `resta` SFT + ReSta (alpha tuned on dev) | 50 / 33 / 7 | 47 / 36 / 9 | 54 / 3 / 6 | 73 / 10 / 20 | 76 / 12 / 24 | 44.96 |
| `safemerge` SFT + SafeMerge | 40 / 37 / 2 | 31 / 53 / 2 | 57 / 2 / 0 | 80 / 3 / 26 | 86 / 6 / 44 | 47.7 |
| `ctheta_all` SFT + C-ΔΘ (Fire − Base) on all circuit layers | 35 / 40 / 3 | 22 / 60 / 2 | 49 / 2 / 6 | 88 / 2 / 32 | 89 / 4 / 56 | 44.46 |
| `ctheta_top8` SFT + C-ΔΘ (Fire − Base) on the top-8 circuit layers, MLP | 45 / 37 / 8 | 56 / 40 / 16 | 56 / 4 / 6 | 82 / 2 / 28 | 90 / 4 / 50 | 42.72 |
| `steer` SFT + refusal-direction steering | 92 / 2 / 18 | 87 / 2 / 36 | 84 / 0 / 8 | 96 / 0 / 71 | 99 / 0 / 91 | – |
| `revert_top8` SFT reverted toward Fire on the top-8 circuit layers | 52 / 37 / 2 | 38 / 51 / 4 | 62 / 2 / 0 | 84 / 4 / 25 | 88 / 6 / 39 | 43.56 |
| `revert_rand8` SFT reverted toward Fire on 8 random layers (control) | 42 / 43 / 2 | 33 / 60 / 2 | 52 / 4 / 0 | 90 / 1 / 37 | 91 / 4 / 58 | 45.16 |
| `revert_all` SFT reverted toward Fire on all layers (WiSE 0.5) | 73 / 13 / 0 | 49 / 33 / 0 | 66 / 1 / 0 | 88 / 1 / 27 | 94 / 4 / 57 | 40.57 |
| `harden` SFT on Bitext + safety mix v1 | 100 / 0 / 22 | 100 / 0 / 16 | 99 / 0 / 46 | 100 / 0 / 91 | 100 / 0 / 100 | 48.47 |
| `harden_wise` harden v1, 50/50 merge with Fire | 87 / 3 / 0 | 73 / 11 / 0 | 88 / 0 / 2 | 99 / 0 / 52 | 100 / 0 / 92 | 39.33 |
| `harden2` SFT on Bitext + safety mix v2 (adds edgy-but-safe examples) | 98 / 2 / 7 | 100 / 0 / 4 | 97 / 0 / 18 | 100 / 0 / 71 | 99 / 0 / 62 | 48.38 |
| `harden2_wise` harden v2, 50/50 merge with Fire | 82 / 12 / 0 | 76 / 16 / 2 | 81 / 0 / 0 | 96 / 0 / 36 | 99 / 0 / 60 | 39.39 |
| `harden2_wise3` harden v2, 70/30 merge with Fire | 92 / 3 / 5 | 87 / 2 / 4 | 91 / 0 / 4 | 100 / 0 / 58 | 99 / 0 / 60 | 44.88 |
| `harden3` SFT on Bitext + safety mix v3 (PolyGuardMix, Nemotron Hindi) | 80 / 12 / 3 | 64 / 31 / 7 | 68 / 2 / 1 | 92 / 1 / 15 | 98 / 1 / 63 | 48.23 |
| `harden3_wise3` harden v3, 70/30 merge with Fire | 77 / 15 / 3 | 64 / 27 / 2 | 67 / 1 / 1 | 90 / 2 / 15 | 98 / 0 / 63 | 45.36 |
| `dpo` harden2_wise3 + DPO on CuratorKIT pairs, round 1 (113 pairs) | 92 / 3 / 5 | 89 / 2 / 4 | 91 / 0 / 4 | 100 / 0 / 57 | 99 / 0 / 59 | 44.82 |
| `dpo2` harden2_wise3 + DPO on CuratorKIT pairs, round 2 (932 pairs) | 98 / 0 / 10 | 96 / 2 / 11 | 77 / 1 / 2 | 98 / 0 / 39 | 94 / 2 / 34 | 37.26 |
| `dpo2_mix05` DPO round 2, 50/50 merge with harden2_wise3 | 100 / 0 / 8 | 100 / 0 / 13 | 85 / 0 / 2 | 98 / 0 / 48 | 98 / 0 / 50 | 41.99 |
| `dpo3` harden3_wise3 + DPO on CuratorKIT pairs + banking anchors, round 3 | 95 / 3 / 2 | 91 / 4 / 2 | 60 / 1 / 0 | 82 / 1 / 9 | 90 / 6 / 29 | 48.73 |
| `dpo3_mix05` DPO round 3, 50/50 merge with harden3_wise3 | 93 / 5 / 0 | 82 / 11 / 2 | 62 / 2 / 0 | 85 / 0 / 10 | 94 / 2 / 44 | 47.67 |
