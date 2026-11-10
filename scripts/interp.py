"""Linear weight interpolation (WiSE-FT style): theta = (1 - s) * theta_a + s * theta_b.

  python scripts/interp.py --a /tmp/sft --b CohereLabs/tiny-aya-fire --s 0.5 --out /tmp/merged
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--s", type=float, default=0.5)
ap.add_argument("--out", required=True)
args = ap.parse_args()

a = AutoModelForCausalLM.from_pretrained(args.a, dtype=torch.bfloat16)
b = AutoModelForCausalLM.from_pretrained(args.b, dtype=torch.bfloat16).state_dict()
with torch.no_grad():
    for k, v in a.state_dict().items():
        if v.is_floating_point():
            v.copy_(((1 - args.s) * v.float() + args.s * b[k].float()).to(v.dtype))
a.save_pretrained(args.out)
AutoTokenizer.from_pretrained(args.a).save_pretrained(args.out)
print("saved", args.out)
