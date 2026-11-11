"""Build the static HF Space "Why the banking bot refuses": CircuitKIT circuits, AuditKIT before/after, examples.

  python scripts/build_space.py --runs <runs-dir> --circuits results/circuits --out space
  (runs-dir holds <run>/results/gens/<tag>.jsonl with judge files, as in scripts/significance.py)
Writes space/index.html plus the circuit pages; push with `--push ashish-soni08/<space>`.
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import random
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lexsi_a.judge import aggregate  # noqa: E402

MODELS = [("fire", "Tiny Aya Fire", "repair"), ("sft", "Banking SFT", "repair"), ("final", "Final (safe)", None)]
SUITES = [("banking_test", "Banking fraud (test)"), ("banking_hinglish", "Banking, Hinglish"),
          ("indicsafe_hi", "IndicSafe Hindi"), ("xstest", "XSTest"), ("orbench", "OR-Bench")]
METRICS = [("refusal_on_harmful", "Refuses harmful", True), ("harmful_compliance", "Complies with harmful", False),
           ("over_refusal", "Refuses harmless", False)]

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Why the banking bot refuses</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#66645e;--card:#fff;--line:#e6e3dc;--a:#2f6f5e;--b:#c4553a;--c:#8a8578;--chip:#f1efe9}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a3a097;--card:#1f1f1c;--line:#34332e;--a:#6cc3a8;--b:#ef8a6f;--c:#8f8b80;--chip:#2a2925}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 64px}h1{font-size:28px;margin:0 0 6px}h2{font-size:20px;margin:36px 0 8px}
p.lead{color:var(--muted);margin:0 0 18px;max-width:760px}nav{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}
nav button{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:7px 14px;border-radius:999px;cursor:pointer;font:inherit}
nav button[aria-selected=true]{background:var(--fg);color:var(--bg)}section{display:none}section.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{font-weight:600;color:var(--muted);font-size:13px}.sig{color:var(--a);font-weight:600}.bad{color:var(--b)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--muted)}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
iframe{width:100%;height:620px;border:1px solid var(--line);border-radius:12px;background:#fff}
.ex{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}@media(max-width:760px){.ex{grid-template-columns:1fr}iframe{height:480px}}
.ans{background:var(--chip);border-radius:8px;padding:10px;font-size:13.5px;white-space:pre-wrap;word-break:break-word;max-height:260px;overflow:auto}
.tag{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;margin-bottom:6px;font-weight:600}
.ok{background:color-mix(in srgb,var(--a) 18%,transparent);color:var(--a)}.no{background:color-mix(in srgb,var(--b) 18%,transparent);color:var(--b)}
.q{font-weight:600;margin:4px 0 10px}.meta{font-size:12.5px;color:var(--muted)}select{font:inherit;padding:5px 8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
svg text{fill:var(--muted);font-size:11px}
</style></head><body><main>
<h1>Why the banking bot refuses</h1>
<p class="lead">A Tiny Aya Fire retail-banking assistant for English, Hindi and Hinglish. Fine-tuning on banking chats wore away its
fraud refusals; we found where refusal lives with <b>CircuitKIT</b>, tried <b>SafeTune</b> repairs, rebuilt the data with
<b>CuratorKIT</b>, trained with <b>AlignTune</b>, and checked every claim with <b>AuditKIT</b>.
<a href="https://huggingface.co/ashish-soni08/tiny-aya-fire-banking-safe">Model</a> ·
<a href="https://github.com/Ashish-Soni08/lexsi-safe-hinglish-banking">Code</a></p>
<nav role="tablist"><button aria-selected="true" data-t="audit">Audit</button><button aria-selected="false" data-t="circuit">Refusal circuit</button><button aria-selected="false" data-t="examples">Examples</button></nav>

<section id="audit" class="on">
<p class="lead">Share of prompts, judged by a 3-model panel (Qwen3Guard-Gen-8B, Llama-Guard-3-1B, Aya Expanse 8B rubric). Banking numbers use the held-out test half only.</p>
<div class="card"><div class="legend">__LEGEND__</div>__BARS__</div>
<h2>Is the change real?</h2><p class="lead">AuditKIT paired bootstrap (5,000 resamples), same prompts for both models. <span class="sig">Green</span> = p &lt; 0.05.</p>
<div class="card">__SIG__</div></section>

<section id="circuit">
<p class="lead">EAP-IG node attribution for "refuse this banking request" (harmful vs. look-alike benign banking prompts), run on each model with the same settings.
Bars show how much attribution sits in each layer; the graph is CircuitKIT's own interactive view.</p>
<div class="card"><div class="legend">__CLEGEND__</div>__LAYERS__</div>
<div class="card"><label>Circuit graph: <select id="csel">__COPTS__</select></label></div>
<iframe id="cframe" title="Circuit graph" src="__CFIRST__"></iframe>
__COMPARE__
</section>

<section id="examples"><p class="lead">The same request to each model. Labels are the judge panel's verdicts.</p>__EXAMPLES__</section>
</main><script>
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button').forEach(x=>x.setAttribute('aria-selected',x===b));
document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id===b.dataset.t))});
const cs=document.getElementById('csel');if(cs)cs.onchange=()=>document.getElementById('cframe').src=cs.value;
</script></body></html>"""

COLORS = ["var(--c)", "var(--b)", "var(--a)"]


def bars(summary: dict) -> str:
    # One small grouped bar chart per (suite, metric): three bars, Fire / SFT / Final.
    rows = []
    for key, name in SUITES:
        cells = []
        for mkey, mname, good_high in METRICS:
            vals = [summary[t].get(key, {}).get(mkey) for t, _, _ in MODELS]
            w, h = 150, 86
            svg = [f'<svg viewBox="0 0 {w} {h + 18}" width="{w}" height="{h + 18}" role="img" aria-label="{name}: {mname}">']
            for i, v in enumerate(vals):
                if v is None:
                    continue
                bh = max(1, v * h)
                x = 12 + i * 46
                svg.append(f'<rect x="{x}" y="{h - bh}" width="34" height="{bh}" rx="3" fill="{COLORS[i]}"><title>{MODELS[i][1]}: {100 * v:.0f}%</title></rect>')
                svg.append(f'<text x="{x + 17}" y="{h + 13}" text-anchor="middle">{100 * v:.0f}%</text>')
            svg.append("</svg>")
            arrow = "higher is better" if good_high else "lower is better"
            cells.append(f'<td style="text-align:center"><div class="meta">{mname} ({arrow})</div>{"".join(svg)}</td>')
        rows.append(f'<tr><td><b>{name}</b></td>{"".join(cells)}</tr>')
    return "<table>" + "".join(rows) + "</table>"


def sig_table(sig: list) -> str:
    want = [("fire", "sft"), ("sft", "final"), ("fire", "final")]
    names = {t: n for t, n, _ in MODELS}
    out = ["<table><tr><th>Comparison</th><th>Suite</th><th>Metric</th><th>Before</th><th>After</th><th>Change</th><th>p</th></tr>"]
    for b, c in want:
        for r in sig:
            if (r["baseline"], r["candidate"]) == (b, c) and r["suite"] in dict(SUITES):
                cls = "sig" if r["p_value"] < 0.05 else ""
                out.append(f'<tr><td>{names[b]} → {names[c]}</td><td>{dict(SUITES)[r["suite"]]}</td><td>{dict((m, n) for m, n, _ in METRICS)[r["metric"]]}</td>'
                           f'<td>{100 * r["mean_baseline"]:.0f}%</td><td>{100 * r["mean_candidate"]:.0f}%</td>'
                           f'<td>{100 * r["delta"]:+.0f}</td><td class="{cls}">{r["p_value"]:.3f}</td></tr>')
    return "".join(out) + "</table>"


def layers_chart(mass: dict) -> str:
    tags = [t for t, _, _ in MODELS if t in mass]
    n_layers = 1 + max(int(k) for t in tags for k in mass[t])
    tot = {t: sum(v["attn"] + v["mlp"] for v in mass[t].values()) or 1 for t in tags}
    w, h, pad = 980, 160, 24
    bw = (w - pad) / n_layers
    svg = [f'<svg viewBox="0 0 {w} {h + 22}" width="100%" role="img" aria-label="Attribution mass per layer">']
    top = max(((mass[t].get(str(l), {"attn": 0, "mlp": 0})["attn"] + mass[t].get(str(l), {"attn": 0, "mlp": 0})["mlp"]) / tot[t])
              for t in tags for l in range(n_layers)) or 1
    for l in range(n_layers):
        for i, t in enumerate(tags):
            v = mass[t].get(str(l), {"attn": 0, "mlp": 0})
            frac = (v["attn"] + v["mlp"]) / tot[t]
            bh = frac / top * h
            x = pad + l * bw + i * (bw / len(tags))
            svg.append(f'<rect x="{x:.1f}" y="{h - bh:.1f}" width="{bw / len(tags) - 1:.1f}" height="{bh:.1f}" fill="{COLORS[i]}">'
                       f'<title>layer {l}, {dict((a, b) for a, b, _ in MODELS)[t]}: {100 * frac:.1f}% of attribution</title></rect>')
        if l % 5 == 0:
            svg.append(f'<text x="{pad + l * bw:.1f}" y="{h + 15}">L{l}</text>')
    svg.append("</svg>")
    return "".join(svg)


def examples(runs: pathlib.Path, files: dict) -> str:
    rows = {t: {r["id"]: r for r in aggregate(str(f))} for t, f in files.items()}
    ids = [i for i in rows["final"] if i.startswith(("harmful-", "benign-")) and int(i.split("-")[1]) >= 25]
    rng = random.Random(1)
    picks = []
    for label, lang in [("harmful", "hinglish"), ("harmful", "hi"), ("harmful", "en"), ("benign", "hinglish"), ("benign", "en"), ("benign", "hi")]:
        cand = [i for i in ids if i.startswith(label) and i.endswith(lang)]
        # Prefer prompts where the models disagree (the interesting cases).
        dis = [i for i in cand if len({rows[t][i]["refused"] for t in rows if i in rows[t]}) > 1]
        picks.append(rng.choice(dis or cand))
    out = []
    for i in picks:
        r = rows["final"][i]
        harmful = i.startswith("harmful")
        cells = []
        for t, name, _ in MODELS:
            a = rows[t].get(i)
            if not a:
                continue
            good = a["refused"] if harmful else not a["refused"]
            verdict = ("Refused" if a["refused"] else "Complied" if harmful else "Answered")
            # Harmful compliance is shown only as an opening excerpt: enough to see the failure, not a usable script.
            text = a["answer"][:220] + " … [cut]" if (harmful and not a["refused"]) else a["answer"][:900]
            cells.append(f'<div><span class="tag {"ok" if good else "no"}">{name}: {verdict}</span>'
                         f'<div class="ans">{html.escape(text)}</div></div>')
        kind = "Fraud request" if harmful else "Legitimate question"
        out.append(f'<div class="card"><div class="meta">{kind} · {r["lang"]}</div><div class="q">{html.escape(r["prompt"])}</div>'
                   f'<div class="ex">{"".join(cells)}</div></div>')
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--final", required=True, help="gens file of the final model, e.g. dpo3e/results/gens/dpo3.jsonl")
    ap.add_argument("--final-tag", default="dpo3")
    ap.add_argument("--circuits", default="results/circuits")
    ap.add_argument("--out", default="space")
    ap.add_argument("--push", default=None)
    args = ap.parse_args()
    runs, out, circ = pathlib.Path(args.runs), pathlib.Path(args.out), pathlib.Path(args.circuits)
    out.mkdir(parents=True, exist_ok=True)

    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    summary = {"fire": summary["fire"], "sft": summary["sft"], "final": summary[args.final_tag]}
    for t in summary:  # summary.json has banking/hinglish under that name
        summary[t]["banking_hinglish"] = summary[t].get("banking/hinglish") or summary[t].get("banking_hinglish") or {}
        for key, _ in SUITES:
            summary[t].setdefault(key, summary[t].get(f"{key}/all", {}))
    sig = [dict(r, candidate="final" if r["candidate"] == args.final_tag else r["candidate"])
           for r in json.loads((ROOT / "results" / "significance.json").read_text())]

    files = {"fire": runs / "repair/results/gens/fire.jsonl", "sft": runs / "repair/results/gens/sft.jsonl",
             "final": runs / args.final}
    legend = "".join(f'<span><i class="sw" style="background:{COLORS[i]}"></i>{n}</span>' for i, (_, n, _) in enumerate(MODELS))

    copts, cfirst, layers, compare = "", "", "<p class='meta'>Circuit results not available.</p>", ""
    if (circ / "layer_mass.json").exists():
        layers = layers_chart(json.loads((circ / "layer_mass.json").read_text()))
        for t, n, _ in MODELS:
            f = circ / f"{t}_circuit.html"
            if f.exists():
                shutil.copy(f, out / f.name)
                copts += f'<option value="{f.name}">{n}</option>'
                cfirst = cfirst or f.name
        if (circ / "comparison.html").exists():
            shutil.copy(circ / "comparison.html", out / "comparison.html")
            compare = '<h2>How similar are the circuits?</h2><p class="lead">CircuitKIT ComparisonDashboard over the three models\' node scores.</p><iframe title="Circuit comparison" src="comparison.html"></iframe>'

    page = (PAGE.replace("__LEGEND__", legend).replace("__CLEGEND__", legend).replace("__BARS__", bars(summary))
            .replace("__SIG__", sig_table(sig)).replace("__LAYERS__", layers).replace("__COPTS__", copts)
            .replace("__CFIRST__", cfirst or "about:blank").replace("__COMPARE__", compare)
            .replace("__EXAMPLES__", examples(runs, files)))
    (out / "index.html").write_text(page)
    (out / "README.md").write_text("---\ntitle: Why the banking bot refuses\nemoji: 🏦\ncolorFrom: green\ncolorTo: gray\nsdk: static\npinned: false\n"
                                   "license: cc-by-nc-4.0\nshort_description: Interpretability and audit of a safe Hinglish banking assistant\n---\n")
    print("wrote", sorted(p.name for p in out.iterdir()))
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, repo_type="space", space_sdk="static", exist_ok=True, private=False)
        api.upload_folder(repo_id=args.push, repo_type="space", folder_path=str(out), commit_message="Update dashboard")
        print("pushed", f"https://huggingface.co/spaces/{args.push}")


if __name__ == "__main__":
    main()
