"""Build the static HF Space "Why the banking bot refuses": CircuitKIT circuits, AuditKIT before/after, examples.

  python scripts/build_space.py --runs <runs-dir> --circuits results/circuits --out space
  python scripts/build_space.py --runs none --reuse <downloaded-space> --out space   (no runs: reuse examples and circuits)
  (runs-dir holds <run>/results/gens/<tag>.jsonl with judge files, as in scripts/significance.py)
Writes space/index.html plus the circuit pages; push with `--push ashish-soni08/<space>`.
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import random
import re
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
<meta name="description" content="A small multilingual banking chatbot lost its fraud refusals when trained on banking chats. This page shows how we found the cause and fixed it, in English, Hindi and Hinglish.">
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#66645e;--card:#fff;--line:#e6e3dc;--a:#2f6f5e;--b:#c4553a;--c:#8a8578;--chip:#f1efe9;--cell:#ebe8e1;--ease-out:cubic-bezier(0.23,1,0.32,1)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161614;--fg:#ecebe6;--muted:#a3a097;--card:#1f1f1c;--line:#34332e;--a:#6cc3a8;--b:#ef8a6f;--c:#8f8b80;--chip:#2a2925;--cell:#2e2d29}}
:root[data-theme=dark]{--bg:#161614;--fg:#ecebe6;--muted:#a3a097;--card:#1f1f1c;--line:#34332e;--a:#6cc3a8;--b:#ef8a6f;--c:#8f8b80;--chip:#2a2925;--cell:#2e2d29}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:1080px;margin:0 auto;padding:40px 16px 72px}h1,h2,h3{text-wrap:balance}p{text-wrap:pretty}
h1{font-size:clamp(28px,4.2vw,40px);line-height:1.12;margin:0 0 14px;letter-spacing:-.02em}h2{font-size:22px;line-height:1.25;margin:56px 0 6px;letter-spacing:-.01em}h3{font-size:16px;margin:0 0 4px}
.eyebrow{font-size:13px;font-weight:600;color:var(--muted);letter-spacing:.02em;margin-bottom:10px}
p.lede{font-size:18px;max-width:720px;margin:0 0 14px}p.lead{color:var(--muted);font-size:14px;margin:4px 0 14px;max-width:760px}
p.takeaway{font-size:16px;max-width:760px;margin:6px 0 10px;padding-left:12px;border-left:3px solid var(--fg)}
a{color:var(--a);text-underline-offset:2px}.links{font-size:14px;color:var(--muted)}
.story{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:28px;counter-reset:s}
.story>div{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.story>div::before{counter-increment:s;content:counter(s);display:inline-grid;place-items:center;width:24px;height:24px;border-radius:50%;background:var(--fg);color:var(--bg);font-size:13px;font-weight:650;margin-bottom:8px}
.story p{margin:0;font-size:14.5px;color:var(--muted)}
.result{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.res{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;display:flex;flex-direction:column;gap:10px}
.res.win{border-color:var(--a);box-shadow:0 0 0 1px var(--a) inset}.res .who{font-weight:650}.res .who small{display:block;font-weight:400;color:var(--muted);font-size:13px}
.res .big{font-size:40px;font-weight:700;line-height:1;letter-spacing:-.02em;font-variant-numeric:tabular-nums}.res .big small{font-size:15px;font-weight:500;color:var(--muted);letter-spacing:0}
.res .sub{font-size:14px;color:var(--muted)}.res .sub b{color:var(--fg);font-variant-numeric:tabular-nums}
.waffle{display:grid;grid-template-columns:repeat(20,1fr);gap:3px;max-width:360px}.waffle i{aspect-ratio:1;border-radius:2px;background:var(--cell)}
.waffle i.r{background:var(--a)}.waffle i.h{background:var(--b)}
.legend{display:flex;gap:6px 14px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin-bottom:10px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:6px 0 12px}
.seg button{background:var(--card);color:var(--muted);border:0;padding:6px 12px;font:inherit;font-size:13px;cursor:pointer;transition:transform 160ms var(--ease-out)}
.seg button:active,nav button:active{transform:scale(.97)}.seg button[aria-pressed=true]{background:var(--fg);color:var(--bg)}
.callout{background:var(--card);border:1px solid var(--line);border-left:4px solid #c98a00;border-radius:12px;padding:16px 18px}.callout p{margin:0 0 8px;max-width:820px}.callout p:last-child{margin:0}
dl.gloss{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 28px;margin:0}dl.gloss div{padding:10px 0;border-top:1px solid var(--line)}
dl.gloss dt{font-weight:650;font-size:14.5px}dl.gloss dd{margin:2px 0 0;color:var(--muted);font-size:14px}
.divider{margin:72px 0 0;padding-top:28px;border-top:2px solid var(--fg)}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 8px}
nav button{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:7px 14px;border-radius:999px;cursor:pointer;font:inherit;font-size:14px;transition:transform 160ms var(--ease-out)}
nav button[aria-selected=true]{background:var(--fg);color:var(--bg)}section{display:none}section.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{font-weight:600;color:var(--muted);font-size:13px}.sig{color:var(--a);font-weight:600}.bad{color:var(--b)}
td small{display:block;color:var(--muted);font-size:12px;white-space:normal;max-width:330px}
tr.grp td{color:var(--muted);font-size:12.5px;font-weight:600;padding-top:16px;border-bottom:0}tr.final td{font-weight:650}
.mini{display:inline-block;height:8px;border-radius:2px;vertical-align:middle;margin-left:6px}
iframe{width:100%;height:620px;border:1px solid var(--line);border-radius:12px;background:#fff}
.ex{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.ans{background:var(--chip);border-radius:8px;padding:10px;font-size:13.5px;white-space:pre-wrap;word-break:break-word;max-height:260px;overflow:auto}
.tag{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;margin-bottom:6px;font-weight:600}
.ok{background:color-mix(in srgb,var(--a) 18%,transparent);color:var(--a)}.no{background:color-mix(in srgb,var(--b) 18%,transparent);color:var(--b)}
.q{font-weight:600;margin:4px 0 10px}.meta{font-size:12.5px;color:var(--muted)}select{font:inherit;padding:5px 8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
details.more{margin-top:10px}details.more summary{cursor:pointer;color:var(--muted);font-size:14px}
svg text{fill:var(--muted);font-size:11px}
@media(max-width:900px){.story{grid-template-columns:repeat(2,minmax(0,1fr))}.result{grid-template-columns:1fr}}
@media(max-width:760px){.story,dl.gloss,.ex{grid-template-columns:1fr}iframe{height:480px}p.lede{font-size:17px}}
</style></head><body><main>
<div class="eyebrow">Lexsi AI Safety &amp; Alignment Hackathon · Track 2b</div>
<h1>Why the banking bot refuses</h1>
<p class="lede">A bank in India wants a chatbot that answers everyday questions in English, Hindi and Hinglish, and says no
to anyone trying to commit fraud. We trained a small open model for the job. Teaching it banking made it worse at saying
no. This page shows how we found out why and fixed it, without making it refuse real customers.</p>
<div class="links"><a href="https://huggingface.co/ashish-soni08/tiny-aya-fire-banking-safe">Final model</a> ·
<a href="https://github.com/Ashish-Soni08/lexsi-safe-hinglish-banking">Code</a> ·
<a href="https://huggingface.co/datasets/ashish-soni08/hinglish-banking-safety-eval">Test questions</a></div>

<div class="story">
<div><h3>The start</h3><p>Tiny Aya Fire, a small model that speaks Indian languages and was already trained to refuse harmful requests.</p></div>
<div><h3>The problem</h3><p>After training it on 8,700 banking chats, it helped with more fraud: OTP scams, phishing scripts, mule accounts.</p></div>
<div><h3>The search</h3><p>We looked inside the model for the parts that decide to refuse, and tried repairing just those. It helped only a little.</p></div>
<div><h3>The fix</h3><p>Better training data: banking chats mixed with safety examples in all three languages, then a final round of preference training.</p></div>
</div>

<h2>Out of 100 fraud requests, how many did each model help with?</h2>
<p class="lead">Each square is one fraud request from the held-out test set (__NHARM__ requests across English, Hindi and Hinglish, never used for training or tuning).
Green means it refused, red means it helped, grey means neither clearly.</p>
<div class="seg" role="group" aria-label="Language" id="langSeg"><button aria-pressed="true" data-v="banking_test">All three languages</button><button aria-pressed="false" data-v="banking_hinglish">Hinglish only</button></div>
<div class="result" id="result"></div>

<h2>What we tried</h2>
<p class="takeaway">Repairing the weights after training barely helped, and steering refused real customers too. Fixing the training data did both jobs: fewer fraud answers and no extra refusals.</p>
<p class="lead">Fraud test set, all languages. "Answer quality" is how close its answers to normal banking questions are to reference answers (chrF++, higher is better).</p>
<div class="card">__TRIED__</div>

<h2>Where it still falls short</h2>
<div class="callout">__LIMITS__</div>

<h2>Words used on this page</h2>
<dl class="gloss">
<div><dt>Refuses harmful</dt><dd>Share of fraud or harmful requests the model declines. Higher is better.</dd></div>
<div><dt>Complies with harmful</dt><dd>Share of harmful requests where it gives real help, as judged by safety models. Lower is better.</dd></div>
<div><dt>Refuses harmless</dt><dd>Share of normal questions it wrongly refuses, like "how do I block my card?". Lower is better.</dd></div>
<div><dt>Hinglish</dt><dd>Hindi written in the Latin alphabet, mixed with English. How many people in India actually text.</dd></div>
<div><dt>Fine-tuning (SFT)</dt><dd>Training an existing model on examples for a new job, here 8,700 banking chats.</dd></div>
<div><dt>Merge</dt><dd>Averaging the weights of two models. A "70/30 merge with Fire" keeps 30% of the original model.</dd></div>
<div><dt>DPO</dt><dd>Preference training: show the model a better and a worse answer to the same question and nudge it towards the better one.</dd></div>
<div><dt>Judge panel</dt><dd>Three AI models (Qwen3Guard-Gen-8B, Llama-Guard-3-1B, Aya Expanse 8B) that label every answer. Refusal is their majority vote.</dd></div>
<div><dt>XSTest, OR-Bench, IndicSafe</dt><dd>Public test sets. XSTest and OR-Bench check refusals of harmless questions that sound dangerous; IndicSafe checks harmful requests in Hindi.</dd></div>
<div><dt>p-value</dt><dd>How likely a difference this big is by chance alone. Below 0.05 we call the change real.</dd></div>
<div><dt>Refusal circuit</dt><dd>The parts of the model that push it to refuse, found by measuring how much each part contributes to the decision.</dd></div>
<div><dt>CircuitKIT, SafeTune, CuratorKIT, AlignTune, AuditKIT</dt><dd>Lexsi's open toolkits for finding circuits, repairing weights, building data, training and auditing.</dd></div>
</dl>

<div class="divider"><div class="eyebrow">For technical readers</div><h2 style="margin-top:0">Full results</h2>
<p class="lead">Real answers from each model, every benchmark with significance tests, and the refusal circuit.</p></div>
<nav role="tablist"><button role="tab" aria-selected="true" data-t="examples">Example answers</button><button role="tab" aria-selected="false" data-t="audit">Audit</button><button role="tab" aria-selected="false" data-t="circuit">Refusal circuit</button></nav>

<section id="examples" class="on"><p class="takeaway">The same request to each model. Labels are the judge panel's verdicts.</p>
<p class="lead">Where a model helped with fraud, only the opening of its answer is shown.</p>__EXAMPLES__</section>

<section id="audit">
<p class="takeaway">__TK_AUDIT__</p>
<p class="lead">Share of prompts, judged by the 3-model panel. Banking numbers use the held-out test half only.</p>
<div class="card"><div class="legend">__LEGEND__</div>__BARS__</div>
<h2>Is the change real?</h2><p class="lead">AuditKIT paired bootstrap (5,000 resamples), same prompts for both models. <span class="sig">Green</span> = p &lt; 0.05.</p>
<div class="card">__SIGMAIN__</div>
<details class="more"><summary>Every comparison (__NSIG__ rows)</summary><div class="card">__SIG__</div></details></section>

<section id="circuit">
<p class="takeaway">__TK_CIRCUIT__</p>
<p class="lead">EAP-IG node attribution for "refuse this banking request" (harmful vs. look-alike benign banking prompts), run on each model with the same settings.
Bars show how much attribution sits in each layer; the graph below is CircuitKIT's own interactive view (large, loads when you open this tab).</p>
<div class="card"><div class="legend">__CLEGEND__</div>__LAYERS__</div>
<div class="card"><label>Circuit graph: <select id="csel">__COPTS__</select></label></div>
<iframe id="cframe" title="Circuit graph" data-src="__CFIRST__" loading="lazy"></iframe>
__COMPARE__
</section>
</main>
<script id="hero" type="application/json">__HERO__</script>
<script>
const H=JSON.parse(document.getElementById('hero').textContent);
function hero(suite){
  document.getElementById('result').innerHTML=H.models.map(([t,name,sub],i)=>{const m=H.data[t][suite]||{};
    const r=Math.round(100*(m.refusal_on_harmful||0)),h=Math.round(100*(m.harmful_compliance||0)),o=Math.round(100*(m.over_refusal||0));
    const cells=Array.from({length:100},(_,k)=>`<i class="${k<r?'r':k>=100-h?'h':''}"></i>`).join('');
    return `<div class="res${i===H.models.length-1?' win':''}"><div class="who">${name}<small>${sub}</small></div>
      <div class="big">${h}%<small> helped with fraud</small></div><div class="waffle" role="img" aria-label="${r} refused, ${h} helped, out of 100">${cells}</div>
      <div class="sub">Refused <b>${r}%</b> of fraud requests. Wrongly refused <b>${o}%</b> of normal banking questions.</div></div>`}).join('');
}
hero('banking_test');
document.querySelectorAll('#langSeg button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#langSeg button').forEach(x=>x.setAttribute('aria-pressed',x===b));hero(b.dataset.v)});
const fr=document.getElementById('cframe');
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button').forEach(x=>x.setAttribute('aria-selected',x===b));
document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id===b.dataset.t));
if(b.dataset.t==='circuit')document.querySelectorAll('#circuit iframe[data-src]').forEach(f=>{if(!f.src)f.src=f.dataset.src});});
const cs=document.getElementById('csel');if(cs)cs.onchange=()=>{fr.src=cs.value};
</script></body></html>"""

# Plain names for the "What we tried" table: (summary key, name, explanation, group).
TRIED = [
    ("fire", "Starting model (Tiny Aya Fire)", "before any banking training", "Baseline"),
    ("sft", "After banking training", "fine-tuned on 8,700 banking chats", "Baseline"),
    ("resta", "Repair: ReSta", "restore safety by weight arithmetic after training", "Repair the weights afterwards"),
    ("safemerge", "Repair: SafeMerge", "merge only the layers that stay safe", "Repair the weights afterwards"),
    ("ctheta_all", "Repair: C-ΔΘ on circuit layers", "add Fire's safety direction on the refusal circuit", "Repair the weights afterwards"),
    ("steer", "Refusal steering", "push activations towards refusing", "Repair the weights afterwards"),
    ("revert_rand8", "Undo training on 8 random layers", "control for the row below", "Use the refusal circuit"),
    ("revert_top8", "Undo training on the 8 circuit layers", "the layers CircuitKIT points to", "Use the refusal circuit"),
    ("revert_all", "Undo half the training everywhere", "50/50 merge with Fire", "Use the refusal circuit"),
    ("harden2_wise3", "Banking + safety examples, then merge", "trilingual safety mix in the training data, 70/30 merge with Fire", "Fix the training data"),
    ("__FINAL__", "Final: + preference training (DPO)", "CuratorKIT-built better/worse answer pairs", "Fix the training data"),
]

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


def sig_table(sig: list, want=(("fire", "sft"), ("sft", "final"), ("fire", "final"))) -> str:
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


def pct(v) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def tried_table(full: dict, final_tag: str) -> str:
    out = ["<table><tr><th>What we did</th><th>Refused fraud</th><th>Helped with fraud</th><th>Refused normal questions</th><th>Answer quality</th></tr>"]
    group = None
    for key, name, expl, grp in TRIED:
        k = final_tag if key == "__FINAL__" else key
        if k not in full:
            continue
        m = full[k].get("banking_test/all", {})
        if grp != group:
            out.append(f'<tr class="grp"><td colspan="5">{html.escape(grp)}</td></tr>')
            group = grp
        r, h, o = m.get("refusal_on_harmful"), m.get("harmful_compliance"), m.get("over_refusal")
        bar = lambda v, c: f'<span class="mini" style="width:{60 * (v or 0):.0f}px;background:{c}"></span>'
        chrf = full[k].get("chrf")
        out.append(f'<tr class="{"final" if key == "__FINAL__" else ""}"><td>{html.escape(name)}<small>{html.escape(expl)}</small></td>'
                   f'<td>{pct(r)}{bar(r, "var(--a)")}</td><td class="{"bad" if (h or 0) >= .2 else ""}">{pct(h)}</td>'
                   f'<td class="{"bad" if (o or 0) >= .1 else ""}">{pct(o)}</td><td>{"–" if chrf is None else f"{chrf:.1f}"}</td></tr>')
    return "".join(out) + "</table>"


def limits(summary: dict, nharm: int) -> str:
    f, fi = summary["final"], summary["fire"]
    g = lambda t, k, m: (t.get(k) or {}).get(m)
    return (f'<p><b>Hinglish is still the weakest language.</b> The final model refuses {pct(g(f, "banking_hinglish", "refusal_on_harmful"))} of Hinglish fraud requests, '
            f'against {pct(g(f, "banking_test", "refusal_on_harmful"))} across all three languages.</p>'
            f'<p><b>Harmless questions that sound dangerous still trip it up sometimes.</b> It refuses {pct(g(f, "orbench", "over_refusal"))} of OR-Bench\'s '
            f'harmless-but-scary prompts (the original Fire model: {pct(g(fi, "orbench", "over_refusal"))}) and {pct(g(f, "xstest", "over_refusal"))} of XSTest\'s '
            f'(Fire: {pct(g(fi, "xstest", "over_refusal"))}).</p>'
            f'<p><b>Outside banking it is slightly easier to talk into harm.</b> On OR-Bench\'s toxic prompts it helps with {pct(g(f, "orbench", "harmful_compliance"))}, '
            f'against {pct(g(fi, "orbench", "harmful_compliance"))} for the original model, and the difference is statistically real.</p>'
            f'<p><b>The test set is small and the labels come from AI judges.</b> {nharm} fraud requests and as many normal questions, labelled by a panel of three '
            f'models rather than people. Two of the judges agree with each other well (Cohen\'s κ 0.85), but a human review would be stronger.</p>')


def extract(old: str, start: str, end: str) -> str:
    i = old.index(start) + len(start)
    return old[i:old.index(end, i)]


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
    ap.add_argument("--reuse", default=None, help="existing space folder: reuse its examples, layer chart and circuit pages "
                    "when --runs or the circuit results are not available locally")
    args = ap.parse_args()
    runs, out, circ = pathlib.Path(args.runs), pathlib.Path(args.out), pathlib.Path(args.circuits)
    out.mkdir(parents=True, exist_ok=True)

    full = json.loads((ROOT / "results" / "summary.json").read_text())
    summary = full
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
    old = None
    if args.reuse:
        old_dir = pathlib.Path(args.reuse)
        old = (old_dir / "index.html").read_text()

    copts, cfirst, layers, compare = "", "", "<p class='meta'>Circuit results not available.</p>", ""
    if (circ / "layer_mass.json").exists():
        layers = layers_chart(json.loads((circ / "layer_mass.json").read_text()))
    elif old and "Attribution mass per layer" in old:
        layers = re.search(r'<svg[^>]*aria-label="Attribution mass per layer".*?</svg>', old, re.S).group(0)
    for t, n, _ in MODELS:
        f = circ / f"{t}_circuit.html"
        if not f.exists() and args.reuse:
            f = pathlib.Path(args.reuse) / f"{t}_circuit.html"
        if f.exists():
            if f.resolve() != (out / f.name).resolve():
                shutil.copy(f, out / f.name)
            copts += f'<option value="{f.name}">{n}</option>'
            cfirst = cfirst or f.name
    cmp_src = circ / "comparison.html"
    if not cmp_src.exists() and args.reuse:
        cmp_src = pathlib.Path(args.reuse) / "comparison.html"
    if cmp_src.exists():
        if cmp_src.resolve() != (out / "comparison.html").resolve():
            shutil.copy(cmp_src, out / "comparison.html")
        compare = ('<h2>How similar are the circuits?</h2><p class="lead">CircuitKIT ComparisonDashboard over the three models\' node scores.</p>'
                   '<iframe title="Circuit comparison" data-src="comparison.html" loading="lazy"></iframe>')

    if (runs / args.final).exists():
        ex_html = examples(runs, files)
    elif old:
        ex_html = extract(old, '<section id="examples">', "</section>")
        ex_html = ex_html[ex_html.index('<div class="card">'):]
    else:
        raise SystemExit("no generations under --runs and no --reuse folder for the examples")

    nharm = next((r["n"] for r in sig if r["suite"] == "banking_test"), 60)
    hero = {"models": [["fire", "Starting model", "Tiny Aya Fire"], ["sft", "After banking training", "Banking SFT"],
                       ["final", "Final model", "Safety data + merge + DPO"]],
            "data": {t: {k: summary[t].get(k, {}) for k in ("banking_test", "banking_hinglish")} for t in summary}}
    f, s0, fi = summary["final"]["banking_test"], summary["sft"]["banking_test"], summary["fire"]["banking_test"]
    ob, obf = summary["final"]["orbench"], summary["fire"]["orbench"]
    tk_audit = (f"On the banking fraud test, banking training dropped refusals from {pct(fi['refusal_on_harmful'])} to {pct(s0['refusal_on_harmful'])}; "
                f"the final model refuses {pct(f['refusal_on_harmful'])}. On the public benchmarks it matches the original model, with one exception: "
                f"on OR-Bench it helps with more toxic prompts ({pct(obf['harmful_compliance'])} to {pct(ob['harmful_compliance'])}) while refusing far fewer harmless ones "
                f"({pct(obf['over_refusal'])} to {pct(ob['over_refusal'])}).")
    rt, rr = full.get("revert_top8", {}).get("banking_test/all", {}), full.get("revert_rand8", {}).get("banking_test/all", {})
    tk_circuit = (f"Undoing the banking training on just the 8 layers this circuit points to brought fraud refusals back to {pct(rt.get('refusal_on_harmful'))}, "
                  f"against {pct(rr.get('refusal_on_harmful'))} for 8 random layers. The circuit finds real refusal machinery, but repairing it alone was not enough.")
    sig_main = sig_table(sig, (("fire", "final"),))

    page = (PAGE.replace("__LEGEND__", legend).replace("__CLEGEND__", legend).replace("__BARS__", bars(summary))
            .replace("__SIGMAIN__", sig_main).replace("__NSIG__", str(sig_table(sig).count("<tr>") - 1))
            .replace("__SIG__", sig_table(sig)).replace("__LAYERS__", layers).replace("__COPTS__", copts)
            .replace("__CFIRST__", cfirst or "about:blank").replace("__COMPARE__", compare)
            .replace("__TRIED__", tried_table(full, args.final_tag)).replace("__LIMITS__", limits(summary, nharm))
            .replace("__NHARM__", str(nharm)).replace("__TK_AUDIT__", tk_audit).replace("__TK_CIRCUIT__", tk_circuit)
            .replace("__HERO__", json.dumps(hero).replace("</", "<\\/"))
            .replace("__EXAMPLES__", ex_html))
    (out / "index.html").write_text(page)
    (out / "README.md").write_text("---\ntitle: Why the banking bot refuses\nemoji: 🏦\ncolorFrom: green\ncolorTo: gray\nsdk: static\npinned: false\n"
                                   "license: cc-by-nc-4.0\nshort_description: Circuits and audit of a safe Hinglish banking bot\n---\n")
    print("wrote", sorted(p.name for p in out.iterdir()))
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, repo_type="space", space_sdk="static", exist_ok=True, private=False)
        api.upload_folder(repo_id=args.push, repo_type="space", folder_path=str(out), commit_message="Update dashboard")
        print("pushed", f"https://huggingface.co/spaces/{args.push}")


if __name__ == "__main__":
    main()
