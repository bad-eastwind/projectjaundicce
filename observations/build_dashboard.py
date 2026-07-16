"""Build a single self-contained observations/dashboard.html.

Both figure themes are inlined as data URIs and swapped by CSS, so the page has no
external requests and reads correctly in light or dark.

    python observations/build_dashboard.py
"""

from __future__ import annotations

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent


def uri(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


FIGS = [
    ("fig3_counterfactual", "Illuminant invariance is the result that holds",
     "The one publication-grade finding. The intervention genuinely moves the input "
     "(raw skin b* swings 12.5 units) and the model barely moves: 5.5x less prediction "
     "swing and 5.8x fewer class flips than ERM. The melanin arm is a real non-result, "
     "not a metric artifact."),
    ("fig2_paired_delta", "No method beats ERM once seeds are paired",
     "Pairing each method against ERM on the same seed and split removes seed variance — "
     "the most favourable honest test available. Every 95% CI still crosses zero. The best "
     "cell is ours (disentangle) pooled over domains 0+1: +0.012 AUC, p=0.13."),
    ("fig1_generalization", "The full grid, with every seed shown",
     "Methods are separated by far less than the spread across seeds. Bar-and-errorbar "
     "plotting would hide this; at n=3 the individual seeds are the honest presentation."),
    ("fig7_power", "The grid was never powered to resolve this",
     "At the effect size actually observed, 3 seeds gives 6-23% power. Another 3-seed grid "
     "changes nothing. One cell — ours vs ERM on domain 0 — needs ~8 seeds and is affordable."),
    ("fig5_fairness_instability", "The fairness metric measures the model, not the babies",
     "ITA is read off each run's own white-balanced image under its own attention, so the "
     "same 119 test images get re-stratified by every run. Same architecture, same images, "
     "different seed: 73 'very light' vs 18. The protected attribute is a function of the "
     "model being audited."),
    ("fig4_ablation", "Ablations are directional at best, and one column is an artifact",
     "Single seed. Every reported fairness gap is set by a stratum of 2-10 images — "
     "ours_full's 0.500 is one bucket of two babies. Recomputed over strata with n>=20 it "
     "is 0.013. The 'removing the causal adversary improves fairness' reading does not survive."),
    ("fig6_domain_confound", "Domain-2 LOCO cannot be cited",
     "Pseudo-domain positive rates spread 0.28 — the cluster boundary partly is the label. "
     "Every method including plain ERM scores .979-.998 held out on domain 2."),
    ("fig8_operating_points", "Operating points behave as designed",
     "The screening threshold trades specificity for recall as intended. IRM is the "
     "recall-first extreme: 0.988 sensitivity at 0.757 specificity."),
    ("fig9_training_curves", "Training converges fast, then early-stops",
     "Physics runs early-stop sooner (28-29 epochs vs 35-36 for the DG baselines)."),
]

CARDS = [
    ("Runs analysed", "68", "5 methods x 4 splits x 3 seeds, + 6 ablations + 2 flagship"),
    ("Cells beating ERM at p&lt;0.05", "0", "every 95% CI crosses zero"),
    ("Power at 3 seeds", "6-23%", "at the effect size actually observed"),
    ("Illuminant swing vs ERM", "5.5x lower", "0.013 vs 0.073 — the result that holds"),
]


def main() -> None:
    figs = "\n".join(
        f"""  <figure class="fig">
    <figcaption><h2>{t}</h2><p>{d}</p></figcaption>
    <img class="light-only" src="{uri(HERE / 'figures' / 'light' / (n + '.png'))}" alt="{t}">
    <img class="dark-only" src="{uri(HERE / 'figures' / 'dark' / (n + '.png'))}" alt="{t}">
  </figure>"""
        for n, t, d in FIGS
    )
    cards = "\n".join(
        f'    <div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="s">{s}</div></div>'
        for k, v, s in CARDS
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project Jaundice — training grid observations</title>
<style>
  :root {{
    color-scheme: light dark;
    --surface: #fcfcfb; --plane: #f9f9f7;
    --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
    --rule: #e1e0d9; --accent: #2a78d6; --good: #0ca30c; --crit: #d03b3b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --surface: #1a1a19; --plane: #0d0d0d;
      --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
      --rule: #2c2c2a; --accent: #3987e5;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0 1.5rem 5rem;
    background: var(--plane); color: var(--ink);
    font: 15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  main {{ max-width: 1100px; margin: 0 auto; }}
  header {{ padding: 3.5rem 0 2rem; border-bottom: 1px solid var(--rule); }}
  h1 {{ font-size: 1.9rem; margin: 0 0 .4rem; letter-spacing: -0.02em; }}
  .sub {{ color: var(--ink2); margin: 0; }}
  .prov {{ color: var(--muted); font-size: .8rem; margin-top: .9rem;
           font-variant-numeric: tabular-nums; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: .75rem; margin: 2rem 0; }}
  .card {{ background: var(--surface); border: 1px solid var(--rule);
           border-radius: 10px; padding: 1rem; }}
  .card .k {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
              color: var(--muted); }}
  .card .v {{ font-size: 1.7rem; font-weight: 600; margin: .3rem 0 .2rem; }}
  .card .s {{ font-size: .78rem; color: var(--ink2); }}
  .lede {{ background: var(--surface); border: 1px solid var(--rule);
           border-left: 3px solid var(--accent); border-radius: 10px;
           padding: 1.1rem 1.3rem; margin: 2rem 0; }}
  .lede p {{ margin: 0 0 .6rem; }} .lede p:last-child {{ margin: 0; }}
  .fig {{ margin: 3rem 0 0; padding-top: 2rem; border-top: 1px solid var(--rule); }}
  .fig h2 {{ font-size: 1.15rem; margin: 0 0 .3rem; letter-spacing: -0.01em; }}
  .fig figcaption p {{ color: var(--ink2); margin: 0 0 1.1rem; max-width: 78ch;
                        font-size: .9rem; }}
  .fig img {{ width: 100%; height: auto; display: block;
              background: var(--surface); border: 1px solid var(--rule);
              border-radius: 10px; }}
  .dark-only {{ display: none; }}
  @media (prefers-color-scheme: dark) {{
    .light-only {{ display: none; }}
    .dark-only {{ display: block; }}
  }}
  footer {{ margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--rule);
            color: var(--muted); font-size: .82rem; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85em;
          background: var(--surface); border: 1px solid var(--rule);
          border-radius: 4px; padding: .1em .35em; }}
  strong.good {{ color: var(--good); }} strong.crit {{ color: var(--crit); }}
</style>
</head>
<body>
<main>
  <header>
    <h1>Project Jaundice — first full cloud training grid</h1>
    <p class="sub">What the 68 runs support, what they don't, and what's a broken instrument.</p>
    <p class="prov">Kaggle T4 &middot; commit da46344 &middot; 2026-07-15/16 &middot;
      derived from experiments/*/metrics.json by aggregate.py + plots.py</p>
  </header>

  <div class="cards">
{cards}
  </div>

  <div class="lede">
    <p><strong class="good">One result holds, and it is the physics one.</strong>
      Illuminant counterfactual invariance is a clean 5.5x win over ERM and is
      publication-grade as-is.</p>
    <p><strong class="crit">The classification grid proves nothing yet.</strong>
      No method &times; split cell beats ERM at p&lt;0.05. The 1&ndash;2 AUC point edge is real
      in direction but sits inside seed noise, and at 3 seeds the grid had 6&ndash;23% power &mdash;
      it was never capable of resolving an effect this size.</p>
    <p><strong class="crit">Two open items are chasing measurement bugs, not model behaviour.</strong>
      The fairness metric is computed over strata of 1&ndash;10 images, on a stratification each
      model invents for itself; the domain-2 split conflates label shift with domain shift.
      Fix the instruments before reading them.</p>
  </div>

{figs}

  <footer>
    Full write-up and revised next steps: <code>observations/OBSERVATIONS.md</code>.
    Underlying tables: <code>observations/tables/*.csv</code>.
    Regenerate: <code>python observations/aggregate.py &amp;&amp; python observations/plots.py
    &amp;&amp; python observations/build_dashboard.py</code>.
  </footer>
</main>
</body>
</html>
"""
    out = HERE / "dashboard.html"
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, self-contained)")


if __name__ == "__main__":
    main()
