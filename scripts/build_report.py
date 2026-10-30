"""Fill report/report_template.html with numbers from the result files and print the PDF.

    python scripts/build_report.py [--no-pdf]

Every number in the report comes from a JSON written by a script:
  {{final.eval.iou.mean:.3f}}        -> dotted path into one of SOURCES, optional format spec
  {{trial.T8.iou:.3f}}               -> field of the experiment-log entry with that id
  {{abl:no GrabCut|iou_mean:.3f}}    -> field of the ablation run with that label
  {{log.trials|count}}               -> length of a list
  {{@exp_table}}                     -> a generated block (functions in BLOCKS)
Unknown paths raise, so a typo can't silently print a stale value.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
SOURCES = {
    "log": ROOT / "experiments" / "log.json",
    "final": ROOT / "results" / "final_eval" / "summary.json",
    "fac": ROOT / "results" / "analysis" / "factors.json",
    "tax": ROOT / "results" / "analysis" / "taxonomy.json",
    "tab": ROOT / "results" / "analysis" / "metrics_table.json",
    "abl": ROOT / "results" / "ablation" / "ablation.json",
}
CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")


def lookup(ctx, path):
    cur = ctx
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def trial_by_id(ctx):
    return {t["id"]: t for t in ctx["log"]["trials"]}


def abl(ctx, label, key):
    return next(r for r in ctx["abl"]["runs"] if r["label"] == label)[key]


def exp_table(ctx):
    rows = ["<table class='num'><tr><th>Trial</th><th>Change</th><th>IoU mean</th><th>IoU med.</th>"
            "<th>Prec.</th><th>Recall</th><th>Bnd F1</th><th>Inst F1</th><th>Empty</th><th>Decision</th></tr>"]
    for t in ctx["log"]["trials"]:
        cls = "dropped" if t["decision"] == "discarded" else "kept"
        rows.append(f"<tr class='{cls}'><td>{t['id']}</td><td>{t['what']}</td><td>{fmt(t['iou'])}</td>"
                    f"<td>{fmt(t['iou_med'])}</td><td>{fmt(t['precision'])}</td><td>{fmt(t['recall'])}</td>"
                    f"<td>{fmt(t['bf1'])}</td><td>{fmt(t['inst_f1'])}</td><td>{t['empty']}</td>"
                    f"<td>{t['decision']}</td></tr>")
    return "\n".join(rows) + "</table>"


def trial_cards(ctx):
    out = []
    for t in ctx["log"]["trials"]:
        params = ", ".join(f"<code>{k}: {json.dumps(v)}</code>" for k, v in t["changes"].items()) or "none (baseline)"
        out.append(
            f"<div class='card'><div class='ct'><b>{t['id']}</b> &nbsp;{t['what']} "
            f"<span class='{'dropped' if t['decision'] == 'discarded' else 'kept'}'>{t['decision']}</span></div><ul>"
            f"<li><b>Why:</b> {t['why']}</li>"
            f"<li><b>Params:</b> on top of {t['extends']} → {params}</li>"
            f"<li><b>Dev:</b> IoU {fmt(t['iou'])} (median {fmt(t['iou_med'])}), boundary F1 {fmt(t['bf1'])}, "
            f"instance F1 {fmt(t['inst_f1'])}, empty masks {t['empty']}</li>"
            f"<li><b>Decision:</b> {t['reason']}</li></ul></div>")
    return "\n".join(out)


def sweep_table(ctx):
    rows = ["<table class='num'><tr><th>Sweep (dev)</th><th>On top of</th><th>Values → dev IoU mean</th><th>Picked</th></tr>"]
    for s in ctx["log"]["sweeps"]:
        best = max(s["runs"], key=lambda r: (round(r["iou_mean"], 3), r["instance_f1"] or 0))
        vals = " · ".join(f"{json.dumps(r['value'])} → {fmt(r['iou_mean'])}" for r in s["runs"])
        rows.append(f"<tr><td><code>{s['param']}</code></td><td>{s['base']}</td><td>{vals}</td>"
                    f"<td>{json.dumps(best['value'])}</td></tr>")
    return "\n".join(rows) + "</table>"


def metrics_table(ctx):
    f = ctx["final"]

    def row(label, key, sub="mean"):
        return f"<tr><td>{label}</td>" + "".join(f"<td>{fmt(f[s][key][sub])}</td>" for s in ("dev", "eval")) + "</tr>"

    def irow(label, key):
        return f"<tr><td>{label}</td>" + "".join(f"<td>{fmt(f[s]['instance'][key])}</td>" for s in ("dev", "eval")) + "</tr>"

    rows = [f"<table class='num narrow'><tr><th>Metric (per-image mean unless noted)</th>"
            f"<th>dev (n={f['dev']['images']})</th><th>eval (n={f['eval']['images']})</th></tr>",
            row("IoU", "iou"), row("IoU — median", "iou", "median"), row("Dice", "dice"),
            row("Pixel precision", "precision"), row("Pixel precision — median", "precision", "median"),
            row("Pixel recall", "recall"), row("Boundary F1 (2 px)", "boundary_f1"),
            irow("Instance precision (IoU ≥ 0.5, pooled)", "precision"),
            irow("Instance recall (IoU ≥ 0.5, pooled)", "recall"), irow("Instance F1", "f1"),
            irow("Egg count exactly right (share of images)", "count_exact"),
            "<tr><td>Images with an empty prediction</td>" + "".join(f"<td>{f[s]['empty_predictions']}</td>" for s in ("dev", "eval")) + "</tr>",
            "<tr><td>Seconds per image (CPU)</td>" + "".join(f"<td>{f[s]['seconds_per_image']:.2f}</td>" for s in ("dev", "eval")) + "</tr>"]
    return "\n".join(rows) + "</table>"


def factor_table(ctx, col, title):
    a, e = ctx["fac"]["all"][col], ctx["fac"]["eval"][col]
    rows = [f"<table class='num narrow'><tr><th>{title}</th><th>n (all)</th><th>IoU (all)</th><th>n (eval)</th><th>IoU (eval)</th></tr>"]
    for k, v in a.items():
        ev = e.get(k)
        rows.append(f"<tr><td>{k.replace('_', ' ')}</td><td>{v['n']}</td><td>{fmt(v['iou_mean'])}</td>"
                    f"<td>{ev['n'] if ev else 0}</td><td>{fmt(ev['iou_mean']) if ev else '–'}</td></tr>")
    return "\n".join(rows) + "</table>"


def taxonomy_table(ctx):
    t = ctx["tax"]
    rows = ["<table class='num narrow'><tr><th>Primary cause</th><th>dev</th><th>eval</th><th>total</th></tr>"]
    for c in t["counts"]:
        d, e = t["by_split"].get("dev", {}).get(c, 0), t["by_split"].get("eval", {}).get(c, 0)
        rows.append(f"<tr><td>{c}</td><td>{d}</td><td>{e}</td><td>{t['counts'][c]}</td></tr>")
    return "\n".join(rows) + "</table>"


def ablation_table(ctx):
    runs = ctx["abl"]["runs"]
    base = runs[0]
    rows = ["<table class='num narrow'><tr><th>Dev, one stage off</th><th>IoU mean</th><th>Δ IoU</th><th>Prec.</th>"
            "<th>Recall</th><th>Bnd F1</th><th>Inst F1</th><th>Empty</th></tr>"]
    for r in runs:
        d = "" if r is base else f"{r['iou_mean'] - base['iou_mean']:+.3f}"
        rows.append(f"<tr><td>{r['label']}</td><td>{fmt(r['iou_mean'])}</td><td>{d}</td><td>{fmt(r['precision'])}</td>"
                    f"<td>{fmt(r['recall'])}</td><td>{fmt(r['boundary_f1'])}</td><td>{fmt(r['instance_f1'])}</td>"
                    f"<td>{r['empty']}</td></tr>")
    return "\n".join(rows) + "</table>"


BLOCKS = {
    "exp_table": exp_table, "trial_cards": trial_cards, "sweep_table": sweep_table,
    "metrics_table": metrics_table, "taxonomy_table": taxonomy_table, "ablation_table": ablation_table,
    "factor_egg_count": lambda c: factor_table(c, "egg_count", "Eggs in image"),
    "factor_hand": lambda c: factor_table(c, "hand", "Hand in image"),
    "factor_lighting": lambda c: factor_table(c, "lighting", "Lighting"),
    "factor_touching": lambda c: factor_table(c, "touching", "Eggs touching"),
}


def render(template, ctx):
    def sub(m):
        expr = m[1].strip()
        if expr.startswith("@"):
            return BLOCKS[expr[1:]](ctx)
        if expr.startswith("abl:"):
            body, _, spec = expr[4:].partition(":")
            label, key = body.split("|")
            val = abl(ctx, label, key)
            return format(val, spec) if spec else str(val)
        path, _, spec = expr.partition(":")
        if path.endswith("|count"):
            return str(len(lookup(ctx, path[:-6])))
        if path.startswith("trial."):
            _, tid, key = path.split(".", 2)
            val = lookup(trial_by_id(ctx)[tid], key)
        else:
            val = lookup(ctx, path)
        return format(val, spec) if spec else str(val)
    return re.sub(r"\{\{(.+?)\}\}", sub, template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    ctx = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in SOURCES.items()}
    html = render((REPORT / "report_template.html").read_text(encoding="utf-8"), ctx)
    (REPORT / "report.html").write_text(html, encoding="utf-8")
    print("report/report.html written")
    if a.no_pdf:
        return
    chrome = CHROME if CHROME.exists() else shutil.which("chrome") or shutil.which("google-chrome")
    if not chrome:
        sys.exit("Chrome not found; open report/report.html and print to PDF by hand.")
    subprocess.run([str(chrome), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={REPORT / 'report.pdf'}", (REPORT / "report.html").resolve().as_uri()],
                   check=True, capture_output=True)
    print("report/report.pdf written")


if __name__ == "__main__":
    main()
