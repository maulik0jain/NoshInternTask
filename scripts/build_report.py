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
    "final2": ROOT / "results" / "final_eval_round2" / "summary.json",
    "cmp": ROOT / "results" / "round2" / "compare.json",
    "loss": ROOT / "results" / "round2" / "loss_check.json",
    "sc": ROOT / "results" / "round3" / "scene_check.json",
    "loso": ROOT / "results" / "round3" / "loso.json",
    "r3": ROOT / "results" / "round3" / "check.json",
    "final3": ROOT / "results" / "final_eval_round3" / "summary.json",
}
OPTION_LABEL = {"skin_off": "skin filter off", "ellipse_prior": "ellipse prior", "ellipse_clip": "ellipse prior + clip",
                "split": "blob splitting", "edge_rank": "edge ranking", "shape_reject": "shape / texture reject"}
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


def trial_round(ctx, tid):
    return trial_by_id(ctx)[tid]["round"]


def exp_table(ctx, rnd=1):
    rows = ["<table class='num'><tr><th>Trial</th><th>Change</th><th>IoU mean</th><th>IoU med.</th>"
            "<th>Prec.</th><th>Recall</th><th>Bnd F1</th><th>Inst F1</th><th>Empty</th><th>Decision</th></tr>"]
    for t in (t for t in ctx["log"]["trials"] if t["round"] == rnd):
        cls = "dropped" if t["decision"] == "discarded" else "kept"
        rows.append(f"<tr class='{cls}'><td>{t['id']}</td><td>{t['what']}</td><td>{fmt(t['iou'])}</td>"
                    f"<td>{fmt(t['iou_med'])}</td><td>{fmt(t['precision'])}</td><td>{fmt(t['recall'])}</td>"
                    f"<td>{fmt(t['bf1'])}</td><td>{fmt(t['inst_f1'])}</td><td>{t['empty']}</td>"
                    f"<td>{t['decision']}</td></tr>")
    return "\n".join(rows) + "</table>"


def trial_cards(ctx, rnd=1):
    out = []
    for t in (t for t in ctx["log"]["trials"] if t["round"] == rnd):
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


def sweep_table(ctx, rnd=1):
    rows = ["<table class='num'><tr><th>Sweep (dev)</th><th>On top of</th><th>Values → dev IoU mean</th><th>Picked</th></tr>"]
    for s in (s for s in ctx["log"]["sweeps"] if trial_round(ctx, s["base"]) == rnd):
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


def rounds_table(ctx):
    sp = ctx["cmp"]["splits"]
    keys = [("IoU mean", "iou_mean", 3), ("IoU median", "iou_median", 3), ("Dice", "dice", 3),
            ("Pixel precision", "precision", 3), ("Pixel recall", "recall", 3), ("Boundary F1 (2 px)", "boundary_f1", 3),
            ("Instance F1", "instance_f1", 3), ("Egg count exactly right", "count_exact", 2),
            ("Images with IoU ≥ 0.8", "n_good", 0), ("Images with IoU = 0", "n_zero", 0), ("Empty predictions", "empty", 0)]
    rows = ["<table class='num narrow'><tr><th></th><th>dev, round 1</th><th>dev, round 2</th>"
            "<th>eval, round 1 (clean)</th><th>eval, round 2 (2nd look)</th></tr>"]
    for label, k, nd in keys:
        cells = []
        for s in ("dev", "eval"):
            for r in ("round1", "round2"):
                v = sp[s][r][k]
                cells.append(f"<td>{v:.{nd}f}</td>" if nd else f"<td>{v}</td>")
        rows.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    return "\n".join(rows) + "</table>"


def cause_table(ctx):
    rows = ["<table class='num narrow'><tr><th>Round-1 failure cause</th><th>n</th><th>IoU round 1</th><th>IoU round 2</th></tr>"]
    for c, v in ctx["cmp"]["round1_failures_by_cause"].items():
        rows.append(f"<tr><td>{c}</td><td>{v['n']}</td><td>{fmt(v['iou_round1'], 2)}</td><td>{fmt(v['iou_round2'], 2)}</td></tr>")
    return "\n".join(rows) + "</table>"


def loss_table(ctx):
    lc = ctx["loss"]
    head = "".join(f"<th>{v} ({lc['variant_trials'][v]})</th>" for v in lc["variants"])
    rows = [f"<table class='num'><tr><th>Image</th><th>Split</th><th>Round 1</th><th>Round 2</th>{head}</tr>"]
    for i in lc["images"]:
        cells = "".join((f"<td><b>{fmt(i['undo'][v], 2)}</b></td>" if v == i["best_undo"] else f"<td>{fmt(i['undo'][v], 2)}</td>")
                        for v in lc["variants"])
        rows.append(f"<tr><td>{i['id']}</td><td>{i['split']}</td><td>{fmt(i['iou_round1'], 2)}</td>"
                    f"<td>{fmt(i['iou_round2'], 2)}</td>{cells}</tr>")
    return "\n".join(rows) + "</table>"


def zero_table(ctx):
    rows = ["<table class='num narrow'><tr><th>Still IoU 0</th><th>Split</th><th>Round-1 cause</th>"
            "<th>Egg removed by saturation gate</th><th>IoU with that gate off too</th></tr>"]
    for z in ctx["loss"]["still_zero"]:
        rows.append(f"<tr><td>{z['id']}</td><td>{z['split']}</td><td>{z['cause']}</td>"
                    f"<td>{z['sat_loss']:.0%}</td><td>{fmt(z['iou_no_saturation'], 2)}</td></tr>")
    return "\n".join(rows) + "</table>"


def opts(names):
    return " + ".join(OPTION_LABEL[n] for n in names) or "none (round-1 pipeline)"


def yes(ok):
    return "<span class='kept'>pass</span>" if ok else "<span class='dropped'>fail</span>"


def scene_rule_table(ctx):
    """Round-2 decisions re-judged per scene (results/round3/scene_check.json)."""
    rows = ["<table class='num'><tr><th>Round-2 change</th><th>vs</th><th>Δ IoU, image mean (round-2 rule)</th>"
            "<th>Δ IoU, scene mean</th><th>Scenes helped / hurt</th><th>Worst scene</th><th>P(gain), scene bootstrap</th>"
            "<th>Round-3 rule</th></tr>"]
    for c in ctx["sc"]["changes"]:
        rows.append(f"<tr><td>{c['label']}</td><td>{c['base']}</td><td>{c['image_delta']:+.3f}</td>"
                    f"<td>{c['scene_delta']:+.3f}</td><td>{c['helped']} / {c['hurt']}</td><td>{c['worst']:+.2f}</td>"
                    f"<td>{c['p_gain']:.3f}</td><td>{yes(c['passed'])}</td></tr>")
    return "\n".join(rows) + "</table>"


def selection_table(ctx):
    """Forward selection of the final round-3 procedure on all dev scenes, every candidate tried."""
    proc = ctx["loso"]["procedures"][ctx["loso"]["final"]["procedure"]]
    rows = ["<table class='num'><tr><th>Step</th><th>Already in</th><th>Candidate</th><th>Δ scene IoU</th>"
            "<th>Helped / hurt</th><th>Worst scene</th><th>P(gain)</th><th>Rule</th></tr>"]
    for i, st in enumerate(proc["steps"], 1):
        for t in sorted(st["tried"], key=lambda t: -t["scene_delta"]):
            add = " → <b>added</b>" if st.get("added") == t["option"] else ""
            failed = ", ".join(k.replace("_", " ") for k, ok in t["checks"].items() if not ok)
            rows.append(f"<tr><td>{i}</td><td>{opts(st['chosen_before'])}</td><td>{OPTION_LABEL[t['option']]}</td>"
                        f"<td>{t['scene_delta']:+.3f}</td><td>{t['helped']} / {t['hurt']}</td><td>{t['worst']:+.2f}</td>"
                        f"<td>{t['p_gain']:.3f}</td><td>{yes(t['passed'])}{add}"
                        f"{'' if t['passed'] else ' <small>(' + failed + ')</small>'}</td></tr>")
    return "\n".join(rows) + "</table>"


def loso_table(ctx):
    rows = ["<table class='num'><tr><th>Selection procedure</th><th>Chosen on all 14 dev scenes</th>"
            "<th>Dev IoU, in-sample</th><th>Dev IoU, held-out scene (LOSO)</th><th>Optimism</th>"
            "<th>Different choices over 14 folds</th></tr>"]
    for name, p in ctx["loso"]["procedures"].items():
        rows.append(f"<tr><td>{name}</td><td>{opts(p['chosen'])}</td><td>{fmt(p['in_sample']['iou_image'])}</td>"
                    f"<td>{fmt(p['loso']['iou_image'])}</td><td>{p['optimism_image']:+.3f}</td>"
                    f"<td>{p['distinct_choices']}</td></tr>")
    return "\n".join(rows) + "</table>"


def stability_table(ctx):
    procs = ctx["loso"]["procedures"]
    names = list(procs)
    rows = ["<table class='num narrow'><tr><th>Change: picked in how many of the 14 folds</th>"
            + "".join(f"<th>{n}</th>" for n in names) + "</tr>"]
    for o in OPTION_LABEL:
        rows.append(f"<tr><td>{OPTION_LABEL[o]}</td>"
                    + "".join(f"<td>{procs[n]['stability'].get(o, '–')}</td>" for n in names) + "</tr>")
    return "\n".join(rows) + "</table>"


def predict_table(ctx):
    rows = ["<table class='num narrow'><tr><th></th><th>Dev, in-sample</th><th>Dev, held-out scene (LOSO)</th>"
            "<th>Eval</th><th>Eval 90% interval</th><th>Gap in-sample − eval</th><th>Gap held-out − eval</th></tr>"]
    for r, v in ctx["r3"]["rounds"].items():
        lo, hi = v["eval_ci90"]
        rows.append(f"<tr><td>Round {r} ({v['trial']})</td><td>{fmt(v['dev_in_sample'])}</td>"
                    f"<td>{fmt(v.get('loso'))}</td><td>{fmt(v['eval'])}</td><td>{lo:.3f} – {hi:.3f}</td>"
                    f"<td>{v['gap_dev_minus_eval']:+.3f}</td>"
                    f"<td>{'–' if 'gap_loso_minus_eval' not in v else format(v['gap_loso_minus_eval'], '+.3f')}</td></tr>")
    return "\n".join(rows) + "</table>"


BLOCKS = {
    "scene_rule_table": scene_rule_table, "selection_table": selection_table, "loso_table": loso_table,
    "stability_table": stability_table, "predict_table": predict_table,
    "exp_table_r3": lambda c: exp_table(c, 3), "trial_cards_r3": lambda c: trial_cards(c, 3),
    "rounds_table": rounds_table, "cause_table": cause_table, "loss_table": loss_table, "zero_table": zero_table,
    "exp_table": exp_table, "trial_cards": trial_cards, "sweep_table": sweep_table,
    "exp_table_r2": lambda c: exp_table(c, 2), "trial_cards_r2": lambda c: trial_cards(c, 2),
    "sweep_table_r2": lambda c: sweep_table(c, 2),
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
