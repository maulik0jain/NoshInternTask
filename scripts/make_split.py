"""Dev / eval split (4:1) with grouping to reduce leakage.

Steps:
  1. source id   = number in "not_damaged_<id>_jpg.rf.<hash>.jpg" (Roboflow
                   adds .rf.<hash>; copies of one photo would share the id)
  2. pHash       = near-identical shots (Hamming <= --phash-max) -> same group
  3. DINOv2      = same scene / session -> agglomerative clustering (cosine)
  4. manual      = data/splits/scene_merges.json, same-setup clusters the
                   embedding kept apart (checked by eye)
  5. split       = whole groups to eval; seeded search for ~20% of images,
                   multi-egg ratio close to overall, as many eval scenes as possible

Writes data/splits/split.csv (frozen, committed) + split_summary.json.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data" / "egg_dataset")
    p.add_argument("--gt", type=Path, default=ROOT / "ground_truth" / "annotations.json")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "splits")
    p.add_argument("--embed-model", default="facebook/dinov2-small")
    p.add_argument("--cluster-dist", type=float, default=0.30, help="cosine distance threshold, average linkage")
    p.add_argument("--phash-max", type=int, default=10)
    p.add_argument("--eval-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--search-iters", type=int, default=20000)
    p.add_argument("--strata-tol", type=float, default=0.05, help="max |multi-egg ratio eval - overall|")
    return p.parse_args()


def source_id(name):
    return int(re.match(r"not_damaged_(\d+)_jpg", name).group(1))


def phash(path):
    g = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (32, 32)).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()[1:]
    return d > np.median(d)


@torch.inference_mode()
def embed(paths, model_name):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()
    out = []
    for p in paths:
        h = model(**proc(images=Image.open(p).convert("RGB"), return_tensors="pt").to(dev)).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)[0].float().cpu().numpy())  # CLS + mean patch
    e = np.array(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        self.p[self.find(i)] = self.find(j)


def cross_split_similarity(sim, is_eval):
    """For each eval image, its highest cosine similarity to any dev image."""
    return sim[np.ix_(is_eval, ~is_eval)].max(axis=1)


def main():
    a = parse_args()
    paths = sorted(a.data.rglob("*.jpg"))
    rels = [p.relative_to(a.data).as_posix() for p in paths]
    sids = [source_id(p.name) for p in paths]
    n = len(paths)

    gt = {r["image"]: r["num_eggs"] for r in json.loads(a.gt.read_text())["images"]}
    n_eggs = np.array([gt[r] for r in rels])

    uf = UnionFind(n)
    # 1. same source photo (augmented copies)
    first = {}
    for i, s in enumerate(sids):
        uf.union(i, first.setdefault(s, i))
    n_dup_src = n - len(first)
    # 2. pHash near-duplicates
    H = np.array([phash(p) for p in paths])
    ham = (H[:, None, :] != H[None, :, :]).sum(-1)
    phash_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if ham[i, j] <= a.phash_max]
    for i, j in phash_pairs:
        uf.union(i, j)
    # 3. scene clusters from embeddings
    E = embed(paths, a.embed_model)
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=a.cluster_dist,
                                  metric="cosine", linkage="average").fit_predict(E)
    for c in set(lab):
        idx = np.where(lab == c)[0]
        for i in idx[1:]:
            uf.union(i, idx[0])
    n_auto = len({uf.find(i) for i in range(n)})
    # 4. manual same-setup merges
    merges = {k: v for k, v in json.loads((a.out / "scene_merges.json").read_text()).items() if not k.startswith("_")}
    by_sid = {s: i for i, s in enumerate(sids)}
    for ids in merges.values():
        for s in ids[1:]:
            uf.union(by_sid[s], by_sid[ids[0]])

    roots = sorted({uf.find(i) for i in range(n)}, key=lambda r: min(sids[i] for i in range(n) if uf.find(i) == r))
    group = np.array([roots.index(uf.find(i)) for i in range(n)])

    # 5. group split: whole groups go to eval. Seeded random search over group
    #    orderings; keep eval at eval_frac +-3% of images, keep the multi-egg
    #    ratio within strata_tol of the overall one, then prefer MORE distinct scenes
    #    (StratifiedGroupKFold gave a valid split with only 4 eval scenes).
    strata = (n_eggs > 1).astype(int)
    sizes = np.bincount(group)
    rng = np.random.default_rng(a.seed)
    lo, hi = (a.eval_frac - 0.03) * n, (a.eval_frac + 0.03) * n
    best = None
    for _ in range(a.search_iters):
        chosen, size = [], 0
        for g in rng.permutation(len(roots)):
            if size + sizes[g] <= hi:
                chosen.append(g)
                size += sizes[g]
            if size >= lo:
                break
        if not lo <= size <= hi:
            continue
        ev_mask = np.isin(group, chosen)
        gap = abs(strata[ev_mask].mean() - strata.mean())
        if gap > a.strata_tol:  # hard constraint: eval must not be all easy single-egg shots
            continue
        score = -len(chosen) + gap
        if best is None or score < best[0]:
            best = (score, ev_mask)
    is_eval = best[1]

    # leakage check: group split vs a plain random split of the same size
    sim = E @ E.T
    grp_nn = cross_split_similarity(sim, is_eval)
    rnd = np.zeros(n, bool)
    rnd[np.random.default_rng(0).choice(n, is_eval.sum(), replace=False)] = True
    rnd_nn = cross_split_similarity(sim, rnd)
    phash_cross = sum(is_eval[i] != is_eval[j] for i, j in phash_pairs)

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "split.csv", "w", newline="") as f:
        f.write("image,source_id,group,num_eggs,split\n")
        for i in np.argsort(sids):
            f.write(f"{rels[i]},{sids[i]},{group[i]},{n_eggs[i]},{'eval' if is_eval[i] else 'dev'}\n")

    summary = {
        "images": n, "dev": int((~is_eval).sum()), "eval": int(is_eval.sum()),
        "groups_total": len(roots), "groups_dev": len(set(group[~is_eval])), "groups_eval": len(set(group[is_eval])),
        "auto_groups_before_manual_merge": n_auto, "manual_merges": len(merges),
        "duplicate_source_ids": n_dup_src, "phash_pairs": len(phash_pairs), "phash_pairs_across_split": int(phash_cross),
        "multi_egg_frac": {"all": round(float(strata.mean()), 3), "dev": round(float(strata[~is_eval].mean()), 3),
                           "eval": round(float(strata[is_eval].mean()), 3)},
        "eggs": {"dev": int(n_eggs[~is_eval].sum()), "eval": int(n_eggs[is_eval].sum())},
        "leakage_check_eval_to_dev_max_cosine": {
            "group_split": {"mean": round(float(grp_nn.mean()), 3), "frac_above_0.9": round(float((grp_nn > 0.9).mean()), 3)},
            "random_split": {"mean": round(float(rnd_nn.mean()), 3), "frac_above_0.9": round(float((rnd_nn > 0.9).mean()), 3)},
        },
        "settings": {"embed_model": a.embed_model, "cluster_dist": a.cluster_dist, "phash_max": a.phash_max,
                     "stratify_on": "num_eggs > 1", "seed": a.seed, "search_iters": a.search_iters, "strata_tol": a.strata_tol},
    }
    (a.out / "split_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(a.out / "embeddings.npy", E)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
