"""Classical (no learning) egg segmentation.

segment(rgb, cfg) runs the stages below in order. Every stage is a small function
whose parameters come from one block of the YAML config, and each can be switched
off with `enabled: false` so a trial changes exactly one thing.

    preprocess   gray-world white balance, CLAHE on L*, blur
    candidate    bright/white pixels: Otsu on a channel, or k-means in Lab
    skin         remove skin-coloured pixels (YCbCr box, Chai & Ngan 1999); optionally
                 only large skin regions touching the border (hands), T12b
    saturation   keep only low-saturation pixels (eggs are white/cream)
    grow         local Otsu around each found egg to add its shaded side (T13b)
    morphology   open -> close -> fill holes
    components   keep egg-shaped blobs (area, solidity, axis ratio, ellipse fit, border)
    grabcut      refine edges with GrabCut seeded from the mask (Rother 2004)
    ellipse      snap near-elliptical blobs to their fitted ellipse (Fitzgibbon 1999)
    watershed    split touching eggs: distance-transform markers + watershed (Vincent & Soille 1991)

Masks are bool HxW, instance maps are int32 HxW (0 = background).
"""
import cv2
import numpy as np


# ---------------------------------------------------------------- preprocess
def gray_world(rgb):
    """Scale each channel so its mean equals the grey mean. Removes colour casts
    (warm bulbs make white eggs look cream); assumes the scene averages to grey."""
    f = rgb.astype(np.float32)
    means = f.reshape(-1, 3).mean(0)
    f *= means.mean() / np.maximum(means, 1e-6)
    return np.clip(f, 0, 255).astype(np.uint8)


def clahe_lightness(rgb, clip, tile):
    """CLAHE on L* only, so contrast is equalised locally without shifting colours."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess(rgb, p):
    out = rgb
    if p.get("white_balance"):
        out = gray_world(out)
    if p["clahe"]["enabled"]:
        out = clahe_lightness(out, p["clahe"]["clip"], p["clahe"]["tile"])
    if p.get("blur", 0) > 1:
        k = int(p["blur"]) | 1  # kernel must be odd
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


# ---------------------------------------------------------------- candidate
def channel(rgb, name, chroma_weight=1.0):
    """Single uint8 channel in which eggs should be bright.

    gray       plain luminance
    L          CIE L* (perceptual lightness)
    whiteness  L* - w * chroma: high only for bright AND colourless pixels, so a
               bright pink bowl or skin scores lower than a white egg
    """
    if name == "gray":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    if name == "L":
        return lab[..., 0].astype(np.uint8)
    if name == "whiteness":
        chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
        return np.clip(lab[..., 0] - chroma_weight * chroma, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown channel {name}")


def otsu(ch):
    """Otsu (1979): threshold that maximises between-class variance of the histogram."""
    _, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m > 0


def kmeans_lab(rgb, k, chroma_weight, samples, seed):
    """k-means on Lab pixels; return the cluster whose centre is most egg-like
    (highest L* - w * chroma). Fitted on a pixel sample for speed, then every
    pixel is assigned to its nearest centre."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    sample = lab[rng.choice(len(lab), min(samples, len(lab)), replace=False)]
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centres = cv2.kmeans(sample, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    score = centres[:, 0] - chroma_weight * np.hypot(centres[:, 1] - 128, centres[:, 2] - 128)
    best = int(np.argmax(score))
    d = ((lab[:, None, :] - centres[None]) ** 2).sum(-1)
    return (d.argmin(1) == best).reshape(rgb.shape[:2])


def candidate(rgb, p):
    if p["method"] == "otsu":
        return otsu(channel(rgb, p["channel"], p.get("chroma_weight", 1.0)))
    if p["method"] == "kmeans":
        return kmeans_lab(rgb, p["k"], p.get("chroma_weight", 1.0), p.get("samples", 20000), p.get("seed", 0))
    raise ValueError(f"unknown candidate method {p['method']}")


# ---------------------------------------------------------------- colour filters
def skin(rgb, cb_range, cr_range):
    """Chai & Ngan (1999) skin box in YCbCr. White eggs sit near Cb=Cr=128, below
    the Cr>=133 edge of the box, so they survive while palms are removed."""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)  # OpenCV order: Y, Cr, Cb
    cr, cb = ycc[..., 1], ycc[..., 2]
    return (cb >= cb_range[0]) & (cb <= cb_range[1]) & (cr >= cr_range[0]) & (cr <= cr_range[1])


def skin_regions(skin_mask, min_frac, close):
    """Keep only skin regions that are large and touch the image border: hands and
    arms enter the photo from the edge. A cream or warm-lit egg also falls inside the
    skin box, but in the middle of the image and egg-sized, so it survives (T12b)."""
    m = skin_mask.astype(np.uint8)
    if close > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(close))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    h, w = skin_mask.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        on_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        keep[i] = on_border and area >= min_frac * h * w
    return keep[lab] & skin_mask


def low_saturation(rgb, s_max):
    """HSV saturation <= s_max (0-255): eggs are close to colourless."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1] <= s_max


# ---------------------------------------------------------------- morphology
def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def fill_holes(mask):
    """Fill every outer contour: removes holes left by specular highlights and specks."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, cnts, -1, 1, cv2.FILLED)
    return out > 0


def morphology(mask, p):
    """Open removes thin bridges/specks (texture of ash, rugs); close bridges small
    gaps across the egg (stems, scratches)."""
    m = mask.astype(np.uint8)
    if p["open"] > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(p["open"]))
    if p["close"] > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(p["close"]))
    m = m > 0
    return fill_holes(m) if p["fill_holes"] else m


# ---------------------------------------------------------------- components
def component_features(comp, img_shape, offset=(0, 0)):
    """Shape features for one blob. `comp` may be a crop of the image whose top-left
    corner sits at `offset` (x, y); the fitted ellipse is returned in image coords."""
    h, w = img_shape
    ox, oy = offset
    area = int(comp.sum())
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    f = {"area_frac": area / (h * w), "solidity": area / hull if hull > 0 else 0.0,
         "axis_ratio": 99.0, "ellipse_iou": 0.0}
    pts = c.reshape(-1, 2) + (ox, oy)
    f["border_frac"] = float(((pts[:, 0] <= 1) | (pts[:, 1] <= 1) | (pts[:, 0] >= w - 2) | (pts[:, 1] >= h - 2)).mean())
    rw, rh = cv2.minAreaRect(c)[1]
    # area / min bounding rectangle: an ellipse is pi/4 = 0.785, a key cap or box ~1
    f["rectangularity"] = area / (rw * rh) if rw * rh > 0 else 1.0
    if len(c) >= 5:
        (cx, cy), (a, b), ang = cv2.fitEllipse(c)  # least-squares ellipse (Fitzgibbon 1999)
        # A near-straight contour gives a degenerate, enormous ellipse: not an egg.
        if 0 < min(a, b) and max(a, b) <= 2 * max(h, w):
            f["axis_ratio"] = max(a, b) / min(a, b)
            # Draw the ellipse on a canvas big enough to hold it, so a blob that is only
            # part of an ellipse is still compared with the whole ellipse.
            pad = int(max(a, b))
            canvas = np.zeros((comp.shape[0] + 2 * pad, comp.shape[1] + 2 * pad), np.uint8)
            cv2.ellipse(canvas, ((cx + pad, cy + pad), (a, b), ang), 1, cv2.FILLED)
            big = np.zeros_like(canvas)
            big[pad:pad + comp.shape[0], pad:pad + comp.shape[1]] = comp
            ell, blob = canvas > 0, big > 0
            f["ellipse_iou"] = float((ell & blob).sum() / max((ell | blob).sum(), 1))
            f["ellipse"] = ((cx + ox, cy + oy), (a, b), ang)
    return f


def passes_shape(f, p):
    return (p["min_area"] <= f["area_frac"] <= p["max_area"] and f["solidity"] >= p["min_solidity"]
            and f["axis_ratio"] <= p["max_axis_ratio"] and f["ellipse_iou"] >= p.get("min_ellipse_iou", 0.0)
            and f["border_frac"] <= p["max_border_frac"]
            and (p.get("max_rectangularity") is None or f["rectangularity"] <= p["max_rectangularity"]))


def filter_components(mask, p):
    """Keep blobs that look like an egg. An egg is a convex, roughly elliptical blob
    of reasonable size; it may touch the image border (some photos crop eggs), but
    a blob running along the border is usually background (table edge, bag)."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(mask.shape, bool)
    for i in range(1, n):
        comp = lab == i
        if passes_shape(component_features(comp, mask.shape), p):
            keep |= comp
    return keep


# ---------------------------------------------------------------- edge-based candidates
def edge_ellipses(ch, edges, edge_dist, allowed, shape_p, p):
    """Ellipses fitted to Canny edge chains (T15b). On a white bag the egg is as bright
    as the background, so no threshold separates it, but its outline is still an edge.
    Each long enough chain gets a least-squares ellipse; it becomes a candidate if
    enough of its perimeter lies on edges and its inside is plausible egg (passes the
    colour gate, not darker than most of the image). Score = edge support."""
    h, w = ch.shape
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    dark = np.percentile(ch, p.get("edge_min_pct", 40))
    out = []
    for c in cnts:
        if len(c) < p.get("edge_min_pts", 40):
            continue
        (cx, cy), (a, b), ang = cv2.fitEllipse(c)
        if not (min(a, b) > 4 and max(a, b) <= max(h, w)):
            continue
        if not shape_p["min_area"] <= np.pi * a * b / 4 / (h * w) <= shape_p["max_area"]:
            continue
        if max(a, b) / min(a, b) > shape_p["max_axis_ratio"]:
            continue
        poly = cv2.ellipse2Poly((int(cx), int(cy)), (int(a / 2), int(b / 2)), int(ang), 0, 360, 4)
        inside = (poly[:, 0] >= 0) & (poly[:, 0] < w) & (poly[:, 1] >= 0) & (poly[:, 1] < h)
        if inside.mean() < 1 - shape_p["max_border_frac"]:
            continue
        support = float((edge_dist[poly[inside, 1], poly[inside, 0]] <= p.get("edge_tol", 2)).mean())
        if support < p.get("edge_min_support", 0.6):
            continue
        full = np.zeros((h, w), np.uint8)
        cv2.ellipse(full, ((cx, cy), (a, b), ang), 1, cv2.FILLED)
        full = full > 0
        if (allowed is not None and allowed[full].mean() < 0.7) or ch[full].mean() < dark:
            continue
        ys, xs = np.nonzero(full)
        x0, y0 = xs.min(), ys.min()
        crop = full[y0:ys.max() + 1, x0:xs.max() + 1]
        out.append((support, int(crop.sum()), int(x0), int(y0), crop))
    return out


# ---------------------------------------------------------------- multi-threshold egg search
def threshold_search(rgb, p, shape_p, allowed=None):
    """Try many thresholds instead of one (T7 showed one global threshold glues the
    egg to bright background). At each level, every blob that passes the shape
    filters becomes a candidate scored by ellipse_iou * solidity (~1 for an egg).
    Candidates are then picked greedily, best score first, skipping anything that
    overlaps an egg already picked. Among nested candidates of almost the same
    score, the largest wins, so the egg isn't shrunk to its bright centre.
    Same idea as MSER (stable extremal regions), with an egg-shape prior.
    """
    ch = channel(rgb, p["channel"], p.get("chroma_weight", 1.0))
    h, w = ch.shape
    lo, hi = np.percentile(ch, p["pct_range"])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge_w = p.get("edge_weight", 0.0)
    edges = edge_dist = None
    if edge_w > 0 or p.get("edge_candidates"):
        edges = cv2.Canny(gray, *p.get("canny", [30, 90]))
        edge_dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F)) if p.get("max_texture") is not None else None
    cands = []  # (score, area, x, y, crop-mask)

    def consider(crop, x, y):
        """Score one blob (crop of the image at x, y); keep it if it looks like an egg."""
        f = component_features(crop, (h, w), offset=(x, y))
        if not passes_shape(f, shape_p):
            return False
        if lap is not None and lap[y:y + crop.shape[0], x:x + crop.shape[1]][crop].mean() > p["max_texture"]:
            return False  # crinkled foil, printed keys: eggs are smooth (T16)
        score = f["ellipse_iou"] * f["solidity"]
        rank = score
        if edge_w > 0:
            # share of the blob outline lying on an image edge: a threshold cut through a
            # flat white bag has no edge under it, a real egg outline does (T15)
            cnts, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pts = max(cnts, key=len).reshape(-1, 2) + (x, y)
            support = float((edge_dist[pts[:, 1], pts[:, 0]] <= p.get("edge_tol", 2)).mean())
            rank = score * (1 - edge_w + edge_w * support)
            if not p.get("edge_rank_only"):
                # T15: the weighted score also faces the min_score cut. It lowers every
                # score, so real eggs fell below the cut; T15c only ranks by it.
                score = rank
        if score >= p["min_score"]:
            cands.append((rank, int(crop.sum()), x, y, crop))
            return True
        return False

    for t in np.linspace(lo, hi, p["levels"]):
        m = ch >= t
        if allowed is not None:
            m &= allowed
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(p["open"])) if p["open"] > 0 else m.astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if not shape_p["min_area"] <= area / (h * w) <= shape_p["max_area"]:
                continue  # cheap reject before the expensive features
            crop = fill_holes(lab[y:y + bh, x:x + bw] == i)
            if consider(crop, x, y) or not p.get("split_merged"):
                continue
            # Touching eggs form one non-egg-shaped blob that would be thrown away here,
            # before the watershed stage ever sees it: split it now, score each piece (T14).
            if area < 2 * shape_p["min_area"] * h * w:
                continue
            pieces = split_instances(crop, {"peak_rel": p.get("split_peak_rel", 0.6),
                                            "min_core_px": p.get("split_min_core_px", 30)})
            for k in range(1, pieces.max() + 1 if pieces.max() > 1 else 1):
                if (pieces == k).any():  # a core can lose all its pixels to its neighbours
                    consider(pieces == k, x, y)
    if p.get("edge_candidates"):
        cands += edge_ellipses(ch, edges, edge_dist, allowed, shape_p, p)

    def full(c):
        out = np.zeros((h, w), bool)
        _, _, x, y, crop = c
        out[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        return out

    cands.sort(key=lambda c: -c[0])
    masks = [full(c) for c in cands]
    score, area = [c[0] for c in cands], [c[1] for c in cands]
    picked = np.zeros((h, w), bool)
    n_picked = 0
    alive = list(range(len(cands)))
    while alive:
        b = alive[0]  # best remaining score
        # nested versions of the same egg (other thresholds): take the largest one
        # whose score is within `score_tol` of the best
        group = {i for i in alive if (masks[i] & masks[b]).sum() > 0.5 * min(area[i], area[b])}
        choice = max((i for i in group if score[i] >= score[b] - p["score_tol"]), key=lambda i: area[i])
        if (masks[choice] & picked).sum() <= p["max_overlap"] * area[choice]:
            picked |= masks[choice]
            n_picked += 1
            if p.get("max_eggs") and n_picked >= p["max_eggs"]:
                break
        alive = [i for i in alive if i not in group and (masks[i] & picked).sum() <= p["max_overlap"] * area[i]]
    return picked


# ---------------------------------------------------------------- growth
def scaled_ellipse_mask(shape, ell, scale):
    (cx, cy), (a, b), ang = ell
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, ((cx, cy), (a * scale, b * scale), ang), 1, cv2.FILLED)
    return m > 0


def grow_local(ch, mask, allowed, p):
    """Add the shaded side of each found egg (T13b). The search keeps the lit cap,
    because the global threshold that isolates the egg also cuts off its shaded half.
    Here the threshold is chosen locally: Otsu on the pixels inside the egg's fitted
    ellipse enlarged by roi_scale (egg vs the background right next to it); pixels
    above it, connected to the egg and inside the ellipse enlarged by keep_scale,
    are added."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    out = mask.copy()
    for i in range(1, n):
        comp = lab == i
        f = component_features(comp, mask.shape)
        if "ellipse" not in f:
            continue
        roi = scaled_ellipse_mask(mask.shape, f["ellipse"], p["roi_scale"])
        t, _ = cv2.threshold(ch[roi].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cand = (ch >= t) & scaled_ellipse_mask(mask.shape, f["ellipse"], p["keep_scale"])
        if p.get("respect_gate", True) and allowed is not None:
            cand &= allowed
        n2, lab2 = cv2.connectedComponents((cand | comp).astype(np.uint8), connectivity=8)
        ids = np.unique(lab2[comp])
        out |= np.isin(lab2, ids[ids > 0])
    return out


# ---------------------------------------------------------------- refinement
def grabcut_refine(rgb, mask, p):
    """GrabCut (Rother 2004, a Boykov-Jolly graph cut with GMM colour models), seeded
    automatically: eroded mask = sure egg, far outside the mask = sure background,
    the band in between = let the graph cut decide. Fixes ragged threshold edges."""
    if not mask.any():
        return mask
    m8 = mask.astype(np.uint8)
    gc = np.full(mask.shape, cv2.GC_PR_BGD, np.uint8)
    gc[m8 > 0] = cv2.GC_PR_FGD
    clip = None
    if p.get("ellipse_prior"):
        # T13: the enlarged fitted ellipse of each egg also starts as "probably egg", so
        # the shaded side is in the egg colour model from the first iteration
        n, lab = cv2.connectedComponents(m8, connectivity=8)
        if p.get("ellipse_clip"):
            clip = mask.copy()
        for i in range(1, n):
            f = component_features(lab == i, mask.shape)
            if "ellipse" in f:
                gc[scaled_ellipse_mask(mask.shape, f["ellipse"], p["ellipse_prior"])] = cv2.GC_PR_FGD
                if clip is not None:
                    clip |= scaled_ellipse_mask(mask.shape, f["ellipse"], p["ellipse_clip"])
    gc[cv2.erode(m8, disk(p["erode"])) > 0] = cv2.GC_FGD
    gc[cv2.dilate(m8, disk(p["bg_dilate"])) == 0] = cv2.GC_BGD
    if not (gc == cv2.GC_FGD).any():  # blob too small to erode: keep it as probable fg only
        gc[m8 > 0] = cv2.GC_PR_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(p.get("seed", 0))  # GMM init uses k-means with OpenCV's RNG: fix it so runs repeat exactly
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, p["iters"], cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask  # degenerate GMM (e.g. no background samples): keep the input
    out = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    if clip is not None:
        # round 3: an egg may grow only up to its fitted ellipse x ellipse_clip, so a surface
        # next to it that looks like the egg (water, steel, pale wall) can't be pulled in
        out &= clip
    if p.get("fill_holes", True):
        out = fill_holes(out)
    # GrabCut refines the eggs it was given; blobs that don't touch the seed are new
    # background objects, not eggs, so drop them.
    n, lab = cv2.connectedComponents(out.astype(np.uint8), connectivity=8)
    touching = np.unique(lab[mask & out])
    return np.isin(lab, touching[touching > 0])


def ellipse_regularise(mask, p):
    """Snap each blob that is already nearly elliptical (ellipse IoU >= min_fit_iou)
    to its fitted ellipse; leave irregular blobs alone, because those are usually
    eggs occluded by fingers/leaves and the GT only covers the visible part."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, bool)
    for i in range(1, n):
        comp = lab == i
        f = component_features(comp, mask.shape)
        if "ellipse" in f and f["ellipse_iou"] >= p["min_fit_iou"]:
            ell = np.zeros(mask.shape, np.uint8)
            cv2.ellipse(ell, f["ellipse"], 1, cv2.FILLED)
            out |= (ell > 0) if p["mode"] == "replace" else ((ell > 0) | comp)
        else:
            out |= comp
    return out


# ---------------------------------------------------------------- instances
def split_instances(mask, p):
    """Distance-transform watershed. Inside each blob, pixels far from the edge
    (dist >= peak_rel * max) form one core per egg; two touching eggs give two cores
    joined only by a thin neck. Cores seed a watershed on the inverted distance map.
    Returns int32 labels."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, np.int32)
    next_id = 1
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        cores = (dist >= p["peak_rel"] * dist.max()).astype(np.uint8)
        nc, markers = cv2.connectedComponents(cores, connectivity=8)
        # drop tiny cores (noise on a bumpy edge) so one egg doesn't split in two
        sizes = np.bincount(markers.ravel(), minlength=nc)
        good = [c for c in range(1, nc) if sizes[c] >= p["min_core_px"]]
        if len(good) <= 1:
            out[comp > 0] = next_id
            next_id += 1
            continue
        mk = np.zeros(mask.shape, np.int32)
        for j, c in enumerate(good, start=1):
            mk[markers == c] = j
        mk[comp == 0] = len(good) + 1  # background marker outside this blob
        relief = cv2.normalize(-dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ws = cv2.watershed(cv2.merge([relief] * 3), mk)
        for j in range(1, len(good) + 1):
            out[(ws == j) & (comp > 0)] = next_id
            next_id += 1
        # watershed lines (-1) inside the blob: give them to the nearest egg so the
        # binary mask is unchanged by splitting
        line = (ws == -1) & (comp > 0)
        if line.any():
            grown = cv2.dilate(out.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
            out[line] = grown[line]
    return out


def components_to_instances(mask):
    _, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return lab.astype(np.int32)


# ---------------------------------------------------------------- full pipeline
def segment(rgb, cfg):
    """Run all enabled stages. Returns (mask bool, instances int32)."""
    img = preprocess(rgb, cfg["preprocess"])
    allowed = np.ones(rgb.shape[:2], bool)  # pixels the colour filters say could be egg
    if cfg["skin"]["enabled"]:
        sk = skin(img, cfg["skin"]["cb"], cfg["skin"]["cr"])
        if cfg["skin"].get("mode", "all") == "border_regions":
            sk = skin_regions(sk, cfg["skin"]["min_region"], cfg["skin"].get("close", 7))
        allowed &= ~sk
    if cfg["saturation"]["enabled"]:
        allowed &= low_saturation(img, cfg["saturation"]["s_max"])
    if cfg["candidate"]["method"] == "search":
        # the search applies the colour and shape filters itself, at every threshold
        m = threshold_search(img, cfg["candidate"], cfg["components"], allowed)
        fallback = cfg["candidate"].get("fallback_min_score")
        if fallback is not None and not m.any():
            # Nothing egg-shaped enough: one more pass with a looser shape score. Only
            # images that would otherwise get an empty mask are affected (T11).
            m = threshold_search(img, {**cfg["candidate"], "min_score": fallback,
                                       "max_eggs": cfg["candidate"].get("fallback_max_eggs")},
                                 cfg["components"], allowed)
    else:
        m = candidate(img, cfg["candidate"]) & allowed
    if cfg.get("grow", {}).get("enabled"):
        m = grow_local(channel(img, cfg["candidate"]["channel"]), m, allowed, cfg["grow"])
    if cfg["morphology"]["enabled"]:
        m = morphology(m, cfg["morphology"])
    if cfg["components"]["enabled"]:
        m = filter_components(m, cfg["components"])
    if cfg["grabcut"]["enabled"]:
        m = grabcut_refine(rgb, m, cfg["grabcut"])  # GrabCut models the original colours
        if cfg["components"]["enabled"]:
            m = filter_components(m, cfg["components"])
    if cfg["ellipse"]["enabled"]:
        m = ellipse_regularise(m, cfg["ellipse"])
    inst = split_instances(m, cfg["watershed"]) if cfg["watershed"]["enabled"] else components_to_instances(m)
    return m, inst
