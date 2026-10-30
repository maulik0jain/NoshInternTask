"""Visualisation helpers (contact sheets for now; overlays and error maps come later)."""
import cv2
import numpy as np


def contact_sheet(images, labels=None, cols=6, tile=200):
    """Grid of square thumbnails (RGB uint8 in, RGB uint8 out), optional text label per tile."""
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * tile, cols * tile, 3), 255, np.uint8)
    for i, img in enumerate(images):
        t = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        if labels is not None:
            cv2.rectangle(t, (0, 0), (tile, 20), (0, 0, 0), -1)
            cv2.putText(t, str(labels[i]), (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = t
    return sheet


TP_COLOR, FP_COLOR, FN_COLOR = (60, 200, 60), (230, 40, 40), (40, 110, 255)


def overlay(rgb, mask, color=(255, 60, 60), alpha=0.5):
    out = rgb.copy()
    out[mask] = ((1 - alpha) * out[mask] + alpha * np.array(color)).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def error_map(rgb, gt, pred):
    """Dimmed image with TP green, FP (predicted, not egg) red, FN (missed egg) blue."""
    out = (rgb * 0.35).astype(np.uint8)
    out[gt & pred] = TP_COLOR
    out[~gt & pred] = FP_COLOR
    out[gt & ~pred] = FN_COLOR
    return out


def _title(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def error_panel(rgb, gt, pred, caption="", size=320):
    """image | GT | prediction | error map, one row."""
    tiles = [_title(rgb, caption or "image"), _title(overlay(rgb, gt, (60, 200, 60)), "GT (SAM)"),
             _title(overlay(rgb, pred, (255, 60, 60)), "prediction"),
             _title(error_map(rgb, gt, pred), "green TP  red FP  blue FN")]
    return np.hstack([cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA) for t in tiles])


def save_rgb(path, rgb, quality=88):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
