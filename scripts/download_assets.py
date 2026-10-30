"""Download what isn't in the repo: the SAM 2.1 checkpoint (and the dataset, if missing).

Grounding DINO and DINOv2 weights are pulled from Hugging Face on first use.
"""
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
DATA_GDRIVE_ID = "1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv"  # "Alternative zip link" from the task PDF


def main():
    ckpt = ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt"
    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", SAM_URL)
        urllib.request.urlretrieve(SAM_URL, ckpt)
    print("checkpoint:", ckpt, f"{ckpt.stat().st_size / 1e6:.0f} MB")

    data = ROOT / "data" / "egg_dataset"
    if not any(data.rglob("*.jpg")):
        import gdown
        zp = ROOT / "data" / "egg_dataset.zip"
        gdown.download(id=DATA_GDRIVE_ID, output=str(zp))
        with zipfile.ZipFile(zp) as z:
            z.extractall(data)
    print("dataset:", data, len(list(data.rglob("*.jpg"))), "images")


if __name__ == "__main__":
    main()
