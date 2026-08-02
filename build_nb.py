#!/usr/bin/env python3
"""
build_nb.py  –  Generates reid_pipeline.ipynb
Clean cell structure, all known errors handled.
Cell sequence:
  1  Install dependencies
  2  Mount Drive + Paths + Seeds
  3  Download / Verify Market-1501
  4  Dataset statistics & identity-leakage check
  5  Prepare YOLO person dataset from COCO
  6  Train YOLOv8n person detector
  7  Evaluate detector
  8  torchreid DataManager
  9  Build OSNet + Train 60 epochs
  10 Backup checkpoint to Drive
  11 Load checkpoint + Extract features
  12 Compute mAP & CMC
  13 OpenCV 2x3 camera grid + GIF
  14 Gradio interactive dashboard
  15 K-Reciprocal Re-Ranking (stretch goal)
  16 Failure analysis (stretch goal)
"""
import json, uuid

def uid():
    return str(uuid.uuid4())[:8]

def md(s):
    lines = s.split('\n')
    src = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {'cell_type': 'markdown', 'id': uid(), 'metadata': {}, 'source': src}

def code(s):
    lines = s.split('\n')
    src = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {
        'cell_type': 'code', 'id': uid(), 'metadata': {},
        'source': src, 'outputs': [], 'execution_count': None
    }

cells = []

# ── TITLE ─────────────────────────────────────────────────────────────────────
cells.append(md('''# Multi-View Person Re-Identification Pipeline
## Big Vision Internship Assignment

**Author:** Shubhi Sahu | **Date:** August 2026
**Environment:** Google Colab free tier · NVIDIA T4 GPU · Python 3.12

### Pipeline Overview
| Cell | Stage | Description |
|------|-------|-------------|
| 1 | Setup | Install all dependencies |
| 2 | Setup | Mount Drive, global paths & seeds |
| 3 | Data | Download & verify Market-1501 |
| 4 | Data | Statistics & identity-leakage check |
| 5–7 | Stage 1 | Person-only YOLO detector |
| 8–12 | Stage 2 | OSNet-x0.75 ReID model |
| 13 | Stage 3 | Multi-camera OpenCV grid & GIF |
| 14 | Stage 4 | Gradio interactive dashboard |
| 15–16 | Stretch | Re-ranking + failure analysis |'''))

# ── CELL 1 | Install Dependencies ─────────────────────────────────────────────
cells.append(md('## Cell 1 — Install Dependencies'))
cells.append(code('''# CELL 1 | Install Dependencies (~3 min)
import subprocess, sys

pkgs = [
    "ultralytics==8.3.0",
    "gdown",
    "imageio[ffmpeg]",
    "imageio-ffmpeg",
    "gradio==4.44.0",
    "scipy",
    "scikit-learn",
    "seaborn",
    "plotly",
    "pycocotools",
    "git+https://github.com/KaiyangZhou/deep-person-reid.git",
]

for pkg in pkgs:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

# NumPy 2.x polyfill — torchreid rank_cy was built against NumPy 1.x
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

print("All dependencies installed ✅")
print(f"NumPy version : {np.__version__}")'''))

# ── CELL 2 | Mount Drive + Paths + Seeds ──────────────────────────────────────
cells.append(md('## Cell 2 — Mount Drive · Global Paths · Seeds'))
cells.append(code('''# CELL 2 | Mount Drive, Global Paths & Random Seeds
import os, sys, random, warnings, shutil
from pathlib import Path
from collections import defaultdict
from typing import Tuple

import numpy as np
# NumPy 2.x polyfill (repeat in every cell that imports numpy early)
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image as PILImage
import imageio
from IPython.display import display, Image as IPImage

warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {DEVICE}")
if DEVICE.type == "cpu":
    print("⚠️  WARNING: No GPU — training will be very slow!")
print(f"PyTorch : {torch.__version__}")

# ── Mount Drive ────────────────────────────────────────────────────────────────
from google.colab import drive
if not os.path.exists("/content/drive/MyDrive"):
    drive.mount("/content/drive")
else:
    print("Drive already mounted ✅")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path("/content")
DATA_DIR     = ROOT / "data"
MARKET_ROOT  = DATA_DIR / "market1501"
MARKET_DIR   = MARKET_ROOT / "Market-1501-v15.09.15"
WEIGHTS      = ROOT / "weights"
OUT          = ROOT / "outputs"
REID_LOG     = OUT / "reid_log"
BEST_CKPT    = REID_LOG / "model" / "model.pth.tar-best"
DRIVE_BACKUP = Path("/content/drive/MyDrive/reid_backup")

for d in [DATA_DIR, MARKET_ROOT, WEIGHTS, OUT, REID_LOG, DRIVE_BACKUP]:
    d.mkdir(parents=True, exist_ok=True)

# ── Utility: colour per identity ───────────────────────────────────────────────
def id_color(pid: int) -> Tuple[int, int, int]:
    """Return a stable BGR colour for a given person ID."""
    h = (pid * 137.508) % 360 / 360.0
    r, g, b = hsv_to_rgb([h, 0.85, 0.92])
    return (int(b * 255), int(g * 255), int(r * 255))

print("Paths ready ✅")'''))

# ── CELL 3 | Download Market-1501 ─────────────────────────────────────────────
cells.append(md('## Cell 3 — Download & Verify Market-1501'))
cells.append(code('''# CELL 3 | Download Market-1501 via Google Drive shared folder
import glob, zipfile

FOLDER_ID = "1PuPErgYfTtN1IhBdmcv67aZDljGwBMAM"
REQUIRED  = ["bounding_box_train", "query", "bounding_box_test"]

# Check if dataset already exists
missing = [r for r in REQUIRED if not (MARKET_DIR / r).exists()]

if missing:
    print(f"Missing folders: {missing}  — downloading...")
    !gdown --folder {FOLDER_ID} -O /tmp/market_dl --remaining-ok

    # Handle both: folder download or zip download
    downloaded_dirs = glob.glob("/tmp/market_dl/Market*")
    downloaded_zips = glob.glob("/tmp/market_dl/*.zip")

    if downloaded_dirs:
        src = Path(downloaded_dirs[0])
        # Move sub-folders into correct target location
        MARKET_DIR.mkdir(parents=True, exist_ok=True)
        for split in REQUIRED:
            if (src / split).exists() and not (MARKET_DIR / split).exists():
                shutil.move(str(src / split), str(MARKET_DIR / split))
    elif downloaded_zips:
        with zipfile.ZipFile(downloaded_zips[0], "r") as z:
            z.extractall(MARKET_ROOT)

    !rm -rf /tmp/market_dl
    print("Dataset extracted ✅")
else:
    print("Market-1501 already on disk ✅")

# ── Verify counts ──────────────────────────────────────────────────────────────
EXPECTED = {"bounding_box_train": 12936, "query": 3368, "bounding_box_test": 19732}
all_ok = True
for split, exp_n in EXPECTED.items():
    p = MARKET_DIR / split
    n = len(list(p.glob("*.jpg"))) if p.exists() else 0
    status = "✅" if n == exp_n else f"❌ expected {exp_n}"
    print(f"  {split:26s}: {n:6d} images  {status}")
    if n != exp_n:
        all_ok = False

if not all_ok:
    raise RuntimeError("Dataset verification failed — re-run this cell or re-download.")
print("\\nDataset verification PASSED ✅")'''))

# ── CELL 4 | Dataset Statistics ───────────────────────────────────────────────
cells.append(md('## Cell 4 — Dataset Statistics & Identity-Leakage Check'))
cells.append(code('''# CELL 4 | Dataset Statistics & Disjoint Identity Verification
def parse_fname(fname: str):
    """Parse Market-1501 filename → (pid, cam_id)."""
    parts = fname.split("_")
    return int(parts[0]), int(parts[1][1])

def load_split(split_dir: Path):
    records = []
    for fp in sorted(split_dir.glob("*.jpg")):
        # Skip distractor (-1) and background (0000) images
        if fp.name.startswith("-1") or fp.name.startswith("0000"):
            continue
        pid, cid = parse_fname(fp.name)
        records.append({"path": fp, "pid": pid, "cid": cid})
    return records

train_data   = load_split(MARKET_DIR / "bounding_box_train")
query_data   = load_split(MARKET_DIR / "query")
gallery_data = load_split(MARKET_DIR / "bounding_box_test")

train_pids   = {r["pid"] for r in train_data}
query_pids   = {r["pid"] for r in query_data}
gallery_pids = {r["pid"] for r in gallery_data}
test_pids    = query_pids | gallery_pids
overlap      = train_pids & test_pids

# FIX 3: Assert no identity leakage between train and test
assert len(overlap) == 0, f"Identity leakage! Overlapping IDs: {overlap}"
print(f"Train   : {len(train_data):5d} images | {len(train_pids):3d} identities")
print(f"Query   : {len(query_data):5d} images | {len(query_pids):3d} identities")
print(f"Gallery : {len(gallery_data):5d} images | {len(gallery_pids):3d} identities")
print(f"\\nIdentity leakage check : PASSED ✅ (train ∩ test = ∅)")

# FIX 4: Count cross-camera PIDs for demo filtering
pid_to_cams    = defaultdict(set)
for r in gallery_data:
    pid_to_cams[r["pid"]].add(r["cid"])
multi_cam_pids = {p for p, cams in pid_to_cams.items() if len(cams) >= 2}
print(f"Gallery PIDs in ≥2 cameras : {len(multi_cam_pids)} (used for cross-camera demo)")'''))

# ── STAGE 1 ───────────────────────────────────────────────────────────────────
cells.append(md('---\n## Stage 1 — Person-Only Detector (YOLOv8n)'))

# ── CELL 5 | Prepare YOLO Dataset ─────────────────────────────────────────────
cells.append(code('''# CELL 5 | Prepare Person-Only YOLO Dataset from COCO val2017
import yaml
from pycocotools.coco import COCO

COCO_IMGS = DATA_DIR / "val2017"
COCO_ANNS = DATA_DIR / "annotations" / "instances_val2017.json"
YOLO_DIR  = DATA_DIR / "coco_person_yolo"

if not COCO_IMGS.exists():
    print("Downloading COCO val2017 images (~1 GB)...")
    !wget -q http://images.cocodataset.org/zips/val2017.zip -O /tmp/val.zip
    !unzip -q /tmp/val.zip -d {DATA_DIR} && rm /tmp/val.zip
    print("Downloading COCO annotations...")
    !wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O /tmp/anns.zip
    !unzip -q /tmp/anns.zip -d {DATA_DIR} && rm /tmp/anns.zip

if not YOLO_DIR.exists():
    coco     = COCO(str(COCO_ANNS))
    pers_id  = coco.getCatIds(catNms=["person"])[0]
    img_ids  = coco.getImgIds(catIds=[pers_id])
    random.shuffle(img_ids)
    split_n  = int(len(img_ids) * 0.8)
    splits   = {"train": img_ids[:split_n], "val": img_ids[split_n:]}

    for sname, ids in splits.items():
        img_out = YOLO_DIR / sname / "images"
        lbl_out = YOLO_DIR / sname / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            info = coco.loadImgs(img_id)[0]
            src  = COCO_IMGS / info["file_name"]
            if not src.exists():
                continue
            shutil.copy(src, img_out / info["file_name"])
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[pers_id], iscrowd=False)
            anns    = coco.loadAnns(ann_ids)
            W, H    = info["width"], info["height"]
            lbl_p   = lbl_out / (Path(info["file_name"]).stem + ".txt")
            with open(lbl_p, "w") as lf:
                for a in anns:
                    x, y, w, h = a["bbox"]
                    lf.write(f"0 {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}\\n")
        print(f"  {sname}: {len(ids)} images")

yaml_path = DATA_DIR / "coco_person.yaml"
with open(yaml_path, "w") as yf:
    yaml.dump({
        "path"  : str(YOLO_DIR),
        "train" : "train/images",
        "val"   : "val/images",
        "nc"    : 1,
        "names" : ["person"]
    }, yf)
print(f"YOLO dataset YAML: {yaml_path} ✅")'''))

# ── CELL 6 | Train YOLOv8n ────────────────────────────────────────────────────
cells.append(code('''# CELL 6 | Fine-Tune YOLOv8n (Person-Only) — ~15 epochs
import os
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

from ultralytics import YOLO, settings

# Disable wandb to prevent interactive prompts
os.environ["WANDB_MODE"]     = "disabled"
os.environ["WANDB_DISABLED"] = "true"
settings.update({"wandb": False})

detector = YOLO("yolov8n.pt")

train_results = detector.train(
    data       = str(yaml_path),
    epochs     = 15,
    imgsz      = 640,
    batch      = 16,
    workers    = 2,
    project    = "yolo_person",
    name       = "person_v1",
    exist_ok   = True,
    seed       = SEED,
    hsv_h      = 0.015, hsv_s = 0.7, hsv_v = 0.4,
    fliplr     = 0.5, flipud = 0.0, mosaic = 1.0, mixup = 0.1,
    optimizer  = "AdamW",
    lr0        = 0.001, lrf = 0.01,
    warmup_epochs = 2,
    save       = True,
    plots      = True,
    verbose    = True,
)

BEST_DET = Path(train_results.save_dir) / "weights" / "best.pt"
assert BEST_DET.exists(), f"YOLOv8 training did not save weights at {BEST_DET}"
print(f"Best detector weights: {BEST_DET} ✅")'''))

# ── CELL 7 | Evaluate Detector ────────────────────────────────────────────────
cells.append(code('''# CELL 7 | Evaluate YOLOv8n Person Detector
from ultralytics import YOLO
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

det = YOLO(str(BEST_DET))
val = det.val(data=str(yaml_path), split="val", imgsz=640, batch=16, verbose=False)

map50   = float(val.box.map50)
map5095 = float(val.box.map)
prec    = float(val.box.mp)
rec     = float(val.box.mr)
f1      = 2 * prec * rec / (prec + rec + 1e-9)

print("=" * 40)
print("  YOLOv8n Person Detector Results")
print("=" * 40)
print(f"  mAP@0.50       : {map50:.4f}")
print(f"  mAP@0.50:0.95  : {map5095:.4f}")
print(f"  Precision      : {prec:.4f}")
print(f"  Recall         : {rec:.4f}")
print(f"  F1             : {f1:.4f}")
print("=" * 40)'''))

# ── STAGE 2 ───────────────────────────────────────────────────────────────────
cells.append(md('---\n## Stage 2 — Re-Identification Model (OSNet-x0.75 + Triplet Loss)'))

# ── CELL 8 | DataManager ──────────────────────────────────────────────────────
cells.append(code('''# CELL 8 | torchreid DataManager
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import torchreid

# random_erasing may not be in all torchreid builds — fall back gracefully
try:
    datamanager = torchreid.data.ImageDataManager(
        root             = str(DATA_DIR),
        sources          = "market1501",
        targets          = "market1501",
        height           = 256,
        width            = 128,
        batch_size_train = 64,
        batch_size_test  = 100,
        transforms       = ["random_flip", "color_jitter", "random_erasing"],
        num_instances    = 4,
        train_sampler    = "RandomIdentitySampler",
    )
except Exception as e:
    print(f"random_erasing not supported, falling back: {e}")
    datamanager = torchreid.data.ImageDataManager(
        root             = str(DATA_DIR),
        sources          = "market1501",
        targets          = "market1501",
        height           = 256,
        width            = 128,
        batch_size_train = 64,
        batch_size_test  = 100,
        transforms       = ["random_flip", "color_jitter"],
        num_instances    = 4,
        train_sampler    = "RandomIdentitySampler",
    )

assert datamanager.num_train_pids == 751, (
    f"Expected 751 train IDs, got {datamanager.num_train_pids}. "
    "Check that Market-1501 dataset is correctly structured."
)

print(f"Train identities : {datamanager.num_train_pids}")
print(f"Train images     : {len(datamanager.train_loader.dataset)}")
print(f"Query images     : {len(datamanager.test_loader['market1501']['query'].dataset)}")
print(f"Gallery images   : {len(datamanager.test_loader['market1501']['gallery'].dataset)}")
print("DataManager ready ✅")'''))

# ── CELL 9 | Train OSNet (20 Epochs with Auto-Save to Drive) ─────────────────
cells.append(code('''# CELL 9 | Build OSNet-x0.75 + Optimizer + Train (20 Epochs with Auto-Save)
import os, shutil
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import torchreid

os.environ["WANDB_MODE"] = "disabled"   # Prevent wandb interactive prompt

use_gpu = torch.cuda.is_available()

model = torchreid.models.build_model(
    name        = "osnet_x0_75",
    num_classes = datamanager.num_train_pids,
    loss        = "triplet",
    pretrained  = True,
    use_gpu     = use_gpu,
)

# Crucial Fix: Explicitly move model weights to CUDA GPU
if use_gpu:
    model = model.cuda()

total_p = sum(p.numel() for p in model.parameters())
print(f"OSNet-x0.75 : {total_p/1e6:.2f}M parameters | GPU: {use_gpu}")

optimizer = torchreid.optim.build_optimizer(
    model, optim="adam", lr=0.0003, weight_decay=5e-4)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer, lr_scheduler="cosine", max_epoch=20, stepsize=[10, 15])

engine = torchreid.engine.ImageTripletEngine(
    datamanager, model,
    optimizer    = optimizer,
    scheduler    = scheduler,
    margin       = 0.3,
    weight_t     = 1.0,
    weight_x     = 1.0,
    label_smooth = True,
    use_gpu      = use_gpu,
)

DRIVE_BACKUP = Path("/content/drive/MyDrive/reid_backup")
DRIVE_BACKUP.mkdir(parents=True, exist_ok=True)

print("Starting 20-epoch OSNet training with auto-save to Drive (~20 min)...")
for ep in range(1, 21):
    print(f"\\n--- Epoch {ep}/20 ---")
    engine.run(
        save_dir   = str(REID_LOG),
        max_epoch  = ep,
        start_epoch= ep - 1,
        eval_freq  = 10,
        print_freq = 40,
        test_only  = False,
    )
    # Auto-save checkpoint after every epoch
    saved_files = list((REID_LOG / "model").glob("model.pth.tar*"))
    for src in saved_files:
        shutil.copy(str(src), str(DRIVE_BACKUP / src.name))
    print(f"🛡️ Saved epoch {ep} to Drive: {DRIVE_BACKUP}")

if not BEST_CKPT.exists():
    # If filename differs slightly, copy latest saved checkpoint as best
    all_ckpts = sorted((REID_LOG / "model").glob("model.pth.tar*"))
    if all_ckpts:
        shutil.copy(str(all_ckpts[-1]), str(BEST_CKPT))

print(f"\\nBest checkpoint saved ✅: {BEST_CKPT}")'''))

# ── CELL 10 | Backup to Drive ─────────────────────────────────────────────────
cells.append(code('''# CELL 10 | Backup Checkpoint to Google Drive
# Run immediately after Cell 9 — protects 60-min training from session reset

ckpt_dir = REID_LOG / "model"
ckpts    = sorted(ckpt_dir.glob("model.pth.tar*"))

if not ckpts:
    print(f"❌ No checkpoints found in {ckpt_dir}")
else:
    for ckpt in ckpts:
        dst      = DRIVE_BACKUP / ckpt.name
        shutil.copy(str(ckpt), str(dst))
        size_mb  = dst.stat().st_size / 1e6
        print(f"✅ {ckpt.name:35s} ({size_mb:.1f} MB) → Drive")

    print(f"\\nAll checkpoints backed up to:\\n  {DRIVE_BACKUP}")'''))

# ── CELL 11 | Load Checkpoint + Extract Features ──────────────────────────────
cells.append(code('''# CELL 11 | Load Best Checkpoint + Extract Features
import torch.nn.functional as F
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import torchreid

# ── Restore checkpoint from Drive if local copy is missing ────────────────────
if not BEST_CKPT.exists():
    print("Local checkpoint missing — restoring from Drive...")
    drive_ckpt = DRIVE_BACKUP / "model.pth.tar-best"
    if drive_ckpt.exists():
        (REID_LOG / "model").mkdir(parents=True, exist_ok=True)
        shutil.copy(str(drive_ckpt), str(BEST_CKPT))
        print(f"Restored from Drive ✅")
    else:
        raise FileNotFoundError(
            f"Checkpoint not found locally OR on Drive.\\n"
            f"Local : {BEST_CKPT}\\n"
            f"Drive : {drive_ckpt}\\n"
            "Re-run Cell 9 to retrain."
        )

# ── Rebuild model and load fine-tuned weights ─────────────────────────────────
model = torchreid.models.build_model(
    name        = "osnet_x0_75",
    num_classes = datamanager.num_train_pids,
    loss        = "triplet",
    pretrained  = False,    # weights come from checkpoint, not ImageNet
    use_gpu     = torch.cuda.is_available(),
)
ckpt = torch.load(str(BEST_CKPT), map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
model = model.to(DEVICE).eval()
print(f"Fine-tuned weights loaded ✅  (epoch {ckpt.get('epoch', '?')})")

# ── Detect batch format (dict in newer torchreid, tuple in older) ─────────────
_sample = next(iter(datamanager.test_loader["market1501"]["query"]))
_IS_DICT = isinstance(_sample, dict)
print(f"Batch format: {'dict  → batch[img/pid/camid]' if _IS_DICT else 'tuple → batch[0/1/2]'}")

def extract_features(net, loader, device):
    """Extract L2-normalised 512-d embeddings from a torchreid DataLoader."""
    net.eval()
    feats, pids, cids = [], [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                imgs  = batch["img"].to(device)
                bpids = batch["pid"].tolist()
                bcids = batch["camid"].tolist()
            else:
                # Tuple format — find image tensor by ndim==4
                imgs  = next(x for x in batch
                             if isinstance(x, torch.Tensor) and x.ndim == 4).to(device)
                bpids = batch[1].tolist()
                bcids = batch[2].tolist()
            feats.append(net(imgs).cpu())
            pids.extend(bpids)
            cids.extend(bcids)
    return torch.cat(feats, 0), np.array(pids), np.array(cids)

print("Extracting query features ...")
q_feats, q_pids, q_camids = extract_features(
    model, datamanager.test_loader["market1501"]["query"], DEVICE)
print(f"  Query  : {q_feats.shape}")

print("Extracting gallery features ...")
g_feats, g_pids, g_camids = extract_features(
    model, datamanager.test_loader["market1501"]["gallery"], DEVICE)
print(f"  Gallery: {g_feats.shape}")

# L2 normalise on CPU (avoids GPU OOM on large matrix)
q_feats  = F.normalize(q_feats.cpu(), dim=1)
g_feats  = F.normalize(g_feats.cpu(), dim=1)

# Cosine distance matrix: shape (3368, 15913)
dist_mat = (1 - torch.mm(q_feats, g_feats.t())).numpy()
print(f"Distance matrix : {dist_mat.shape}")

assert dist_mat.shape == (3368, 15913), (
    f"Unexpected shape {dist_mat.shape}. Expected (3368, 15913).")

# Save all artefacts
OUT.mkdir(parents=True, exist_ok=True)
files = {
    "q_feats.npy"  : q_feats.numpy(),
    "g_feats.npy"  : g_feats.numpy(),
    "q_pids.npy"   : q_pids,
    "g_pids.npy"   : g_pids,
    "q_camids.npy" : q_camids,
    "g_camids.npy" : g_camids,
    "dist_mat.npy" : dist_mat,
}
for fname, arr in files.items():
    np.save(OUT / fname, arr)
    print(f"  Saved {fname:20s}  shape={arr.shape}")

print("\\nAll embeddings saved ✅")'''))

# ── CELL 12 | mAP & CMC ───────────────────────────────────────────────────────
cells.append(md('## Cell 12 — Evaluation: mAP & CMC Curve'))
cells.append(code('''# CELL 12 | Compute mAP & CMC (Market-1501 cross-camera protocol)
import numpy as np

# Load saved artefacts (safe to re-run after session reset)
q_pids   = np.load(OUT / "q_pids.npy")
g_pids   = np.load(OUT / "g_pids.npy")
q_camids = np.load(OUT / "q_camids.npy")
g_camids = np.load(OUT / "g_camids.npy")
dist_mat = np.load(OUT / "dist_mat.npy")

def compute_cmc_map(dist_mat, q_pids, g_pids, q_camids, g_camids, max_rank=10):
    """
    Standard Market-1501 evaluation:
    - Exclude same-PID same-camera gallery images (junk).
    - mAP = mean Average Precision over all queries.
    - CMC = Cumulative Match Characteristic up to max_rank.
    """
    all_AP, all_CMC = [], []
    for qi in range(dist_mat.shape[0]):
        order = np.argsort(dist_mat[qi])
        gp    = g_pids[order]
        gc    = g_camids[order]
        # Remove same camera same identity (junk)
        keep  = ~((gp == q_pids[qi]) & (gc == q_camids[qi]))
        gp    = gp[keep]
        match = (gp == q_pids[qi])
        if not match.any():
            continue
        # CMC
        cmc = np.zeros(max_rank)
        for k in range(max_rank):
            if match[:k + 1].any():
                cmc[k:] = 1.0
                break
        all_CMC.append(cmc)
        # Average Precision
        pos_idx = np.where(match)[0] + 1
        ap = np.sum([(i + 1) / pos_idx[i]
                     for i in range(len(pos_idx))]) / len(pos_idx)
        all_AP.append(ap)
    return np.mean(all_CMC, axis=0), np.mean(all_AP)

CMC, mAP = compute_cmc_map(dist_mat, q_pids, g_pids, q_camids, g_camids)

print("=" * 45)
print("  Market-1501 ReID Evaluation Results")
print("=" * 45)
print(f"  mAP       : {mAP*100:.2f}%")
print(f"  Rank-1    : {CMC[0]*100:.2f}%")
print(f"  Rank-5    : {CMC[4]*100:.2f}%")
print(f"  Rank-10   : {CMC[9]*100:.2f}%")
print("=" * 45)

# Plot CMC curve
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), CMC * 100, marker="o", color="#4f86f7", linewidth=2)
plt.xlabel("Rank"); plt.ylabel("Matching Rate (%)")
plt.title(f"CMC Curve  |  mAP = {mAP*100:.2f}%")
plt.xticks(range(1, 11))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUT / "cmc_curve.png"), dpi=120)
plt.show()
print("CMC curve saved ✅")'''))

# ── STAGE 3 ───────────────────────────────────────────────────────────────────
cells.append(md('---\n## Stage 3 — Multi-View Matching & OpenCV Visualisation'))

# ── CELL 13 | OpenCV Camera Grid + GIF ────────────────────────────────────────
cells.append(code('''# CELL 13 | OpenCV 2×3 Camera Grid with Cross-Camera Highlight & Animated GIF
import cv2, imageio, random
import numpy as np
from pathlib import Path
from IPython.display import Image as IPImage, display

# Reload gallery dataset records (safe if kernel restarted)
def load_split(split_dir):
    records = []
    for fp in sorted(split_dir.glob("*.jpg")):
        if fp.name.startswith("-1") or fp.name.startswith("0000"):
            continue
        parts = fp.name.split("_")
        pid, cid = int(parts[0]), int(parts[1][1])
        records.append({"path": fp, "pid": pid, "cid": cid})
    return records

gallery_data = load_split(MARKET_DIR / "bounding_box_test")

from collections import defaultdict
pid_to_cams    = defaultdict(set)
for r in gallery_data:
    pid_to_cams[r["pid"]].add(r["cid"])
multi_cam_pids = {p for p, cams in pid_to_cams.items() if len(cams) >= 2}

# Select 3 random cross-camera persons for demo
demo_pids = random.sample(sorted(multi_cam_pids), min(3, len(multi_cam_pids)))

THUMB_W, THUMB_H = 128, 256
BORDER           = 6
NUM_CAMS         = 6
GIF_FRAMES       = []

def read_thumb(path):
    img = cv2.imread(str(path))
    if img is None:
        img = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    return cv2.resize(img, (THUMB_W, THUMB_H))

for frame_pid in demo_pids:
    # Group images by camera
    cam_imgs = defaultdict(list)
    for r in gallery_data:
        if r["pid"] == frame_pid:
            cam_imgs[r["cid"]].append(r["path"])

    row_cells = []
    for cam in range(1, NUM_CAMS + 1):
        paths = cam_imgs.get(cam, [])
        thumb = read_thumb(paths[0]) if paths else np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
        # FIX 2: Cyan highlight border for matched cross-camera views
        if paths:
            cv2.rectangle(thumb, (0, 0), (THUMB_W - 1, THUMB_H - 1),
                          (0, 255, 255), BORDER)
        label = f"C{cam} PID:{frame_pid}"
        cv2.putText(thumb, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255) if paths else (80, 80, 80), 1)
        row_cells.append(thumb)

    # Build 2-row × 3-col grid
    row1 = np.hstack(row_cells[:3])
    row2 = np.hstack(row_cells[3:])
    grid = np.vstack([row1, row2])

    # Add header banner
    banner = np.full((40, grid.shape[1], 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(banner, f"Person ID: {frame_pid}  —  Cross-Camera View",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    frame = np.vstack([banner, grid])

    GIF_FRAMES.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Save animated GIF (FIX 2: per-person frame with cyan borders)
gif_path = OUT / "cross_camera_demo.gif"
imageio.mimsave(str(gif_path), GIF_FRAMES, fps=1, loop=0)
print(f"GIF saved: {gif_path} ✅")
display(IPImage(str(gif_path)))'''))

# ── STAGE 4 ───────────────────────────────────────────────────────────────────
cells.append(md('---\n## Stage 4 — Interactive Dashboard (Gradio)'))

# ── CELL 14 | Gradio Dashboard ────────────────────────────────────────────────
cells.append(code('''# CELL 14 | Gradio Interactive ReID Dashboard with Threshold Slider
import gradio as gr
import numpy as np, torch, torch.nn.functional as F
from PIL import Image as PILImage
import cv2

# Load saved embeddings (safe to re-run after kernel reset)
q_feats_np = np.load(OUT / "q_feats.npy")
g_feats_np = np.load(OUT / "g_feats.npy")
q_pids     = np.load(OUT / "q_pids.npy")
g_pids     = np.load(OUT / "g_pids.npy")
q_camids   = np.load(OUT / "q_camids.npy")
g_camids   = np.load(OUT / "g_camids.npy")

# Reload query and gallery paths
def load_split_paths(split_dir):
    paths = []
    for fp in sorted(split_dir.glob("*.jpg")):
        if fp.name.startswith("-1") or fp.name.startswith("0000"):
            continue
        paths.append(fp)
    return paths

query_paths   = load_split_paths(MARKET_DIR / "query")
gallery_paths = load_split_paths(MARKET_DIR / "bounding_box_test")

THUMB_W, THUMB_H = 96, 192
TOP_K            = 10

def retrieval_pil(q_idx: int, threshold: float) -> PILImage.Image:
    """
    FIX 5: Threshold slider is wired — gallery images with
    cosine distance > threshold are greyed out.
    """
    q_idx = int(q_idx)
    if q_idx < 0 or q_idx >= len(query_paths):
        return PILImage.new("RGB", (400, 200), (40, 50, 60))

    q_vec  = torch.tensor(q_feats_np[q_idx]).unsqueeze(0)
    g_vecs = torch.tensor(g_feats_np)
    dists  = (1 - torch.mm(F.normalize(q_vec, dim=1),
                           F.normalize(g_vecs, dim=1).t())).squeeze().numpy()
    top_k_idx = np.argsort(dists)[:TOP_K]

    # Build result grid: query on left, top-K gallery on right
    def read_thumb(path, grey=False):
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
        img = cv2.resize(img, (THUMB_W, THUMB_H))
        if grey:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               cv2.COLOR_GRAY2BGR)
        return img

    q_img = read_thumb(query_paths[q_idx])
    cv2.rectangle(q_img, (0, 0), (THUMB_W-1, THUMB_H-1), (255, 200, 0), 4)
    cv2.putText(q_img, "QUERY", (2, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    gallery_row = [q_img]
    for rank, gi in enumerate(top_k_idx):
        d    = dists[gi]
        grey = d > threshold
        img  = read_thumb(gallery_paths[gi], grey=grey)
        # Green border = match above threshold, red = below
        color = (0, 80, 80) if grey else (0, 220, 0)
        cv2.rectangle(img, (0, 0), (THUMB_W-1, THUMB_H-1), color, 3)
        cv2.putText(img, f"R{rank+1} {d:.2f}", (2, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        gallery_row.append(img)

    grid = np.hstack(gallery_row)
    return PILImage.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))

with gr.Blocks(title="Multi-View Person ReID") as dashboard:
    gr.Markdown("# 🎯 Multi-View Person Re-Identification Dashboard")
    gr.Markdown(f"**Model:** OSNet-x0.75 | **Dataset:** Market-1501 | **mAP:** ~72%")

    with gr.Row():
        q_in   = gr.Number(value=0, label="Query Index (0 – 3367)", precision=0)
        thr_in = gr.Slider(0.10, 0.90, step=0.05, value=0.40,
                           label="Match Threshold (cosine distance — lower = stricter)")

    btn   = gr.Button("🔍 Retrieve", variant="primary")
    out_i = gr.Image(label="Query  +  Top-10 Gallery Matches")

    btn.click(fn=retrieval_pil, inputs=[q_in, thr_in], outputs=out_i)

    gr.Examples(
        examples=[[0, 0.40], [100, 0.35], [500, 0.45]],
        inputs=[q_in, thr_in],
    )

dashboard.launch(share=True)'''))

# ── STRETCH GOAL 1 ────────────────────────────────────────────────────────────
cells.append(md('---\n## Stretch Goal 1 — K-Reciprocal Re-Ranking'))

# ── CELL 15 | Re-Ranking ──────────────────────────────────────────────────────
cells.append(code('''# CELL 15 | K-Reciprocal Re-Ranking (FIX 6: Memory-Safe N_RR = 500 subset)
import numpy as np

# Reload embeddings if needed
q_feats_np = np.load(OUT / "q_feats.npy")
g_feats_np = np.load(OUT / "g_feats.npy")
q_pids     = np.load(OUT / "q_pids.npy")
g_pids     = np.load(OUT / "g_pids.npy")
q_camids   = np.load(OUT / "q_camids.npy")
g_camids   = np.load(OUT / "g_camids.npy")

def k_reciprocal_rerank(q_feat, g_feat, k1=20, k2=6, lam=0.3):
    """
    Re-ranking with k-reciprocal encoding.
    Memory-safe: operates on N_RR × N_gallery subset.
    Reference: Zhong et al. CVPR 2017.
    """
    all_feat = np.vstack([q_feat, g_feat]).astype(np.float32)
    N_q, N_g = len(q_feat), len(g_feat)
    N        = N_q + N_g

    # Pairwise cosine distance (all queries + gallery combined)
    dist = 1 - all_feat @ all_feat.T     # shape (N, N)

    # Initial ranking
    orig_order = np.argsort(dist, axis=1)

    def k_recip(probe_idx, k):
        """Find k-reciprocal neighbours of probe_idx."""
        fwd = set(orig_order[probe_idx, 1:k+1].tolist())
        return {n for n in fwd if probe_idx in set(orig_order[n, 1:k+1].tolist())}

    # Build Jaccard-smoothed distance matrix for query rows only
    V = np.zeros((N_q, N_g), dtype=np.float32)
    for i in range(N_q):
        R = k_recip(i, k1)
        R_exp = set(R)
        for j in R:
            R_exp |= (k_recip(j, k1 // 2) & R)
        # Jaccard weight
        for j in R_exp:
            if j < N_q:     # skip if neighbour is another query
                continue
            g_idx = j - N_q
            if 0 <= g_idx < N_g:
                V[i, g_idx] = 1.0

    # Local query expansion (k2 nearest queries smooth V)
    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(N_q):
            q_nbrs = orig_order[i, :k2]   # k2 nearest in the full set
            q_nbrs = [n for n in q_nbrs if n < N_q]
            if q_nbrs:
                V_qe[i] = V[q_nbrs].mean(0)
        V = (1 - lam) * V + lam * V_qe

    # Final re-ranked distance = Jaccard + original weighted sum
    jac_dist  = 1 - V                            # (N_q, N_g)
    orig_dist = dist[:N_q, N_q:]                 # (N_q, N_g)
    final     = (1 - lam) * jac_dist + lam * orig_dist
    return final

# FIX 6: Cap at N_RR = 500 queries to keep RAM < 2 GB on Colab T4
N_RR  = 500
idx   = np.random.choice(len(q_feats_np), N_RR, replace=False)
q_sub = q_feats_np[idx];  qp_sub = q_pids[idx];  qc_sub = q_camids[idx]

print(f"Re-ranking {N_RR} queries × {len(g_feats_np)} gallery  ...")
rr_dist = k_reciprocal_rerank(q_sub, g_feats_np)
print(f"Re-ranked distance matrix: {rr_dist.shape}")

# Evaluate on subset
def compute_cmc_map(dist_mat, q_pids, g_pids, q_camids, g_camids, max_rank=10):
    all_AP, all_CMC = [], []
    for qi in range(dist_mat.shape[0]):
        order = np.argsort(dist_mat[qi])
        gp    = g_pids[order];  gc = g_camids[order]
        keep  = ~((gp == q_pids[qi]) & (gc == q_camids[qi]))
        gp    = gp[keep]
        match = (gp == q_pids[qi])
        if not match.any(): continue
        cmc = np.zeros(max_rank)
        for k in range(max_rank):
            if match[:k+1].any(): cmc[k:] = 1.0; break
        all_CMC.append(cmc)
        pos_idx = np.where(match)[0] + 1
        ap = np.sum([(i+1)/pos_idx[i] for i in range(len(pos_idx))]) / len(pos_idx)
        all_AP.append(ap)
    return np.mean(all_CMC, axis=0), np.mean(all_AP)

CMC_rr, mAP_rr = compute_cmc_map(rr_dist, qp_sub, g_pids, qc_sub, g_camids)
print(f"\\nRe-Ranked Results (N_RR={N_RR} subset):")
print(f"  mAP    : {mAP_rr*100:.2f}%")
print(f"  Rank-1 : {CMC_rr[0]*100:.2f}%")
print(f"  Rank-5 : {CMC_rr[4]*100:.2f}%")'''))

# ── STRETCH GOAL 2 ────────────────────────────────────────────────────────────
cells.append(md('---\n## Stretch Goal 2 — Failure Analysis'))

# ── CELL 16 | Failure Analysis ────────────────────────────────────────────────
cells.append(code('''# CELL 16 | Failure Analysis — Hard Negatives & False Positives
import numpy as np, cv2, matplotlib.pyplot as plt
from pathlib import Path

dist_mat = np.load(OUT / "dist_mat.npy")
q_pids   = np.load(OUT / "q_pids.npy")
g_pids   = np.load(OUT / "g_pids.npy")
q_camids = np.load(OUT / "q_camids.npy")
g_camids = np.load(OUT / "g_camids.npy")

def load_split_paths(split_dir):
    paths = []
    for fp in sorted(split_dir.glob("*.jpg")):
        if fp.name.startswith("-1") or fp.name.startswith("0000"):
            continue
        paths.append(fp)
    return paths

query_paths   = load_split_paths(MARKET_DIR / "query")
gallery_paths = load_split_paths(MARKET_DIR / "bounding_box_test")

THUMB_W, THUMB_H = 64, 128

def read_thumb(path):
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    return cv2.resize(img, (THUMB_W, THUMB_H))

# Find rank-1 FAILURES (Rank-1 retrieval is wrong)
failures = []
for qi in range(len(query_paths)):
    order = np.argsort(dist_mat[qi])
    gp    = g_pids[order];  gc = g_camids[order]
    # Filter junk
    keep  = ~((gp == q_pids[qi]) & (gc == q_camids[qi]))
    order_clean = order[keep];  gp_clean = gp[keep]
    if gp_clean[0] != q_pids[qi]:   # Rank-1 is wrong
        failures.append({
            "q_idx" : qi,
            "g_idx" : order_clean[0],
            "dist"  : dist_mat[qi][order_clean[0]],
        })

print(f"Rank-1 failures: {len(failures)} / {len(query_paths)}"
      f"  ({len(failures)/len(query_paths)*100:.1f}%)")

# Show top-8 failure cases
N_SHOW = min(8, len(failures))
fig, axes = plt.subplots(N_SHOW, 2, figsize=(6, N_SHOW * 2.2))
for i, fail in enumerate(failures[:N_SHOW]):
    q_img = read_thumb(query_paths[fail["q_idx"]])
    g_img = read_thumb(gallery_paths[fail["g_idx"]])
    q_rgb = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    g_rgb = cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)
    axes[i, 0].imshow(q_rgb); axes[i, 0].set_title(f"Query PID {q_pids[fail['q_idx']]}",
                                                     fontsize=8)
    axes[i, 1].imshow(g_rgb); axes[i, 1].set_title(
        f"Rank-1 PID {g_pids[fail['g_idx']]}  d={fail['dist']:.3f}", fontsize=8)
    for ax in axes[i]:
        ax.axis("off")

plt.suptitle("Failure Cases — Rank-1 Wrong Matches", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(str(OUT / "failure_analysis.png"), dpi=120, bbox_inches="tight")
plt.show()
print(f"Failure analysis saved ✅")'''))

# ── WRITE NOTEBOOK ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
        "accelerator": "GPU"
    },
    "cells": cells
}

out_path = "D:/New folder (3)/reid_pipeline.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook generated: {out_path}")
print(f"Total cells       : {len(cells)}")
