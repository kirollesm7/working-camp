"""
Clothes Counter — معسكر العمل 2026 (منه وله)

A phone camera watches the boxes. When the "منه وله" sticker is seen and
held steady, the app takes one photo, straightens it, and reads:

  * which total box is ticked (30 / 40 / 50 / other)  — ink density, reliable
  * the handwritten numbers for أولاد / بنات / رجال / سيدات
    — offline digit recogniser (HOG + KNN) that learns from every correction
  * the location field is kept as an image; the operator picks the place

Every read is shown for review before it is saved.
"""

import os
import sys
import csv
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRectF, QPointF
from PySide6.QtGui import (
    QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPixmap,
    QBrush, QPolygonF, QKeySequence, QShortcut
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QVBoxLayout, QHBoxLayout, QGridLayout, QStackedWidget, QSpinBox,
    QComboBox, QCheckBox, QLineEdit, QDialog, QMessageBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / "assets"
# CLOTHES_DATA_DIR lets a test run use a throwaway folder
DATA_DIR = Path(
    os.environ.get("CLOTHES_DATA_DIR") or APP_DIR / "data" / "clothes"
)

TEMPLATE_FILE = ASSETS / "clothes_sticker.png"
BOXES_FILE = DATA_DIR / "boxes.csv"
CONFIG_FILE = DATA_DIR / "config.json"
IMAGES_DIR = DATA_DIR / "images"
SAMPLES_DIR = DATA_DIR / "digit_samples"
LOCATIONS_FILE = DATA_DIR / "locations.json"

# Category cells stay empty when nothing was written on the sticker.
# "check" = 1 when a number was read with low confidence.
BOX_FIELDS = [
    "id", "time", "total", "boys", "girls", "men", "women",
    "location", "source", "image", "check",
]


# ============================================================
# IDENTITY
# ============================================================

KRAFT = "#E1CDAD"
KRAFT_DARK = "#C9B08A"
PAPER = "#F6EEDF"
INK = "#1E1B18"
INK_SOFT = "#5B4E3C"
AIR_RED = "#D61B2D"
AIR_BLUE = "#055F9E"
OK_GREEN = "#2E7D32"
WARN = "#E08A00"

FONT = "'Cairo','Tajawal','Segoe UI',Tahoma"

TITLE_AR = "عدّاد الهدوم"
CAMP_AR = "معسكر العمل 2026"
MOTTO = "منه وله"


# ============================================================
# STICKER GEOMETRY (pixels on the 1168 x 812 template)
# ============================================================

# Read right-to-left: ☐٣٠ ☐٤٠ ☐٥٠ ☐.......
CHECKBOXES = [
    (30, (727, 494, 33, 34)),
    (40, (599, 495, 33, 33)),
    (50, (470, 494, 33, 34)),
    ("other", (336, 495, 33, 33)),
]

# Handwritten total written over the dots next to the 4th box
OTHER_FIELD = (242, 486, 90, 52)

CATEGORIES = [
    ("boys", "أولاد", (773, 600, 130, 51)),
    ("girls", "بنات", (614, 600, 130, 51)),
    ("men", "رجال", (456, 599, 130, 51)),
    ("women", "سيدات", (299, 600, 130, 51)),
]

LOCATION_FIELD = (411, 686, 282, 60)

DEFAULT_CONFIG = {
    # "0" = laptop camera. Phone: IP Webcam -> http://<ip>:8080/video
    #                             DroidCam  -> http://<ip>:4747/video
    "camera": "0",
    "digits": "english",         # english (012…) or arabic (٠١٢…)
    "auto_save": False,          # save without review when every field is confident
    "phone_camera": True,        # False = phones stop the camera and use manual entry
    "reader": "ensemble",        # ensemble (CNN + classic) | cnn | classic
    "read_grey": True,           # read captures in grey-scale (more reliable)
}


# ============================================================
# STORAGE
# ============================================================

def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    return cfg


def save_config(cfg):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_locations():
    try:
        return list(json.loads(LOCATIONS_FILE.read_text(encoding="utf-8")))
    except Exception:
        return []


def save_locations(locs):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOCATIONS_FILE.write_text(
        json.dumps(locs, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_boxes():
    if not BOXES_FILE.exists():
        return []
    try:
        with BOXES_FILE.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_boxes(rows):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with BOXES_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BOX_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def append_box(row):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Upgrade files written before a column was added
    if BOXES_FILE.exists():
        with BOXES_FILE.open("r", encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        if header != BOX_FIELDS:
            write_boxes(load_boxes())
    new_file = not BOXES_FILE.exists()
    with BOXES_FILE.open(
        "a", encoding="utf-8-sig" if new_file else "utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=BOX_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def to_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def box_stats(rows):
    stats = {
        "boxes": len(rows),
        "pieces": 0,
        "unsorted": 0,
        "by_location": defaultdict(int),
    }
    for key, _, _ in CATEGORIES:
        stats[key] = 0
    for r in rows:
        total = to_int(r.get("total"))
        cats = sum(to_int(r.get(k)) for k, _, _ in CATEGORIES)
        stats["pieces"] += total
        stats["unsorted"] += max(0, total - cats)
        for key, _, _ in CATEGORIES:
            stats[key] += to_int(r.get(key))
        stats["by_location"][r.get("location") or "—"] += total
    return stats


# ============================================================
# STICKER DETECTION (ORB features + homography)
# ============================================================

class StickerDetector:

    WORK_W = 800        # template width used for features
    FRAME_MAX = 960     # frames are downscaled to this before matching (ORB)
    SIFT_MAX = 1280     # SIFT works on (up to) this size

    def __init__(self, template_bgr):
        self.tpl = template_bgr
        self.th, self.tw = template_bgr.shape[:2]
        self.scale = self.WORK_W / self.tw

        small = cv2.resize(
            template_bgr, None, fx=self.scale, fy=self.scale,
            interpolation=cv2.INTER_AREA
        )
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        self.orb_t = cv2.ORB_create(nfeatures=3000)
        self.kp_t, self.des_t = self.orb_t.detectAndCompute(g, None)

        self.orb_f = cv2.ORB_create(nfeatures=2500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # SIFT finds the sticker in close-ups, steep angles and glare where
        # ORB fails (real photos: 14/14 found vs 8/14), ~110 ms a frame.
        self.sift = None
        if hasattr(cv2, "SIFT_create"):
            self.sift = cv2.SIFT_create(nfeatures=4000)
            self.kp_s, self.des_s = self.sift.detectAndCompute(
                cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY), None)
            self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))

        # Printed artwork (text, dots, borders) — ignored when looking for ink
        tg = template_bgr.min(axis=2)
        self.printed = cv2.dilate(
            (tg < 150).astype(np.uint8), np.ones((5, 5), np.uint8)
        )

    def detect(self, frame):
        if self.sift is not None:
            return self._detect_sift(frame)
        return self._detect_orb(frame)

    def _detect_sift(self, frame):
        h, w = frame.shape[:2]
        fs = min(1.0, self.SIFT_MAX / max(w, h))
        small = (cv2.resize(frame, None, fx=fs, fy=fs, interpolation=cv2.INTER_AREA)
                 if fs < 1.0 else frame)
        g = self.clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        kp, des = self.sift.detectAndCompute(g, None)
        if des is None or len(kp) < 30:
            return None
        pairs = self.flann.knnMatch(self.des_s, des, k=2)
        # strict matching first; a looser pass only if that finds nothing
        for ratio in (0.75, 0.8):
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
            if len(good) < 20:
                continue
            src = np.float32([self.kp_s[m.queryIdx].pt for m in good])
            dst = np.float32([kp[m.trainIdx].pt for m in good]) / fs
            found = self._fit(src, dst, w, h, min_inliers=15)
            if found is not None:
                return found
        return None

    def _detect_orb(self, frame):
        h, w = frame.shape[:2]
        fs = min(1.0, self.FRAME_MAX / max(w, h))
        small = (
            cv2.resize(frame, None, fx=fs, fy=fs, interpolation=cv2.INTER_AREA)
            if fs < 1.0 else frame
        )
        g = self.clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

        kp, des = self.orb_f.detectAndCompute(g, None)
        if des is None or len(kp) < 30:
            return None

        pairs = self.matcher.knnMatch(self.des_t, des, k=2)
        good = [
            p[0] for p in pairs
            if len(p) == 2 and p[0].distance < 0.75 * p[1].distance
        ]
        if len(good) < 25:
            return None

        src = np.float32([self.kp_t[m.queryIdx].pt for m in good]) / self.scale
        dst = np.float32([kp[m.trainIdx].pt for m in good]) / fs
        return self._fit(src, dst, w, h, min_inliers=20)

    def _fit(self, src, dst, w, h, min_inliers):

        H, inliers = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 5.0)
        if H is None or int(inliers.sum()) < min_inliers:
            return None
        # Refit on every agreeing point: steadier corners from frame to
        # frame (a robust fit alone uses few points and jitters). Keep the
        # robust fit if the refit gives an impossible shape.
        keep = inliers.ravel().astype(bool)
        H2, _ = cv2.findHomography(src[keep], dst[keep], 0)
        box = np.float32([[0, 0], [self.tw, 0], [self.tw, self.th], [0, self.th]]).reshape(-1, 1, 2)
        corners = None
        for cand in ([H2, H] if H2 is not None else [H]):
            c = cv2.perspectiveTransform(box, cand).reshape(-1, 2)
            if cv2.isContourConvex(c.astype(np.int32)) and cv2.contourArea(c) >= 0.03 * w * h:
                H, corners = cand, c
                break
        if corners is None:
            return None

        return {"H": H, "corners": corners, "inliers": int(inliers.sum())}

    def warp(self, frame, H):
        return cv2.warpPerspective(
            frame, np.linalg.inv(H), (self.tw, self.th),
            flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
        )


# ============================================================
# DIGIT RECOGNITION (offline, learns from corrections)
# ============================================================

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
PERSIAN_EXTRA = {"۴": 4, "۵": 5, "۶": 6}     # common handwritten shapes

ENGLISH_FONTS = [
    "Inkfree.ttf", "segoepr.ttf", "segoeprb.ttf", "segoesc.ttf",
    "comic.ttf", "comicbd.ttf", "arial.ttf", "arialbd.ttf", "tahoma.ttf",
    "times.ttf", "calibri.ttf", "georgia.ttf", "consola.ttf",
    "bahnschrift.ttf", "ARIALN.TTF", "Gabriola.ttf", "verdana.ttf",
    "trebuc.ttf",
]
ARABIC_FONTS = [
    "arial.ttf", "arialbd.ttf", "tahoma.ttf", "tahomabd.ttf", "times.ttf",
    "majalla.ttf", "majallab.ttf", "arabtype.ttf", "andlso.ttf",
    "aldhabi.ttf", "segoeui.ttf", "segoeuib.ttf", "trado.ttf",
    "simpo.ttf", "ebrima.ttf",
]


def thin(img):
    """Zhang-Suen thinning of a 0/1 image (skeleton, 1px strokes)."""
    pad = np.pad(img.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = pad
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            B = sum(x.astype(np.int16) for x in ring[:8])
            A = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int16)
                    for i in range(8))
            m = (P[1:-1, 1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1)
            if step == 0:
                m &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                m &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if m.any():
                pad[1:-1, 1:-1][m] = 0
                changed = True
    return pad[1:-1, 1:-1]


# True: strokes are thinned to a skeleton and redrawn at one width (helps
# computer fonts). False: keep the real stroke shape (keeps small loops
# in handwritten 6/8/9 open).
NORM_THIN = True     # set per digit style by DigitReader

# A digit counts as "sure" only when this share of its nearest examples
# agree with the classifier (6 of 7 lets ~2% wrong reads through
# unflagged on real handwriting; 7 of 7 lets ~0.4% through).
CONF_OK = 0.95


def norm_tag():
    return "t" if NORM_THIN else "r"


def normalize_glyph(mask):
    """
    Binary glyph (any size) -> 28x28 uint8, aspect kept, centred.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    g = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    h, w = g.shape
    side = max(h, w)
    sq = np.zeros((side, side), np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = g
    if NORM_THIN:
        big = (cv2.resize(sq * 255, (40, 40), interpolation=cv2.INTER_AREA) > 100)
        skel = thin(big.astype(np.uint8))
        skel = cv2.dilate(skel, np.ones((3, 3), np.uint8))
        small = cv2.resize(skel * 255, (20, 20), interpolation=cv2.INTER_AREA)
    else:
        small = cv2.resize(sq * 255, (20, 20), interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), np.uint8)
    out[4:24, 4:24] = small
    return out


def has_hole(mask):
    cnts, hier = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    return hier is not None and bool((hier[0][:, 3] >= 0).any())


class DigitReader:

    def __init__(self, style="arabic"):
        global NORM_THIN
        self.style = style
        # English handwriting keeps its real strokes (small loops stay
        # open); Arabic, learned from fonts, reads better thinned.
        NORM_THIN = style == "arabic"
        # OpenCV 5 moved HOG and the ML module out of the main package;
        # without them the classic reader is skipped and the CNN reads alone.
        self.classic_ok = hasattr(cv2, "HOGDescriptor") and hasattr(cv2, "ml")
        self.hog = (cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)
                    if self.classic_ok else None)
        self.knn = None
        self.train()

    # ---------------- training data
    def _font_dirs(self):
        return [Path("C:/Windows/Fonts"), ASSETS / "fonts"]

    def _synthetic(self):
        from PIL import Image, ImageDraw, ImageFont

        if self.style == "arabic":
            glyphs = [(c, i) for i, c in enumerate(ARABIC_DIGITS)]
            glyphs += list(PERSIAN_EXTRA.items())
            fonts = ARABIC_FONTS + [
                p.name for p in (ASSETS / "fonts").glob("*.ttf")
            ]
        else:
            glyphs = [(str(i), i) for i in range(10)]
            fonts = ENGLISH_FONTS

        samples = []
        for fname in fonts:
            path = next(
                (d / fname for d in self._font_dirs() if (d / fname).exists()),
                None,
            )
            if path is None:
                continue
            try:
                font = ImageFont.truetype(str(path), 56)
            except Exception:
                continue

            missing = self._render(font, "\uffff")
            for ch, label in glyphs:
                img = self._render(font, ch)
                if img is None or (
                    missing is not None
                    and img.shape == missing.shape
                    and np.array_equal(img, missing)
                ):
                    continue          # font lacks this glyph
                for angle in (-9, 0, 9):
                    for shear in (-0.25, 0.0, 0.25):
                        for thick in (0, 2):
                            g = self._augment(img, angle, thick, shear)
                            n = normalize_glyph(g)
                            if n is not None:
                                samples.append((n, label))
        return samples

    @staticmethod
    def _render(font, ch):
        from PIL import Image, ImageDraw
        im = Image.new("L", (110, 110), 0)
        ImageDraw.Draw(im).text((25, 10), ch, fill=255, font=font)
        a = (np.array(im) > 128).astype(np.uint8)
        return a if a.any() else None

    @staticmethod
    def _augment(img, angle, thick, shear=0.0):
        h, w = img.shape
        S = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
        g = cv2.warpAffine(img, S, (w, h), flags=cv2.INTER_NEAREST)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        g = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_NEAREST)
        k = np.ones((3, 3), np.uint8)
        if thick < 0:
            g = cv2.erode(g, k)
        for _ in range(max(0, thick)):
            g = cv2.dilate(g, k)
        return g

    def _mnist(self, limit=None):
        """Real handwritten digits (MNIST, 60,000 samples by ~250 writers),
        put through the same thinning/normalising as the sticker digits."""
        import gzip
        base = DATA_DIR / "mnist"
        try:
            imgs = np.frombuffer(
                gzip.open(base / "train-images-idx3-ubyte.gz").read(), np.uint8, offset=16
            ).reshape(-1, 28, 28)
            labels = np.frombuffer(
                gzip.open(base / "train-labels-idx1-ubyte.gz").read(), np.uint8, offset=8
            )
        except Exception:
            return []
        if limit:
            imgs, labels = imgs[:limit], labels[:limit]
        out = []
        for img, lab in zip(imgs, labels):
            n = normalize_glyph((img > 100).astype(np.uint8))
            if n is not None:
                out.append((n, int(lab)))
        return out

    def _user_samples(self):
        out = []
        base = SAMPLES_DIR / (self.style + ("" if NORM_THIN else "_raw"))
        if not base.exists():
            return out
        for d in base.iterdir():
            if not d.is_dir() or not d.name.isdigit():
                continue
            for f in d.glob("*.png"):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is not None and img.shape == (28, 28):
                    out.append((img, int(d.name)))
        return out

    @staticmethod
    def _jitter(img28):
        out = []
        for angle in (-8, 0, 8):
            for scale in (0.9, 1.0, 1.1):
                for dx in (-1, 0, 1):
                    M = cv2.getRotationMatrix2D((14, 14), angle, scale)
                    M[0, 2] += dx
                    out.append(cv2.warpAffine(img28, M, (28, 28)))
        return out

    def _feat(self, img28):
        return self.hog.compute(img28).reshape(-1)

    def _cached(self, name, build):
        """Load feature arrays from DATA_DIR/name, or build and save them."""
        cache = DATA_DIR / name
        try:
            z = np.load(cache)
            return z["X"], z["y"]
        except Exception:
            samples = build()
            X = np.float32([self._feat(i) for i, _ in samples]).reshape(-1, 324)
            y = np.float32([l for _, l in samples])
            try:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, X=X, y=y)
            except Exception:
                pass
            return X, y

    def train(self):
        self._load_cnn()
        if not self.classic_ok:
            self.user_count = len(self._user_samples())
            if self.cnn is None:
                raise RuntimeError(
                    "This OpenCV has no HOG/ML module and no CNN model was found. "
                    "Install OpenCV 4: pip install \"opencv-python<5\"")
            self.mode = "cnn"
            print("Note: OpenCV without HOG/ML — reading digits with the CNN only.", flush=True)
            return
        X, y = self._cached(f"digit_synth_{self.style}_v2{'' if NORM_THIN else '_raw'}.npz", self._synthetic)

        # Real handwriting for English digits (MNIST). The KNN gets all of
        # it; the SVM gets a slice so training stays quick.
        Xm = ym = None
        if self.style == "english":
            Xm, ym = self._cached(f"digit_mnist_v1{'' if NORM_THIN else '_raw'}.npz", self._mnist)
            if not len(Xm):
                Xm = ym = None

        # Camera-like examples made by train_clothes.py (photographed,
        # dark, blurred handwriting cut out the same way the app does it)
        Xp = yp = None
        pipe = DATA_DIR / "digit_pipe_english_raw.npz"
        if self.style == "english" and not NORM_THIN and pipe.exists():
            try:
                z = np.load(pipe)
                Xp, yp = z["X"], z["y"]
            except Exception:
                Xp = yp = None
        self.pipe_count = 0 if yp is None else len(yp)

        user = self._user_samples()
        self.user_count = len(user)
        Xu, yu = [], []
        # Your own handwriting outweighs everything else: every stored
        # digit is added in several slightly shifted/turned copies.
        for img, label in user:
            for v in self._jitter(img):
                Xu.append(self._feat(v))
                yu.append(label)
        Xu = np.float32(Xu).reshape(-1, X.shape[1])
        yu = np.float32(yu)

        extra_X = ([Xm] if Xm is not None else []) + ([Xp] if Xp is not None else [])
        extra_y = ([ym] if ym is not None else []) + ([yp] if yp is not None else [])
        X_knn = np.vstack([X] + extra_X + [Xu])
        y_knn = np.concatenate([y] + extra_y + [yu])
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(X_knn, cv2.ml.ROW_SAMPLE, y_knn.reshape(-1, 1))

        X_svm = np.vstack([X] + ([Xm[:12000]] if Xm is not None else [])
                          + ([Xp[-15000:]] if Xp is not None else []) + [Xu])
        y_svm = np.concatenate([y] + ([ym[:12000]] if ym is not None else [])
                               + ([yp[-15000:]] if yp is not None else []) + [yu])

        # The SVM is the slow part: reuse a saved model while the set of
        # learned samples hasn't changed.
        import hashlib
        key = hashlib.md5(
            f"{self.style}|{norm_tag()}|{len(X_svm)}|{float(X_svm.sum()):.3f}".encode()
        ).hexdigest()[:12]
        model = DATA_DIR / f"digit_svm_{self.style}_{key}.xml"
        self.svm = None
        if model.exists():
            try:
                self.svm = cv2.ml.SVM_load(str(model))
            except Exception:
                self.svm = None
        if self.svm is None:
            self.svm = cv2.ml.SVM_create()
            self.svm.setType(cv2.ml.SVM_C_SVC)
            self.svm.setKernel(cv2.ml.SVM_RBF)
            self.svm.setC(12.5)
            self.svm.setGamma(0.5)
            self.svm.train(X_svm, cv2.ml.ROW_SAMPLE, y_svm.astype(np.int32))
            try:
                for old in DATA_DIR.glob(f"digit_svm_{self.style}_*.xml"):
                    old.unlink()
                self.svm.save(str(model))
            except Exception:
                pass

    # ---------------- reading
    def _pieces(self, ink):
        """Ink mask of one field -> glyph masks (sub, x, y, h), left to right."""
        if SEG_CLOSE:
            ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        H, W = ink.shape
        biggest = max([st[i, 4] for i in range(1, n)] or [0])
        comps = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < 6:
                continue
            # Things stuck to the edge of the box are the printed frame,
            # a shadow, a fold or the postmark stamp — not handwriting.
            near_lr = x < 0.06 * W or x + w > 0.94 * W
            near_tb = y <= 1 or y + h >= H - 1
            if near_lr and w < 0.3 * h and h > 0.4 * H:
                continue                      # thin vertical line at the side
            if near_lr and w < 0.2 * W and h > 0.85 * H:
                continue                      # a band down the side of the box
            if near_tb and h < 0.15 * H:
                continue                      # a band along the top/bottom
            if (near_lr or near_tb) and a < 0.2 * biggest:
                continue                      # a bit of stamp/frame at the edge
            if a < 0.004 * H * W:
                continue                      # tiny specks (perforation, dust)
            # a tiny speck next to real strokes is a pen dot / dirt
            # (Arabic zero is a small dot, so only for English)
            if self.style == "english" and a < 0.08 * biggest:
                continue
            comps.append([x, y, w, h, [i]])
        if not comps:
            return []
        comps.sort(key=lambda c: c[0])

        area = lambda c: c[2] * c[3]

        # Merge broken pieces of one digit (a small fragment overlapping a
        # bigger stroke) — but keep two full, slanted digits apart.
        merged = [comps[0]]
        for c in comps[1:]:
            m = merged[-1]
            overlap = (m[0] + m[2]) - c[0]
            small_part = min(area(m), area(c)) < 0.35 * max(area(m), area(c))
            # Digits of a number sit side by side, never on top of each
            # other: two pieces stacked vertically are one broken digit.
            v_overlap = min(m[1] + m[3], c[1] + c[3]) - max(m[1], c[1])
            stacked = v_overlap < 0.3 * min(m[3], c[3])
            if overlap > 0.5 * min(m[2], c[2]) and (small_part or stacked):
                x0, y0 = min(m[0], c[0]), min(m[1], c[1])
                x1 = max(m[0] + m[2], c[0] + c[2])
                y1 = max(m[1] + m[3], c[1] + c[3])
                merged[-1] = [x0, y0, x1 - x0, y1 - y0, m[4] + c[4]]
            else:
                merged.append(c)

        pieces = []
        for x, y, w, h, ids in merged:
            sub = np.isin(lab[y:y + h, x:x + w], ids).astype(np.uint8)
            pieces.extend(self._split_touching(sub, x, y))
        return pieces

    def segment(self, ink):
        """Ink mask of one field -> (28x28 glyphs, sizes), left to right."""
        return self._glyphs(self._pieces(ink))

    @staticmethod
    def _glyphs(pieces):
        if not pieces:
            return [], []
        tallest = max(p[3] for p in pieces)
        glyphs, sizes = [], []
        for sub, x, y, h in pieces:
            g = normalize_glyph(sub)
            if g is not None:
                glyphs.append(g)
                sizes.append((sub.shape[1], h, tallest, has_hole(sub), float(sub.mean())))
        return glyphs, sizes

    @staticmethod
    def _split_touching(sub, x, y, force=False):
        """A blob much wider than tall is two digits touching: cut it at
        the thinnest column near the middle."""
        h, w = sub.shape
        if w < 12 or (not force and w <= SPLIT_RATIO * h):
            return [(sub, x, y, h)]
        cols = sub.sum(axis=0)
        lo, hi = int(w * 0.3), int(w * 0.7)
        cut = lo + int(np.argmin(cols[lo:hi]))
        out = []
        for part, ox in ((sub[:, :cut], x), (sub[:, cut:], x + cut)):
            ys = np.nonzero(part.any(axis=1))[0]
            if len(ys) and part.sum() >= 6:
                part = part[ys.min():ys.max() + 1]
                out.append((part, ox, y + ys.min(), part.shape[0]))
        return out or [(sub, x, y, h)]

    def _load_cnn(self):
        """The CNN trained by train_cnn.py, if there is one."""
        self.cnn = None
        self.mode = load_config().get("reader", "ensemble")
        model = DATA_DIR / f"cnn_{self.style}.npz"
        if not model.exists():
            # the trained model shipped with the app (for a new laptop)
            model = ASSETS / f"cnn_{self.style}.npz"
        if self.mode in ("cnn", "ensemble") and model.exists():
            try:
                from cnn import CNN
                self.cnn = CNN.load(model)
            except Exception:
                self.cnn = None

    def _classify(self, subs):
        """
        Raw (digit, confidence) for each glyph mask.
          classic  — SVM picks the digit, KNN neighbours give the confidence
          cnn      — the CNN alone
          ensemble — both; sure only when they agree, ⚠ when they differ
                     (tested on 40 real photos: as good or better than either
                     alone in colour, grey-scale and black-and-white)
        """
        glyphs = [normalize_glyph(s) for s in subs]
        cnn = getattr(self, "cnn", None)
        if cnn is None:
            return self._classify_classic(glyphs)
        x = (np.float32(glyphs) / 255.0)[:, None, :, :]
        prob = cnn.predict_proba(x)
        net = [(int(p.argmax()), float(p.max())) for p in prob]
        if getattr(self, "mode", "ensemble") == "cnn":
            return net
        out = []
        for (d1, c1), (d2, c2) in zip(self._classify_classic(glyphs), net):
            if d1 == d2:
                out.append((d1, max(c1, c2)))
            else:
                out.append((d1 if c1 >= c2 else d2, 0.3))
        return out

    def _digit_probs(self, subs):
        """Probability of each digit 0-9 for each glyph (CNN and KNN votes
        averaged when both exist). Used to find the next-best readings."""
        glyphs = [normalize_glyph(s) for s in subs]
        parts = []
        if getattr(self, "cnn", None) is not None:
            parts.append(self.cnn.predict_proba((np.float32(glyphs) / 255.0)[:, None, :, :]))
        if self.classic_ok and self.knn is not None:
            feats = np.float32([self._feat(g) for g in glyphs])
            _, _, neigh, _ = self.knn.findNearest(feats, k=7)
            votes = np.zeros((len(glyphs), 10), np.float32)
            for i, row in enumerate(neigh):
                for d in row:
                    votes[i, int(d)] += 1.0 / len(row)
            parts.append(votes)
        if not parts:
            return np.full((len(glyphs), 10), 0.1, np.float32)
        return np.mean(parts, axis=0) + 1e-4

    def _alternatives(self, subs, best, top=8):
        """Most likely readings of a field, best first: [(value, score)]."""
        probs = self._digit_probs(subs)
        beams = [("", 1.0)]
        for p in probs:
            cand = np.argsort(p)[::-1][:3]
            beams = sorted(((b + str(d), sc * float(p[d])) for b, sc in beams for d in cand),
                           key=lambda t: -t[1])[:top]
        alts = [(int(b), sc) for b, sc in beams]
        if best is not None and all(v != best for v, _ in alts):
            alts.insert(0, (best, max(sc for _, sc in alts)))
        return alts

    def _classify_classic(self, glyphs):
        feats = np.float32([self._feat(g) for g in glyphs])
        _, _, neigh, _ = self.knn.findNearest(feats, k=7)
        pred = self.svm.predict(feats)[1].ravel()
        return [
            (int(d), float((neigh[i] == d).sum()) / neigh.shape[1])
            for i, d in enumerate(pred)
        ]

    def read(self, ink, cell_h):
        """-> (value or None, confidence 0..1, glyphs)"""
        pieces = self._pieces(ink)
        if not pieces:
            return 0, 1.0, []          # empty field = 0

        # Two digits that touch look like one wide shape. Try reading it
        # both whole and cut in two, and keep whichever reads cleanly.
        refined = []
        for p in pieces:
            sub = p[0]
            h, w = sub.shape
            # Handwriting is often wide (a big 2 or 0), so a shape is only
            # cut when it is really wide AND does not read as one digit.
            if w > SPLIT_TRY * h and w >= 12:
                parts = self._split_touching(sub, p[1], p[2], force=True)
                if len(parts) == 2:
                    _, whole_c = self._classify([sub])[0]
                    split = self._classify([q[0] for q in parts])
                    if whole_c < SPLIT_WHOLE_MAX and min(c for _, c in split) >= CONF_OK:
                        refined.extend(parts)
                        continue
            refined.append(p)
        pieces = refined

        glyphs, sizes = self._glyphs(pieces)
        if len(glyphs) > 3:
            return None, 0.0, glyphs   # scribble / noise

        digits, confs = [], []
        raw = self._classify([p[0] for p in pieces])

        for i, (w, h, tallest, hole, fill) in enumerate(sizes):
            d, c = raw[i]
            # Shapes that don't look like one clean digit are never trusted:
            # they are flagged ⚠ for a person to check instead.
            aspect = w / max(1, h)
            # How wide each digit can honestly be written (people write
            # 0 and 2 wide and round; 1, 4, 7, 9 never that wide).
            max_aspect = {0: 1.6, 1: 0.6, 2: 1.6, 3: 1.4, 4: 1.3, 5: 1.4,
                          6: 1.25, 7: 1.3, 8: 1.25, 9: 1.25}.get(d, 1.1)
            if aspect > max_aspect:                  # likely two digits merged
                c = min(c, 0.4)
            if fill > 0.62 and aspect > 0.4 and not hole:   # a filled-in loop
                c = min(c, 0.4)
            short = 0.45 if d == 0 else 0.55        # 0 is often written small
            if len(sizes) > 1 and h < short * tallest and not (
                    self.style == "arabic" and not hole and aspect < 2.5):
                c = min(c, 0.4)                      # a broken fragment
            # Arabic zero is a small solid dot (٥ is a loop with a hole).
            # "Small" means next to the other digits in the same box;
            # a lone digit only counts as a dot if it is tiny for the box.
            if self.style == "arabic":
                if len(sizes) > 1:
                    is_dot = h < 0.45 * tallest
                else:
                    is_dot = h < 0.25 * cell_h
                if is_dot and 0.4 < w / max(1, h) < 2.5 and not hole:
                    d, c = 0, max(c, 0.85)
                elif d == 0:
                    c *= 0.5
            digits.append(d)
            confs.append(c)

        value = int("".join(map(str, digits)))
        try:
            self.last_alts = self._alternatives([p[0] for p in pieces], value)
        except Exception:
            self.last_alts = [(value, 1.0)]
        return value, min(confs), glyphs

    def learn(self, glyphs, value):
        """Store glyphs as samples when their count matches the digits."""
        s = str(int(value))
        if not glyphs or len(glyphs) != len(s):
            return 0
        stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        for i, (g, ch) in enumerate(zip(glyphs, s)):
            d = SAMPLES_DIR / (self.style + ("" if NORM_THIN else "_raw")) / ch
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{stamp}_{i}.png"), g)
        return len(s)


# ============================================================
# READING A STRAIGHTENED STICKER
# ============================================================

# Reading settings (tuned on real handwriting photographed in normal
# light and in the dark)
INK_SCALE = 2        # fields are enlarged before the ink is cut out
INK_RATIO = 0.66     # ink = darker than this share of the local paper
                     # ("auto" = chosen per field from its own contrast)
DARK_LEVEL = 90      # paper darker than this = a dark photo
PEN_SAT = 60         # colour saturation above this = coloured pen (red/blue)
GRAY_INK = 0.35      # grey pixels count as ink only if this dark (black pen);
                     # lighter grey is print shading, a fold or the sticker edge
GRAY_MODE = 0.66     # no coloured ink in the field (black pen, grey-scale or
                     # black-and-white photo): tuned on real photos turned grey
INK_MEDIAN = True    # smooth camera noise before cutting the ink out
SEG_CLOSE = False    # closing joins broken strokes but fills small loops
SPLIT_RATIO = 1.5    # a blob this much wider than tall is two digits
SPLIT_TRY = 1.25     # only shapes at least this wide may be tried as 2 digits
SPLIT_WHOLE_MAX = 0.6  # ...and only when the whole shape reads badly


def field_ink(canon, printed, rect, inset=3, scale=None):
    """
    Pen ink inside one field. Works in the dark and under shadows: each
    pixel is compared with the local paper brightness around it, after
    the camera noise is smoothed out. The field is enlarged first so the
    loops in 6, 8, 9 and 0 stay open.
    """
    s = INK_SCALE if scale is None else scale
    x, y, w, h = rect
    sat = None
    if canon.ndim == 3:
        # colour photo: also know which pixels are coloured pen ink
        bgr = canon[y + inset:y + h - inset, x + inset:x + w - inset]
        roi = bgr.min(axis=2)
        sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1]
    else:
        roi = canon[y + inset:y + h - inset, x + inset:x + w - inset]
    prn = printed[y + inset:y + h - inset, x + inset:x + w - inset]
    dark = float(np.percentile(roi, 90)) < DARK_LEVEL
    if INK_MEDIAN:
        # Dark photos are much noisier: smooth harder
        roi = cv2.medianBlur(roi, 5 if dark else 3)
    if s != 1:
        roi = cv2.resize(roi, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        prn = cv2.resize(prn, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        if sat is not None:
            sat = cv2.resize(sat, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    roi = roi.astype(np.float32)
    # Local paper level: brightest value in a neighbourhood wider than a stroke
    k = int(15 * s) | 1
    paper = cv2.dilate(roi, np.ones((k, k), np.uint8))
    paper = cv2.GaussianBlur(paper, (0, 0), 3 * s)
    diff = paper - roi
    ratio = roi / np.maximum(paper, 1.0)
    # Very dark photos: require a real step in brightness, not just noise
    min_step = max(10.0, 0.18 * float(np.percentile(paper, 50)))
    if INK_RATIO == "auto":
        # Otsu on this field's own brightness ratios: the split between
        # paper and pen, kept within sensible limits
        r8 = np.clip(ratio * 255, 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(r8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        limit = float(np.clip(t / 255.0, 0.5, 0.72))
        # an empty field has no real split: fall back to the fixed level
        if float((ratio < limit).mean()) > 0.35:
            limit = 0.6
    else:
        limit = INK_RATIO
        # Washed-out / over-exposed photos: the pen is only a little
        # darker than the paper. Put the cut-off between this field's
        # darkest pen and its paper instead of a fixed level.
        r_ink = float(np.percentile(ratio, 1.5))
        if r_ink < 0.9:
            limit = min(0.85, max(INK_RATIO, r_ink + 0.55 * (1.0 - r_ink)))
    ink = (ratio < limit) & (diff > min_step)
    coloured = False
    if sat is not None and ink.any():
        # colour of the pen compared with the colour of this paper
        sat_paper = float(np.median(sat))
        sat_pen = float(np.percentile(sat[ink], 90))
        pen_sat = max(sat_paper + 15, sat_paper + 0.4 * (sat_pen - sat_paper))
        coloured = (sat_pen - sat_paper > 20
                    and int((ink & (sat >= pen_sat)).sum()) > 30 * s * s)
    if coloured:
        # Coloured pen in a colour photo: grey that isn't very dark is
        # print shading, a fold or the sticker edge, not pen.
        ink &= (sat >= pen_sat) | (ratio < GRAY_INK)
    else:
        # Black pen, or a grey-scale / black-and-white photo: there is no
        # colour to go by, so only clearly dark strokes count.
        ink &= ratio < max(GRAY_MODE, limit)
    ink = ink.astype(np.uint8)
    ink[prn > 0] = 0
    # Drop isolated noise specks
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for i in range(1, n):
        if st[i, 4] < 4 * s * s:
            ink[lab == i] = 0
    return ink


def read_sticker(canon_bgr, detector, reader):
    cmin = canon_bgr          # colour image: field_ink uses the colour too
    out = {"image": canon_bgr}

    # Ticked total box
    fills = []
    for value, rect in CHECKBOXES:
        ink = field_ink(cmin, detector.printed, rect, inset=5)
        fills.append((ink.mean(), value))
    # A box counts as ticked when it has clear ink AND clearly more than
    # any other box (a long tick line can stray into the next box).
    fills.sort(key=lambda fv: fv[0], reverse=True)
    (f1, best), (f2, _) = fills[0], fills[1]
    if f1 > 0.07 and (f2 < 0.07 or f1 >= 2.5 * f2):
        out["total_choice"], out["total_conf"] = best, 1.0
    elif f1 > 0.07:
        # several boxes inked (two ticks, or numbers written in them)
        out["total_choice"], out["total_conf"] = best, 0.3
    else:
        out["total_choice"], out["total_conf"] = None, 0.5

    # Handwritten "other" total
    ink = field_ink(cmin, detector.printed, OTHER_FIELD)
    v, c, g = reader.read(ink, OTHER_FIELD[3] * INK_SCALE)
    out["other"] = {"value": v, "conf": c, "glyphs": g,
                    "crop": crop(canon_bgr, OTHER_FIELD)}

    # Categories
    out["cats"] = {}
    for key, _, rect in CATEGORIES:
        ink = field_ink(cmin, detector.printed, rect)
        reader.last_alts = []
        v, c, g = reader.read(ink, rect[3] * INK_SCALE)
        out["cats"][key] = {"value": v, "conf": c, "glyphs": g,
                            "alts": list(reader.last_alts) if g else [],
                            "crop": crop(canon_bgr, rect)}

    out["location_crop"] = crop(canon_bgr, LOCATION_FIELD)

    # Handwriting fingerprint, used to refuse scanning the same box twice
    out["signature"], out["ink_amount"] = ink_signature(cmin, detector.printed)
    return out


def ink_signature(cmin, printed):
    """
    Blurred, low-resolution picture of the handwriting in the number
    fields. The same sticker photographed twice gives nearly the same
    picture even with a different angle or light; two different boxes
    (even with the same numbers) do not.
    """
    parts, amount = [], 0
    for rect in [r for _, _, r in CATEGORIES] + [OTHER_FIELD]:
        ink = field_ink(cmin, printed, rect)
        amount += int(ink.sum())
        g = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 2.5)
        parts.append(cv2.resize(g, (40, 16), interpolation=cv2.INTER_AREA).ravel())
    return np.concatenate(parts).astype(np.float32), amount


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum()) / d if d > 0 else 0.0


def signature_similarity(a, b):
    """
    How alike two fingerprints are, -1..1 (1 = identical). Each number
    field is compared at small shifts (a photo is never lined up to the
    pixel) and the best match counts; fields are weighted by their ink.
    """
    fa, fb = a.reshape(-1, 16, 40), b.reshape(-1, 16, 40)
    if len(fa) != len(fb):
        return _corr(a.ravel(), b.ravel())
    total = weight = 0.0
    for x, y in zip(fa, fb):
        w = float(x.sum() + y.sum())
        if w <= 0:
            continue                       # both empty: says nothing
        best = -1.0
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-3, -2, -1, 0, 1, 2, 3):
                ys = slice(max(0, dy), 16 + min(0, dy))
                yt = slice(max(0, -dy), 16 + min(0, -dy))
                xs = slice(max(0, dx), 40 + min(0, dx))
                xt = slice(max(0, -dx), 40 + min(0, -dx))
                best = max(best, _corr(x[ys, xs], y[yt, xt]))
        total += best * w
        weight += w
    return total / weight if weight > 0 else 0.0


def learn_from_boxes(reader, detector, rows, images_dir=None):
    """
    Teach the reader the handwriting on saved boxes. Only boxes whose
    numbers were confirmed (no ⚠ left on them) are used, and only fields
    where the photo cuts into exactly as many digits as the saved number.
    Returns how many digits were learned.
    """
    images_dir = images_dir or IMAGES_DIR
    learned = 0
    for r in rows:
        if r.get("check") == "1" or not r.get("image"):
            continue
        canon = cv2.imread(str(images_dir / r["image"]))
        if canon is None:
            continue
        cmin = canon
        for key, _, rect in CATEGORIES:
            v = str(r.get(key, "")).strip()
            if not v.isdigit():
                continue
            pieces = reader._pieces(field_ink(cmin, detector.printed, rect))
            if len(pieces) != len(str(int(v))):
                continue
            learned += reader.learn([normalize_glyph(p[0]) for p in pieces], int(v))
    return learned


def crop(img, rect, pad=4):
    x, y, w, h = rect
    return img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad].copy()


def to_qimage(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# ============================================================
# CAMERA + AUTO-CAPTURE THREAD
# ============================================================

class CameraWorker(QThread):

    frame_ready = Signal(QImage)
    status = Signal(str, str)          # text, kind
    captured = Signal(object, object)  # canon image, detection

    SEARCH, HOLD, REVIEW, CLEAR = range(4)

    def __init__(self, source, detector):
        super().__init__()
        self.source = source
        self.detector = detector
        self.running = True
        self.paused = False
        self.state = self.SEARCH
        self.history = []
        self.misses = 0
        self.last_center = None

    def set_source(self, source):
        self.source = source
        self._reopen = True

    def rearm(self):
        # After a save/discard: wait until this box leaves the view
        self.misses = 0
        self.state = self.CLEAR

    def stop(self):
        self.running = False

    def _open(self):
        src = str(self.source).strip()
        if src.isdigit():
            cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap if cap.isOpened() else None

    def run(self):
        cap = None
        self._reopen = False
        n = 0
        last_det = None
        while self.running:
            if cap is None or self._reopen:
                if cap is not None:
                    cap.release()
                self._reopen = False
                self.status.emit("بيوصل بالكاميرا…", "wait")
                cap = self._open()
                if cap is None:
                    self.status.emit(
                        "الكاميرا مش متوصلة — راجع العنوان في الإعدادات", "error"
                    )
                    time.sleep(2.0)
                    continue
                self.state = self.SEARCH

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                self.status.emit("الاتصال بالكاميرا وقع — بيحاول تاني…", "error")
                time.sleep(1.0)
                continue

            if self.paused:
                self.frame_ready.emit(self._preview(frame, None))
                time.sleep(0.03)
                continue

            n += 1
            every = 4 if self.state == self.REVIEW else 2
            if n % every == 0:
                last_det = self.detector.detect(frame)
                self._step(frame, last_det)

            self.frame_ready.emit(self._preview(frame, last_det))

        if cap is not None:
            cap.release()

    def _step(self, frame, det):
        diag = float(np.hypot(*frame.shape[:2]))

        if self.state == self.SEARCH:
            if det:
                self.history = [(frame, det)]
                self.misses = 0
                self.state = self.HOLD
                self.status.emit("ثبّت الكرتونة…", "hold")
            else:
                self.status.emit("وجّه الكاميرا على الاستيكر", "search")

        elif self.state == self.HOLD:
            if det is None:
                self.misses += 1
                if self.misses > 3:
                    self.state = self.SEARCH
                return
            self.misses = 0
            self.history = (self.history + [(frame, det)])[-5:]
            if len(self.history) < 4:
                return
            ref = self.history[-1][1]["corners"]
            moved = max(
                float(np.abs(h[1]["corners"] - ref).max()) for h in self.history
            )
            if moved > 0.012 * diag:
                self.status.emit("ثبّت الكرتونة…", "hold")
                return
            # Steady: keep the sharpest frame
            best = max(
                self.history,
                key=lambda fd: cv2.Laplacian(
                    cv2.cvtColor(fd[0], cv2.COLOR_BGR2GRAY), cv2.CV_64F
                ).var(),
            )
            canon = self.detector.warp(best[0], best[1]["H"])
            self.last_center = best[1]["corners"].mean(axis=0)
            self.state = self.REVIEW
            self.status.emit("اتصوّرت ✔ — راجع الأرقام", "captured")
            self.captured.emit(canon, best[1])

        elif self.state == self.CLEAR:
            if det is None:
                self.misses += 1
                if self.misses >= 5:
                    self.state = self.SEARCH
            else:
                self.misses = 0
                c = det["corners"].mean(axis=0)
                # A different sticker came into view without a gap
                if self.last_center is not None and \
                        np.hypot(*(c - self.last_center)) > 0.25 * diag:
                    self.state = self.SEARCH
                else:
                    self.status.emit("شيل الكرتونة وهات اللي بعدها", "clear")

    def _preview(self, frame, det):
        h, w = frame.shape[:2]
        s = min(1.0, 960 / max(w, h))
        img = cv2.resize(frame, None, fx=s, fy=s) if s < 1 else frame.copy()
        if det is not None and self.state != self.SEARCH:
            pts = (det["corners"] * s).astype(np.int32)
            color = {
                self.HOLD: (0, 170, 255),
                self.REVIEW: (60, 170, 60),
                self.CLEAR: (160, 160, 160),
            }.get(self.state, (0, 170, 255))
            cv2.polylines(img, [pts], True, color, 4, cv2.LINE_AA)
        return to_qimage(img)


# ============================================================
# WIDGETS
# ============================================================

def qfont(px, weight=QFont.Bold):
    f = QFont()
    f.setFamilies(["Cairo", "Tajawal", "Segoe UI", "Tahoma"])
    f.setPixelSize(int(px))
    f.setWeight(weight)
    return f


class AirmailFrame(QWidget):
    """Kraft background with the red/blue airmail border of the sticker."""

    BAND = 16

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h, b = self.width(), self.height(), self.BAND
        p.fillRect(self.rect(), QColor("white"))

        colors = [QColor(AIR_RED), QColor("white"), QColor(AIR_BLUE), QColor("white")]
        step = 22
        p.setPen(Qt.NoPen)
        i = 0
        for x in range(-h, w + h, step):
            p.setBrush(colors[i % 4])
            poly = QPolygonF([
                QPointF(x, 0), QPointF(x + step, 0),
                QPointF(x + step + h, h), QPointF(x + h, h),
            ])
            p.drawPolygon(poly)
            i += 1

        p.fillRect(QRectF(b, b, w - 2 * b, h - 2 * b), QColor(KRAFT))
        p.setPen(QPen(QColor(INK_SOFT), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(b + 6, b + 6, w - 2 * b - 12, h - 2 * b - 12))


class StatTile(QFrame):

    def __init__(self, title, color=INK):
        super().__init__()
        self.setObjectName("tile")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        t = QLabel(title)
        t.setObjectName("tileTitle")
        t.setAlignment(Qt.AlignCenter)
        self.value = QLabel("0")
        self.value.setObjectName("tileValue")
        self.value.setAlignment(Qt.AlignCenter)
        self.value.setStyleSheet(f"color: {color};")
        lay.addWidget(t)
        lay.addWidget(self.value)


def pixmap_from(bgr, max_w, max_h):
    pix = QPixmap.fromImage(to_qimage(bgr))
    return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ============================================================
# REVIEW PANEL
# ============================================================

class ReviewPanel(QFrame):

    saved = Signal(dict)
    discarded = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        self.reading = None
        self.read_values = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)

        title = QLabel("راجع الكرتونة")
        title.setObjectName("panelTitle")
        lay.addWidget(title)

        self.sticker = QLabel()
        self.sticker.setAlignment(Qt.AlignCenter)
        self.sticker.setMinimumHeight(150)
        lay.addWidget(self.sticker)

        # Total
        trow = QHBoxLayout()
        tl = QLabel("العدد الإجمالي")
        tl.setObjectName("fieldLabel")
        self.total_combo = QComboBox()
        for v in (30, 40, 50):
            self.total_combo.addItem(str(v), v)
        self.total_combo.addItem("عدد تاني", "other")
        self.total_combo.addItem("مش متعلّم (مجموع الفئات)", None)
        self.other_crop = QLabel()
        self.other_spin = QSpinBox()
        self.other_spin.setRange(0, 9999)
        trow.addWidget(tl)
        trow.addWidget(self.total_combo, 1)
        trow.addWidget(self.other_crop)
        trow.addWidget(self.other_spin)
        lay.addLayout(trow)

        # Categories
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        self.cat_spins, self.cat_crops = {}, {}
        for i, (key, label, _) in enumerate(CATEGORIES):
            l = QLabel(label)
            l.setObjectName("fieldLabel")
            cimg = QLabel()
            cimg.setFixedSize(140, 56)
            cimg.setAlignment(Qt.AlignCenter)
            sp = QSpinBox()
            sp.setRange(0, 9999)
            sp.setMinimumWidth(90)
            sp.valueChanged.connect(self.update_check)
            r, c = divmod(i, 2)
            grid.addWidget(l, r * 2, c * 2, 1, 2)
            grid.addWidget(cimg, r * 2 + 1, c * 2)
            grid.addWidget(sp, r * 2 + 1, c * 2 + 1)
            self.cat_spins[key], self.cat_crops[key] = sp, cimg
        lay.addLayout(grid)

        # Location
        lrow = QHBoxLayout()
        ll = QLabel("المكان")
        ll.setObjectName("fieldLabel")
        self.loc_crop = QLabel()
        self.loc_crop.setFixedSize(200, 46)
        self.loc_crop.setAlignment(Qt.AlignCenter)
        self.loc_combo = QComboBox()
        self.loc_combo.setEditable(True)
        self.loc_combo.setInsertPolicy(QComboBox.NoInsert)
        self.loc_combo.lineEdit().setPlaceholderText("اكتب أو اختار المكان")
        lrow.addWidget(ll)
        lrow.addWidget(self.loc_crop)
        lrow.addWidget(self.loc_combo, 1)
        lay.addLayout(lrow)

        self.check = QLabel()
        self.check.setObjectName("checkLabel")
        self.check.setWordWrap(True)
        lay.addWidget(self.check)

        brow = QHBoxLayout()
        self.save_btn = QPushButton("حفظ  (Enter)")
        self.save_btn.setObjectName("saveButton")
        self.discard_btn = QPushButton("تجاهل  (Esc)")
        self.discard_btn.setObjectName("plainButton")
        self.save_btn.clicked.connect(self.on_save)
        self.discard_btn.clicked.connect(self.discarded.emit)
        brow.addWidget(self.save_btn, 2)
        brow.addWidget(self.discard_btn, 1)
        lay.addLayout(brow)

        self.total_combo.currentIndexChanged.connect(self.update_check)
        self.other_spin.valueChanged.connect(self.update_check)

    # ---------------- fill
    def show_reading(self, reading, locations):
        self.reading = reading
        self.read_values = {}

        if reading is None:            # manual box, no photo
            self.sticker.setText("إدخال يدوي")
            self.total_combo.setCurrentIndex(0)
            self.other_spin.setValue(0)
            self.other_crop.clear()
            for key in self.cat_spins:
                self.cat_spins[key].setValue(0)
                self.cat_crops[key].clear()
                self._flag(self.cat_spins[key], 1.0)
            self.loc_crop.clear()
        else:
            self.sticker.setPixmap(pixmap_from(reading["image"], 420, 170))

            choice = reading["total_choice"]
            idx = self.total_combo.findData(choice)
            self.total_combo.setCurrentIndex(
                idx if idx >= 0 else self.total_combo.count() - 1
            )
            self._flag(self.total_combo, reading["total_conf"])

            o = reading["other"]
            self.other_spin.setValue(o["value"] or 0)
            self.other_crop.setPixmap(pixmap_from(o["crop"], 90, 46))
            self.read_values["other"] = o["value"]

            for key, _, _ in CATEGORIES:
                c = reading["cats"][key]
                self.cat_spins[key].setValue(c["value"] or 0)
                self.cat_crops[key].setPixmap(pixmap_from(c["crop"], 140, 56))
                self._flag(self.cat_spins[key], c["conf"] if c["value"] is not None else 0)
                self.read_values[key] = c["value"]

            self.loc_crop.setPixmap(pixmap_from(reading["location_crop"], 200, 46))

        current = self.loc_combo.currentText()
        self.loc_combo.clear()
        self.loc_combo.addItems(locations)
        self.loc_combo.setEditText(current if current in locations else
                                   (locations[0] if locations else ""))
        self.update_check()
        self.save_btn.setFocus()

    def _flag(self, w, conf):
        color = OK_GREEN if conf >= 0.8 else (WARN if conf >= 0.5 else AIR_RED)
        w.setStyleSheet(f"border: 2px solid {color}; border-radius: 6px;")

    def total_value(self):
        choice = self.total_combo.currentData()
        cats = sum(sp.value() for sp in self.cat_spins.values())
        if choice == "other":
            return self.other_spin.value()
        if choice is None:
            return cats
        return int(choice)

    def update_check(self):
        self.other_spin.setEnabled(self.total_combo.currentData() == "other")
        total = self.total_value()
        cats = sum(sp.value() for sp in self.cat_spins.values())
        if cats and cats != total:
            self.check.setText(
                f"⚠ مجموع الفئات {cats} مش زي الإجمالي {total} — "
                f"الفرق {total - cats}"
            )
            self.check.setStyleSheet(f"color: {AIR_RED};")
        else:
            self.check.setText(f"الإجمالي: {total} قطعة")
            self.check.setStyleSheet(f"color: {OK_GREEN};")

    def confident(self):
        if self.reading is None:
            return False
        r = self.reading
        if r["total_choice"] is None or r["total_conf"] < 1.0:
            return False
        if r["total_choice"] == "other" and r["other"]["conf"] < 0.8:
            return False
        for c in r["cats"].values():
            if c["value"] is None or c["conf"] < 0.8:
                return False
        cats = sum(c["value"] for c in r["cats"].values())
        return cats == 0 or cats == self.total_value()

    def on_save(self):
        values = {
            "total": self.total_value(),
            "location": self.loc_combo.currentText().strip(),
        }
        for key, sp in self.cat_spins.items():
            values[key] = sp.value()
        values["_other"] = self.other_spin.value()
        values["_other_used"] = self.total_combo.currentData() == "other"
        if values["total"] <= 0:
            QMessageBox.warning(self, "حفظ", "الإجمالي لازم يكون أكبر من صفر.")
            return
        self.saved.emit(values)


# ============================================================
# SETTINGS DIALOG
# ============================================================

class SettingsDialog(QDialog):

    def __init__(self, parent, cfg):
        super().__init__(parent)
        self.setWindowTitle("الإعدادات")
        self.setLayoutDirection(Qt.RightToLeft)
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)

        lay.addWidget(self._label("الكاميرا"))
        self.cam = QLineEdit(cfg["camera"])
        self.cam.setLayoutDirection(Qt.LeftToRight)
        lay.addWidget(self.cam)
        help_ = QLabel(
            "موبايل (IP Webcam):  http://192.168.1.5:8080/video\n"
            "موبايل (DroidCam):  http://192.168.1.5:4747/video\n"
            "كاميرا اللابتوب:  0\n"
            "الموبايل واللابتوب لازم يكونوا على نفس الواي فاي."
        )
        help_.setObjectName("hint")
        lay.addWidget(help_)

        test = QPushButton("جرّب الاتصال")
        test.setObjectName("plainButton")
        test.clicked.connect(self.test_camera)
        lay.addWidget(test)

        lay.addSpacing(8)
        lay.addWidget(self._label("شكل الأرقام اللي بتتكتب على الاستيكر"))
        self.digits = QComboBox()
        self.digits.addItem("عربي  ٠١٢٣٤٥٦٧٨٩", "arabic")
        self.digits.addItem("إنجليزي  0123456789", "english")
        self.digits.setCurrentIndex(max(0, self.digits.findData(cfg["digits"])))
        lay.addWidget(self.digits)

        lay.addSpacing(8)
        self.auto = QCheckBox("احفظ لوحده لما كل الأرقام تتقري بثقة (من غير مراجعة)")
        self.auto.setChecked(bool(cfg["auto_save"]))
        lay.addWidget(self.auto)

        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("إلغاء")
        cancel.setObjectName("plainButton")
        ok = QPushButton("حفظ")
        ok.setObjectName("saveButton")
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(ok)
        lay.addSpacing(10)
        lay.addLayout(row)

    def _label(self, t):
        l = QLabel(t)
        l.setObjectName("fieldLabel")
        return l

    def test_camera(self):
        src = self.cam.text().strip()
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW) if src.isdigit() \
            else cv2.VideoCapture(src)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            QMessageBox.information(self, "الكاميرا", "الكاميرا شغالة ✔")
        else:
            QMessageBox.warning(
                self, "الكاميرا",
                "مش قادر يوصل للكاميرا.\n"
                "اتأكد إن تطبيق الكاميرا شغال على الموبايل، "
                "وإن العنوان صح، وإنكم على نفس الواي فاي."
            )

    def values(self):
        return {
            "camera": self.cam.text().strip() or "0",
            "digits": self.digits.currentData(),
            "auto_save": self.auto.isChecked(),
        }


# ============================================================
# MAIN WINDOW
# ============================================================

class ClothesWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{TITLE_AR} — {CAMP_AR}")
        self.setMinimumSize(1250, 760)

        self.cfg = load_config()
        self.locations = load_locations()
        self.last_signature = None

        tpl = cv2.imread(str(TEMPLATE_FILE))
        if tpl is None:
            raise SystemExit(f"Sticker template missing: {TEMPLATE_FILE}")
        self.detector = StickerDetector(tpl)
        self.reader = DigitReader(self.cfg["digits"])

        root = AirmailFrame()
        root.setLayoutDirection(Qt.RightToLeft)
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        b = AirmailFrame.BAND + 16
        outer.setContentsMargins(b, b - 4, b, b - 4)
        outer.setSpacing(8)

        outer.addLayout(self.build_header())

        body = QHBoxLayout()
        body.setSpacing(12)

        # Camera
        cam_col = QVBoxLayout()
        self.preview = QLabel("الكاميرا")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(560, 380)
        self.banner = QLabel("…")
        self.banner.setObjectName("banner")
        self.banner.setAlignment(Qt.AlignCenter)
        cam_col.addWidget(self.preview, 1)
        cam_col.addWidget(self.banner)
        body.addLayout(cam_col, 5)

        # Right side: stats or review
        self.side = QStackedWidget()
        self.side.addWidget(self.build_stats())
        self.review = ReviewPanel()
        self.review.saved.connect(self.save_box)
        self.review.discarded.connect(self.discard_box)
        self.side.addWidget(self.review)
        body.addWidget(self.side, 4)

        outer.addLayout(body, 1)
        outer.addLayout(self.build_bottom())

        self.apply_styles()
        self.refresh()

        QShortcut(QKeySequence(Qt.Key_Return), self, self.shortcut_save)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self.shortcut_save)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.shortcut_discard)

        self.auto_timer = QTimer(self)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.timeout.connect(self.review.on_save)

        self.worker = CameraWorker(self.cfg["camera"], self.detector)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.captured.connect(self.on_captured)
        self.worker.start()

    # ---------------- layout
    def build_header(self):
        row = QHBoxLayout()
        row.setSpacing(14)

        logo = QLabel()
        pix = QPixmap(str(TEMPLATE_FILE))
        if not pix.isNull():
            logo.setPixmap(pix.copy(945, 80, 135, 232).scaled(
                60, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        row.addWidget(logo)

        names = QVBoxLayout()
        names.setSpacing(0)
        t = QLabel(TITLE_AR)
        t.setObjectName("title")
        s = QLabel(f"{CAMP_AR}  •  الكاميرا بتقرا الاستيكر وتعد لوحدها")
        s.setObjectName("subtitle")
        names.addStretch(1)
        names.addWidget(t)
        names.addWidget(s)
        names.addStretch(1)
        row.addLayout(names)
        row.addStretch(1)

        stamp = QLabel()
        if not pix.isNull():
            stamp.setPixmap(pix.copy(405, 150, 340, 300).scaled(
                120, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        row.addWidget(stamp)
        return row

    def build_stats(self):
        w = QFrame()
        w.setObjectName("panel")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(8)

        cap = QLabel("إجمالي القطع")
        cap.setObjectName("panelTitle")
        cap.setAlignment(Qt.AlignCenter)
        self.big = QLabel("0")
        self.big.setObjectName("big")
        self.big.setAlignment(Qt.AlignCenter)
        self.boxes_lbl = QLabel()
        self.boxes_lbl.setObjectName("subtitle")
        self.boxes_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(cap)
        lay.addWidget(self.big)
        lay.addWidget(self.boxes_lbl)

        grid = QGridLayout()
        grid.setSpacing(8)
        self.tiles = {}
        colors = [AIR_BLUE, AIR_RED, INK, OK_GREEN]
        for i, (key, label, _) in enumerate(CATEGORIES):
            t = StatTile(label, colors[i])
            self.tiles[key] = t
            grid.addWidget(t, i // 2, i % 2)
        lay.addLayout(grid)

        self.unsorted_lbl = QLabel()
        self.unsorted_lbl.setObjectName("hint")
        self.unsorted_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.unsorted_lbl)

        loc_t = QLabel("حسب المكان")
        loc_t.setObjectName("fieldLabel")
        lay.addWidget(loc_t)
        self.loc_lbl = QLabel()
        self.loc_lbl.setObjectName("locList")
        self.loc_lbl.setWordWrap(True)
        self.loc_lbl.setAlignment(Qt.AlignTop | Qt.AlignRight)
        lay.addWidget(self.loc_lbl, 1)
        return w

    def build_bottom(self):
        row = QHBoxLayout()
        row.setSpacing(10)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["#", "الوقت", "الإجمالي", "أولاد", "بنات", "رجال", "سيدات", "المكان"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setFixedHeight(170)
        row.addWidget(self.table, 1)

        btns = QGridLayout()
        btns.setSpacing(6)
        spec = [
            ("إضافة يدوي", self.manual_box, "plainButton"),
            ("مسح المحدد", self.delete_selected, "plainButton"),
            ("إيقاف الكاميرا", self.toggle_pause, "plainButton"),
            ("تصدير Excel", self.export_excel, "saveButton"),
            ("الإعدادات", self.open_settings, "plainButton"),
            ("ريسيت", self.reset_all, "dangerButton"),
        ]
        for i, (text, fn, name) in enumerate(spec):
            bt = QPushButton(text)
            bt.setObjectName(name)
            bt.setMinimumHeight(44)
            bt.setCursor(Qt.PointingHandCursor)
            bt.clicked.connect(fn)
            btns.addWidget(bt, i // 2, i % 2)
            if fn == self.toggle_pause:
                self.pause_btn = bt
        row.addLayout(btns)
        return row

    # ---------------- camera events
    def on_frame(self, img):
        pix = QPixmap.fromImage(img).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview.setPixmap(pix)

    def on_status(self, text, kind):
        color = {
            "search": INK_SOFT, "hold": WARN, "captured": OK_GREEN,
            "clear": AIR_BLUE, "error": AIR_RED, "wait": INK_SOFT,
        }.get(kind, INK)
        self.banner.setText(text)
        self.banner.setStyleSheet(f"color: {color};")

    def on_captured(self, canon, det):
        reading = read_sticker(canon, self.detector, self.reader)
        self.pending = reading
        self.review.show_reading(reading, self.locations)
        self.side.setCurrentWidget(self.review)

        # Same ink pattern as the box just saved? Probably scanned twice.
        if self.last_signature is not None:
            a, b = reading["signature"], self.last_signature
            if a.any() and np.abs(a.astype(int) - b.astype(int)).mean() < 0.02:
                self.review.check.setText(
                    "⚠ شكلها نفس الكرتونة اللي لسه متسجلة — اتأكد قبل الحفظ"
                )
                self.review.check.setStyleSheet(f"color: {AIR_RED};")
                return

        if self.cfg["auto_save"] and self.review.confident():
            self.on_status("كل الأرقام واضحة — هيتحفظ لوحده…", "captured")
            self.auto_timer.start(1500)

    # ---------------- save / discard
    def save_box(self, values):
        self.auto_timer.stop()
        reading = getattr(self, "pending", None)
        rows = load_boxes()
        box_id = max([to_int(r.get("id")) for r in rows] + [0]) + 1

        image_name = ""
        if reading is not None:
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            image_name = f"box_{box_id:05d}.jpg"
            cv2.imwrite(str(IMAGES_DIR / image_name), reading["image"],
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.learn_from(reading, values)
            self.last_signature = reading["signature"]

        row = {
            "id": box_id,
            "time": datetime.now().isoformat(timespec="seconds"),
            "total": values["total"],
            "location": values["location"],
            "source": "camera" if reading is not None else "manual",
            "image": image_name,
        }
        for key, _, _ in CATEGORIES:
            row[key] = values[key]
        append_box(row)

        loc = values["location"]
        if loc and loc not in self.locations:
            self.locations.insert(0, loc)
            save_locations(self.locations)

        self.finish_review()
        self.on_status(f"اتسجلت كرتونة #{box_id} — {values['total']} قطعة ✔", "captured")

    def learn_from(self, reading, values):
        """Fields the operator corrected (or that were unsure) become samples."""
        learned = 0
        fields = [(k, reading["cats"][k], values[k]) for k, _, _ in CATEGORIES]
        if values["_other_used"]:
            fields.append(("other", reading["other"], values["_other"]))
        for _, read, final in fields:
            if final <= 0:
                continue
            if read["value"] != final or read["conf"] < 0.8:
                learned += self.reader.learn(read["glyphs"], final)
        if learned:
            self.reader.train()

    def discard_box(self):
        self.auto_timer.stop()
        self.finish_review()
        self.on_status("اتلغت — هات الكرتونة اللي بعدها", "search")

    def finish_review(self):
        self.pending = None
        self.side.setCurrentIndex(0)
        self.worker.rearm()
        self.refresh()

    def shortcut_save(self):
        if self.side.currentWidget() is self.review:
            self.review.on_save()

    def shortcut_discard(self):
        if self.side.currentWidget() is self.review:
            self.discard_box()

    def manual_box(self):
        self.pending = None
        self.review.show_reading(None, self.locations)
        self.side.setCurrentWidget(self.review)
        self.worker.state = CameraWorker.REVIEW

    # ---------------- data views
    def refresh(self):
        rows = load_boxes()
        st = box_stats(rows)
        self.big.setText(f"{st['pieces']:,}")
        self.boxes_lbl.setText(f"{st['boxes']:,} كرتونة")
        for key, _, _ in CATEGORIES:
            self.tiles[key].value.setText(f"{st[key]:,}")
        self.unsorted_lbl.setText(
            f"قطع من غير فئة: {st['unsorted']:,}" if st["unsorted"] else ""
        )
        locs = sorted(st["by_location"].items(), key=lambda kv: -kv[1])
        self.loc_lbl.setText(
            "\n".join(f"{name}:  {n:,} قطعة" for name, n in locs[:8]) or "—"
        )

        self.table.setRowCount(0)
        for r in reversed(rows[-60:]):
            i = self.table.rowCount()
            self.table.insertRow(i)
            t = r.get("time", "")[11:16]
            vals = [r.get("id"), t, r.get("total")] + \
                   [r.get(k) for k, _, _ in CATEGORIES] + [r.get("location")]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v or ""))
                it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, c, it)

    def delete_selected(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "مسح", "اختار كرتونة من الجدول الأول.")
            return
        box_id = self.table.item(sel[0].row(), 0).text()
        if QMessageBox.question(
            self, "مسح", f"تمسح كرتونة #{box_id}؟",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        ) != QMessageBox.Yes:
            return
        write_boxes([r for r in load_boxes() if str(r.get("id")) != box_id])
        self.refresh()

    def toggle_pause(self):
        self.worker.paused = not self.worker.paused
        self.pause_btn.setText("تشغيل الكاميرا" if self.worker.paused else "إيقاف الكاميرا")
        if self.worker.paused:
            self.on_status("الكاميرا متوقفة", "wait")

    def open_settings(self):
        dlg = SettingsDialog(self, self.cfg)
        dlg.setStyleSheet(self.styleSheet())
        if dlg.exec() != QDialog.Accepted:
            return
        new = dlg.values()
        if new["digits"] != self.cfg["digits"]:
            self.reader = DigitReader(new["digits"])
        if new["camera"] != self.cfg["camera"]:
            self.worker.set_source(new["camera"])
        self.cfg.update(new)
        save_config(self.cfg)

    def reset_all(self):
        if QMessageBox.warning(
            self, "ريسيت",
            "تمسح كل الكراتين المتسجلة؟\n\n"
            "هيتعمل نسخة احتياطية الأول في data/clothes/backup.\n"
            "الإعدادات والأماكن وتعلّم الخط هيفضلوا زي ما هما.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        ) != QMessageBox.Yes:
            return
        backup = DATA_DIR / "backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        n = 2
        while backup.exists():
            backup = backup.with_name(f"{backup.name.split('-')[0]}-{n}")
            n += 1
        try:
            backup.mkdir(parents=True)
            if BOXES_FILE.exists():
                shutil.copy2(BOXES_FILE, backup / BOXES_FILE.name)
            if IMAGES_DIR.exists():
                shutil.copytree(IMAGES_DIR, backup / "images")
        except Exception as e:
            QMessageBox.critical(self, "ريسيت", f"النسخة الاحتياطية فشلت، مفيش حاجة اتمسحت.\n{e}")
            return
        BOXES_FILE.unlink(missing_ok=True)
        shutil.rmtree(IMAGES_DIR, ignore_errors=True)
        self.last_signature = None
        self.refresh()
        QMessageBox.information(self, "ريسيت", f"تم. النسخة الاحتياطية في:\n{backup}")

    def export_excel(self):
        default = Path.home() / "Documents" / f"Clothes_{datetime.now():%Y%m%d_%H%M}.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "تصدير Excel", str(default), "Excel (*.xlsx)"
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            export_workbook(path, load_boxes())
        except PermissionError:
            QMessageBox.warning(self, "Excel", "اقفل الملف في Excel وجرّب تاني.")
            return
        except Exception as e:
            QMessageBox.critical(self, "Excel", f"التصدير فشل:\n{e}")
            return
        QMessageBox.information(self, "Excel", f"اتحفظ:\n{path}")

    # ---------------- style
    def apply_styles(self):
        self.setStyleSheet(f"""
            QWidget {{ font-family: {FONT}; color: {INK}; }}
            QLabel#title {{ font-size: 34px; font-weight: 900; }}
            QLabel#subtitle {{ font-size: 14px; font-weight: 700; color: {INK_SOFT}; }}
            QLabel#preview {{
                background: #1b1b1b; color: #bbb; border: 3px solid {INK};
                border-radius: 10px;
            }}
            QLabel#banner {{ font-size: 22px; font-weight: 900; }}
            QFrame#panel {{
                background: {PAPER}; border: 2px dashed {INK_SOFT}; border-radius: 10px;
            }}
            QLabel#panelTitle {{ font-size: 20px; font-weight: 900; }}
            QLabel#big {{ font-size: 84px; font-weight: 900; color: {AIR_RED}; }}
            QFrame#tile {{ background: white; border: 1px solid {KRAFT_DARK}; border-radius: 8px; }}
            QLabel#tileTitle {{ font-size: 16px; font-weight: 800; color: {INK_SOFT}; }}
            QLabel#tileValue {{ font-size: 32px; font-weight: 900; }}
            QLabel#fieldLabel {{ font-size: 15px; font-weight: 900; }}
            QLabel#hint {{ font-size: 12px; color: {INK_SOFT}; }}
            QLabel#checkLabel {{ font-size: 15px; font-weight: 800; }}
            QLabel#locList {{ font-size: 14px; font-weight: 700; }}
            QSpinBox, QComboBox, QLineEdit {{
                background: white; border: 1px solid {KRAFT_DARK}; border-radius: 6px;
                padding: 5px 8px; font-size: 16px; font-weight: 800;
            }}
            QPushButton {{ border-radius: 8px; padding: 6px 12px; font-size: 14px; font-weight: 800; }}
            QPushButton#saveButton {{ background: {AIR_BLUE}; color: white; }}
            QPushButton#plainButton {{ background: white; border: 1px solid {KRAFT_DARK}; }}
            QPushButton#dangerButton {{ background: white; color: {AIR_RED}; border: 2px solid {AIR_RED}; }}
            QPushButton:hover {{ border: 2px solid {INK}; }}
            QTableWidget {{
                background: white; border: 1px solid {KRAFT_DARK}; border-radius: 8px;
                font-size: 13px; gridline-color: #e6dccb;
            }}
            QHeaderView::section {{
                background: {INK}; color: white; padding: 5px; border: none;
                font-weight: 800; font-size: 13px;
            }}
            QDialog {{ background: {KRAFT}; }}
        """)

    def closeEvent(self, e):
        self.worker.stop()
        self.worker.wait(2000)
        e.accept()


# ============================================================
# EXCEL
# ============================================================

def export_workbook(path, rows):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="1E1B18")

    wb = Workbook()
    ws = wb.active
    ws.title = "Boxes"
    headers = ["#", "Time", "Total", "Boys", "Girls", "Men", "Women",
               "Unsorted", "Location", "Source", "Photo", "Check"]
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        ws.cell(1, c).font = head_font
        ws.cell(1, c).fill = head_fill
        ws.cell(1, c).alignment = Alignment(horizontal="center")

    for i, r in enumerate(rows, start=2):
        try:
            t = datetime.fromisoformat(r.get("time", ""))
        except Exception:
            t = r.get("time", "")
        blank = lambda v: None if str(v or "").strip() == "" else to_int(v)
        ws.append([
            to_int(r.get("id")), t, to_int(r.get("total")),
            blank(r.get("boys")), blank(r.get("girls")),
            blank(r.get("men")), blank(r.get("women")),
            f"=MAX(0,C{i}-SUM(D{i}:G{i}))",
            r.get("location", ""), r.get("source", ""), r.get("image", ""), "⚠" if r.get("check") == "1" else "",
        ])
        ws.cell(i, 2).number_format = "yyyy-mm-dd hh:mm"
    last = len(rows) + 1

    for col, w in zip("ABCDEFGHIJKL", [7, 17, 10, 9, 9, 9, 9, 12, 22, 10, 16, 9]):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"

    sm = wb.create_sheet("Summary", 0)
    sm["A1"] = "Clothes Counter — Working Camp 2026"
    sm["A1"].font = Font(bold=True, size=14)
    lines = [
        ("Boxes", f"=COUNT(Boxes!A2:A{max(2, last)})"),
        ("Total pieces", f"=SUM(Boxes!C2:C{max(2, last)})"),
        ("Boys", f"=SUM(Boxes!D2:D{max(2, last)})"),
        ("Girls", f"=SUM(Boxes!E2:E{max(2, last)})"),
        ("Men", f"=SUM(Boxes!F2:F{max(2, last)})"),
        ("Women", f"=SUM(Boxes!G2:G{max(2, last)})"),
        ("Unsorted", f"=SUM(Boxes!H2:H{max(2, last)})"),
    ]
    for i, (k, v) in enumerate(lines, start=3):
        sm.cell(i, 1, k).font = Font(bold=True)
        sm.cell(i, 2, v)

    r0 = 3 + len(lines) + 1
    sm.cell(r0, 1, "Location").font = head_font
    sm.cell(r0, 1).fill = head_fill
    sm.cell(r0, 2, "Pieces").font = head_font
    sm.cell(r0, 2).fill = head_fill
    sm.cell(r0, 3, "Boxes").font = head_font
    sm.cell(r0, 3).fill = head_fill
    places = sorted({r.get("location", "") or "—" for r in rows})
    for j, place in enumerate(places, start=r0 + 1):
        sm.cell(j, 1, place)
        crit = place if place != "—" else ""
        sm.cell(j, 2, f'=SUMIF(Boxes!I2:I{max(2, last)},"{crit}",Boxes!C2:C{max(2, last)})')
        sm.cell(j, 3, f'=COUNTIF(Boxes!I2:I{max(2, last)},"{crit}")')
    sm.column_dimensions["A"].width = 24
    sm.column_dimensions["B"].width = 14
    sm.column_dimensions["C"].width = 12

    wb.save(path)


# ============================================================
# MAIN
# ============================================================

def load_fonts():
    for f in list((ASSETS / "fonts").glob("*.ttf")) + list((ASSETS / "fonts").glob("*.otf")):
        QFontDatabase.addApplicationFont(str(f))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(TITLE_AR)
    load_fonts()
    app.setFont(qfont(14, QFont.Normal))
    win = ClothesWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
