"""
Train the clothes counter's digit reader on camera-like photos.

Real handwritten digits (MNIST training set) are written onto the sticker,
photographed at random angles in normal light and in the dark (blur, noise,
shadows, JPEG), then cut out exactly the way the app does it. Every digit
that comes out cleanly becomes a training example.

Run it as often as you like — each run ADDS more examples:

    python train_clothes.py            # 1500 photos (~5-10 min)
    python train_clothes.py 4000       # more photos, more training

Restart the clothes server afterwards to use the new training.
"""

import sys
import gzip
import time
import random

import cv2
import numpy as np

import clothes_counter as cc


def load_mnist_train():
    base = cc.DATA_DIR / "mnist"
    imgs = np.frombuffer(
        gzip.open(base / "train-images-idx3-ubyte.gz").read(), np.uint8, offset=16
    ).reshape(-1, 28, 28)
    labels = np.frombuffer(
        gzip.open(base / "train-labels-idx1-ubyte.gz").read(), np.uint8, offset=8
    )
    return imgs, labels


PENS = [(150, 60, 30), (40, 40, 40), (160, 40, 20), (20, 20, 120), (90, 90, 90)]  # BGR


def write_number(img, rect, value, pen, imgs, by_digit):
    x, y, w, h = rect
    digits = []
    for ch in str(value):
        g = imgs[random.choice(by_digit[int(ch)])]
        ys, xs = np.nonzero(g > 60)
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        hh = random.randint(24, 42)
        ww = max(4, int(g.shape[1] * hh / g.shape[0] * random.uniform(0.85, 1.2)))
        digits.append(cv2.resize(g, (ww, hh), interpolation=cv2.INTER_LINEAR))
    gap = random.randint(0, 7)
    total_w = sum(d.shape[1] for d in digits) + gap * (len(digits) - 1)
    cx = x + (w - total_w) // 2 + random.randint(-12, 12)
    for d in digits:
        dy = y + (h - d.shape[0]) // 2 + random.randint(-4, 4)
        dy = max(y + 2, min(dy, y + h - d.shape[0] - 2))
        cx = max(x + 3, min(cx, x + w - d.shape[1] - 3))
        a = (d.astype(np.float32) / 255.0)[..., None]
        roi = img[dy:dy + d.shape[0], cx:cx + d.shape[1]].astype(np.float32)
        img[dy:dy + d.shape[0], cx:cx + d.shape[1]] = (roi * (1 - a) + np.float32(pen) * a).astype(np.uint8)
        cx += d.shape[1] + gap


def camera(sticker):
    """Photograph the sticker and straighten it again, like the app does."""
    H, W = 720, 1280
    th, tw = sticker.shape[:2]
    sw = random.uniform(0.5, 0.88) * W
    sh = sw * th / tw
    cx, cy = W / 2 + random.uniform(-90, 90), H / 2 + random.uniform(-50, 50)
    j = lambda: random.uniform(-0.09, 0.09) * sw
    dst = np.float32([[cx - sw / 2 + j(), cy - sh / 2 + j()], [cx + sw / 2 + j(), cy - sh / 2 + j()],
                      [cx + sw / 2 + j(), cy + sh / 2 + j()], [cx - sw / 2 + j(), cy + sh / 2 + j()]])
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    M = cv2.getPerspectiveTransform(src, dst)
    bg = np.random.randint(60, 200, (H, W, 3), dtype=np.uint8)
    warped = cv2.warpPerspective(sticker, M, (W, H))
    mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), M, (W, H))
    out = np.where(mask[..., None] > 0, warped, bg).astype(np.float32)

    gx = np.linspace(random.uniform(0.55, 1.0), random.uniform(0.55, 1.0), W)
    gy = np.linspace(random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), H)
    out *= (gy[:, None] * gx[None, :])[..., None]
    dark = random.random() < 0.45
    if dark:
        out *= random.uniform(0.22, 0.5)
        out *= np.float32([random.uniform(0.7, 0.9), random.uniform(0.85, 0.95), 1.05])
        sigma = random.uniform(5, 13)
    else:
        sigma = random.uniform(1.5, 6)
    out = cv2.GaussianBlur(out, (0, 0), random.uniform(0.5, 1.6))
    out += np.random.normal(0, sigma, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, random.randint(55, 90)])
    frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # straighten with a slightly-off homography, as real detection is never perfect
    noisy = dst + np.random.normal(0, 1.5, dst.shape).astype(np.float32)
    Minv = cv2.getPerspectiveTransform(noisy, src)
    return cv2.warpPerspective(frame, Minv, (tw, th), borderValue=(255, 255, 255))


def main():
    photos = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    random.seed(time.time())
    np.random.seed(int(time.time()) % 2**31)

    cc.NORM_THIN = False                          # English handwriting mode
    imgs, labels = load_mnist_train()
    by_digit = {d: np.where(labels == d)[0] for d in range(10)}
    tpl = cv2.imread(str(cc.TEMPLATE_FILE))
    det = cc.StickerDetector(tpl)
    cutter = cc.DigitReader.__new__(cc.DigitReader)   # only its cutting code is used
    cutter.style = "english"
    hog = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)

    X, y = [], []
    kept = dropped = 0
    t0 = time.time()
    for n in range(photos):
        s = tpl.copy()
        pen = random.choice(PENS)
        values = []
        for key, _, rect in cc.CATEGORIES:
            v = random.choice([random.randint(0, 9), random.randint(10, 99),
                               random.randint(10, 99), random.randint(100, 999), ""])
            values.append(v)
            if v != "":
                write_number(s, rect, v, pen, imgs, by_digit)
        canon = camera(s)
        cmin = canon.min(axis=2)
        for (key, _, rect), v in zip(cc.CATEGORIES, values):
            if v == "":
                continue
            pieces = cutter._pieces(cc.field_ink(cmin, det.printed, rect))
            if len(pieces) != len(str(v)):
                dropped += 1                      # cut wrongly: don't learn from it
                continue
            for (sub, _, _, _), ch in zip(pieces, str(v)):
                g = cc.normalize_glyph(sub)
                if g is not None:
                    X.append(hog.compute(g).reshape(-1))
                    y.append(int(ch))
                    kept += 1
        if (n + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {n + 1}/{photos} photos | {kept} digits learned | "
                  f"{dropped} fields skipped | {el:.0f}s", flush=True)

    X = np.float32(X)
    y = np.float32(y)
    out = cc.DATA_DIR / "digit_pipe_english_raw.npz"
    if out.exists():
        old = np.load(out)
        X = np.vstack([old["X"], X])
        y = np.concatenate([old["y"], y])
    np.savez_compressed(out, X=X, y=y)
    # the saved SVM no longer matches: it is rebuilt on next start
    for f in cc.DATA_DIR.glob("digit_svm_english_*.xml"):
        f.unlink()
    print(f"Done: {len(y)} camera-like training digits in total ({kept} new).")
    print("Restart the clothes server to use them.")


if __name__ == "__main__":
    main()
