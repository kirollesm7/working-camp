"""
Train the CNN digit reader for the clothes counter (no extra libraries).

Data comes from what is already on disk:
  * MNIST — 60,000 real handwritten digits (data/clothes/mnist)
  * your own digits learned from confirmed boxes (data/clothes/digit_samples)

Every epoch each digit is distorted again ON THE FLY (turned, slanted,
squashed, thicker/thinner pen, broken strokes, specks, blur) — nothing
extra is written to disk. Only the trained weights are saved.

    python train_cnn.py            # up to 20 epochs (stops early when it stops improving)
    python train_cnn.py 30         # allow more epochs
    python train_cnn.py tune 4     # carry on from the current model for a few
                                   # epochs (after learning new real digits)

Restart the clothes server afterwards to use the new model.
"""

import sys
import gzip
import time

import cv2
import numpy as np

import clothes_counter as cc
from cnn import CNN

USER_REPEAT = 40          # each of your own digits is shown this many times per epoch


def load_mnist(kind):
    base = cc.DATA_DIR / "mnist"
    imgs = np.frombuffer(gzip.open(base / f"{kind}-images-idx3-ubyte.gz").read(),
                         np.uint8, offset=16).reshape(-1, 28, 28)
    labels = np.frombuffer(gzip.open(base / f"{kind}-labels-idx1-ubyte.gz").read(),
                           np.uint8, offset=8)
    return imgs, labels.astype(np.int64)


def user_digits():
    imgs, labels = [], []
    base = cc.SAMPLES_DIR / "english_raw"
    if base.exists():
        for d in base.iterdir():
            if d.is_dir() and d.name.isdigit():
                for f in d.glob("*.png"):
                    g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                    if g is not None and g.shape == (28, 28):
                        imgs.append(g)
                        labels.append(int(d.name))
    return imgs, labels


def hook_one(img, rng):
    """Many people write 1 with a flag on top (and sometimes a foot). MNIST
    has few of those, so the reader took them for 4, 7 or 9. Draw the flag
    onto a plain MNIST 1."""
    ys, xs = np.nonzero(img > 100)
    if not len(ys):
        return img
    g = img.copy()
    top = ys.min()
    x0 = int(xs[ys <= top + 1].mean())
    t = int(rng.integers(2, 4))
    dx = int(rng.integers(3, 7))
    dy = max(2, int(round(dx * rng.uniform(0.8, 1.4))))     # steep, not a 7's bar
    cv2.line(g, (x0, top), (max(0, x0 - dx), min(27, top + dy)), 255, t)
    if rng.random() < 0.25:                    # a foot under the 1
        bot = ys.max()
        xb = int(xs[ys >= bot - 1].mean())
        w = int(rng.integers(3, 7))
        cv2.line(g, (max(0, xb - w), bot), (min(27, xb + w), bot), 255, t)
    return g


def distort(img, rng, label=None):
    """One random camera/handwriting distortion of a 28x28 digit, then the
    same normalising the app uses when it reads a sticker."""
    g = img
    if label == 1 and rng.random() < 0.5:
        g = hook_one(g, rng)
    # pen thickness
    r = rng.random()
    if r < 0.25:
        g = cv2.erode(g, np.ones((2, 2), np.uint8))
    elif r < 0.55:
        g = cv2.dilate(g, np.ones((2, 2) if rng.random() < 0.6 else (3, 3), np.uint8))
    # turn, slant, squash
    a = np.deg2rad(rng.uniform(-15, 15))
    sh = rng.uniform(-0.35, 0.35)
    sx, sy = rng.uniform(0.75, 1.25), rng.uniform(0.85, 1.15)
    M = np.array([[np.cos(a) * sx, -np.sin(a) + sh, 0],
                  [np.sin(a), np.cos(a) * sy, 0]], np.float32)
    M[:, 2] = np.array([14, 14]) - M[:, :2] @ np.array([14, 14])
    g = cv2.warpAffine(g, M, (28, 28), flags=cv2.INTER_LINEAR)
    # blur (out of focus)
    if rng.random() < 0.3:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.4, 1.0))
    ink = (g > rng.uniform(70, 140)).astype(np.uint8)
    # broken stroke (faint pen / glare)
    if rng.random() < 0.15:
        ys, xs = np.nonzero(ink)
        if len(xs):
            k = rng.integers(len(xs))
            s = rng.integers(2, 4)
            ink[max(0, ys[k] - s):ys[k] + s, max(0, xs[k] - s):xs[k] + s] = 0
    # specks of dirt
    if rng.random() < 0.15:
        for _ in range(rng.integers(1, 3)):
            ink[rng.integers(0, 28), rng.integers(0, 28)] = 1
    out = cc.normalize_glyph(ink)
    return out if out is not None else cc.normalize_glyph((img > 100).astype(np.uint8))


def to_input(glyphs):
    return (np.float32(glyphs) / 255.0)[:, None, :, :]


def real_validation(detector, cutter):
    """Digits cut from real sticker photos the reader never learned from
    (data/clothes/real_validation). Returns (inputs, labels)."""
    import json
    base = cc.DATA_DIR / "real_validation"
    if not (base / "truth.json").exists():
        return None, None
    glyphs, labels = [], []
    for item in json.loads((base / "truth.json").read_text(encoding="utf-8")):
        canon = cv2.imread(str(base / item["image"]))
        if canon is None:
            continue
        for key, _, rect in cc.CATEGORIES:
            v = str(item.get(key, "")).strip()
            if not v.isdigit():
                continue
            pieces = cutter._pieces(cc.field_ink(canon, detector.printed, rect))
            if len(pieces) != len(str(int(v))):
                continue
            for p, ch in zip(pieces, str(int(v))):
                glyphs.append(cc.normalize_glyph(p[0]))
                labels.append(int(ch))
    if not glyphs:
        return None, None
    return to_input(glyphs), np.array(labels)


def main():
    args = sys.argv[1:]
    tune = bool(args) and args[0] == "tune"
    if tune:
        args = args[1:]
    max_epochs = int(args[0]) if args else (4 if tune else 20)
    patience = 3                               # early stopping
    cc.NORM_THIN = False                       # English handwriting mode
    rng = np.random.default_rng(int(time.time()))

    tr_x, tr_y = load_mnist("train")
    te_x, te_y = load_mnist("t10k")
    ux, uy = user_digits()
    print(f"training digits: {len(tr_y)} handwritten + {len(uy)} of yours (x{USER_REPEAT})")

    # Checks (none of these digits are trained on):
    #   clean  — 5,000 handwritten test digits as written
    #   hard   — 5,000 other test digits distorted (fixed), chooses the best model
    #   real   — digits cut from your real sticker photos never learned from
    #   train  — 5,000 training digits distorted the same way: if train is much
    #            higher than hard, the network is memorising (overfitting);
    #            if both are low it hasn't learned enough (underfitting)
    val_clean = to_input([cc.normalize_glyph((g > 100).astype(np.uint8)) for g in te_x[:5000]])
    val_clean_y = te_y[:5000]
    vrng = np.random.default_rng(123)
    val_hard = to_input([distort(g, vrng) for g in te_x[5000:]])
    val_hard_y = te_y[5000:]
    trng = np.random.default_rng(7)
    pick = trng.choice(len(tr_y), 5000, replace=False)
    train_chk = to_input([distort(tr_x[i], trng) for i in pick])
    train_chk_y = tr_y[pick]
    det = cc.StickerDetector(cv2.imread(str(cc.TEMPLATE_FILE)))
    cutter = cc.DigitReader.__new__(cc.DigitReader)
    cutter.style = "english"
    real_x, real_y = real_validation(det, cutter)
    print(f"checks: 5000 clean + 5000 hard test digits, "
          f"{0 if real_y is None else len(real_y)} digits from your held-out real photos")

    acc = lambda net, x, y: float((net.predict_proba(x).argmax(1) == y).mean())

    out = cc.DATA_DIR / "cnn_english.npz"
    batch = 128
    best, best_ep, waited = -1.0, 0, 0
    score = lambda hard, real: hard if np.isnan(real) else (hard + real) / 2
    if tune:
        # Fine-tune: start from the current model, smaller steps, a slice of
        # MNIST each epoch so it doesn't forget, your real digits often.
        start = out if out.exists() else cc.ASSETS / "cnn_english.npz"
        net = CNN.load(start)
        lr, per_epoch = 3e-4, 20000
        a_h = acc(net, val_hard, val_hard_y)
        a_r = acc(net, real_x, real_y) if real_y is not None else float("nan")
        best = score(a_h, a_r)
        print(f"start (from {start.name}): hard {a_h:.2%} | real {a_r:.2%}")
    else:
        net = CNN(seed=int(rng.integers(1 << 30)))
        lr, per_epoch = 1e-3, len(tr_y)
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        idx = np.concatenate([rng.choice(len(tr_y), per_epoch, replace=False),
                              -1 - np.repeat(np.arange(len(uy)), USER_REPEAT)])
        rng.shuffle(idx)
        losses = []
        for b in range(0, len(idx), batch):
            ids = idx[b:b + batch]
            imgs, ys = [], []
            for i in ids:
                if i >= 0:
                    imgs.append(distort(tr_x[i], rng, tr_y[i])); ys.append(tr_y[i])
                else:
                    j = -1 - i
                    imgs.append(distort(ux[j], rng)); ys.append(uy[j])
            losses.append(net.train_batch(to_input(imgs), np.array(ys), lr=lr, rng=rng))

        a_train = acc(net, train_chk, train_chk_y)
        a_clean = acc(net, val_clean, val_clean_y)
        a_hard = acc(net, val_hard, val_hard_y)
        a_real = acc(net, real_x, real_y) if real_y is not None else float("nan")
        gap = a_train - a_hard
        state = "OVERFITTING" if gap > 0.03 else ("underfitting" if a_train < 0.9 else "ok")

        note = ""
        now = score(a_hard, a_real) if tune else a_hard
        if now > best:
            best, best_ep, waited = now, ep, 0
            net.save(out)
            note = "  (saved)"
        else:
            waited += 1
            lr *= 0.5                                   # slow down when stuck
            note = f"  (no gain {waited}/{patience}, lr -> {lr:.1e})"
        print(f"epoch {ep:2d} | loss {np.mean(losses):.3f} | train {a_train:.2%} | clean {a_clean:.2%} | "
              f"hard {a_hard:.2%} | real {a_real:.2%} | gap {gap:+.2%} {state} | {time.time() - t0:.0f}s{note}",
              flush=True)
        if waited >= patience:
            print(f"Early stop: no gain for {patience} epochs.")
            break
    if best_ep == 0:
        print("No epoch beat the starting model — kept it.")
    else:
        print(f"Best model: epoch {best_ep}, score {best:.2%}. Saved to {out}")
    print("Restart the clothes server to use it.")


if __name__ == "__main__":
    main()
