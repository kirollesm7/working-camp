"""
A small convolutional neural network written in plain NumPy (no PyTorch).

    input 28x28
    conv 3x3 x16 -> ReLU -> maxpool 2   (16 x 14 x 14)
    conv 3x3 x32 -> ReLU -> maxpool 2   (32 x 7 x 7)
    dense 1568 -> 128 -> ReLU -> dropout
    dense 128 -> 10  (softmax)

Trained with Adam; weights are saved to / loaded from an .npz file.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _conv_forward(x, W, b):
    """3x3 convolution, padding 1, stride 1. x: (N, C, H, W)."""
    N, C, H, Wd = x.shape
    F = W.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = sliding_window_view(xp, (3, 3), axis=(2, 3))          # N,C,H,W,3,3
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * Wd, C * 9)
    out = cols @ W.reshape(F, -1).T + b
    return out.reshape(N, H, Wd, F).transpose(0, 3, 1, 2), cols


def _tmat(a, b, chunk=128):
    """a.T @ b for very tall a, b. Done in row chunks: this NumPy build is
    5x slower on one long reduction than on several short ones."""
    out = a[:chunk].T @ b[:chunk]
    for i in range(chunk, len(a), chunk):
        out += a[i:i + chunk].T @ b[i:i + chunk]
    return out


def _conv_backward(dout, cols, x_shape, W, need_dx=True):
    F = W.shape[0]
    d = dout.transpose(0, 2, 3, 1).reshape(-1, F)                 # N*H*W, F
    dW = _tmat(d, cols).reshape(W.shape)
    db = d.sum(axis=0)
    if not need_dx:
        return None, dW, db
    # gradient w.r.t. the input = a 3x3 convolution of dout with the
    # weights turned 180 degrees (one big matrix product, no loops)
    W_rot = np.ascontiguousarray(W.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1])
    dx, _ = _conv_forward(dout, W_rot, np.zeros(W_rot.shape[0], dout.dtype))
    return dx, dW, db


def _pool_forward(x):
    N, C, H, W = x.shape
    r = x.reshape(N, C, H // 2, 2, W // 2, 2)
    out = r.max(axis=(3, 5))
    mask = r == out[:, :, :, None, :, None]
    return out, mask


def _pool_backward(dout, mask):
    d = mask * dout[:, :, :, None, :, None]
    N, C, h, _, w, _ = d.shape
    return d.reshape(N, C, h * 2, w * 2)


class CNN:

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        he = lambda shape, fan_in: (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.p = {
            "W1": he((16, 1, 3, 3), 9), "b1": np.zeros(16, np.float32),
            "W2": he((32, 16, 3, 3), 144), "b2": np.zeros(32, np.float32),
            "W3": he((1568, 128), 1568), "b3": np.zeros(128, np.float32),
            "W4": he((128, 10), 128), "b4": np.zeros(10, np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ---------------- forward / backward
    def _forward(self, x, train=False, drop=0.25, rng=None):
        p = self.p
        c1, cols1 = _conv_forward(x, p["W1"], p["b1"])
        r1 = np.maximum(c1, 0)
        p1, m1 = _pool_forward(r1)
        c2, cols2 = _conv_forward(p1, p["W2"], p["b2"])
        r2 = np.maximum(c2, 0)
        p2, m2 = _pool_forward(r2)
        f = p2.reshape(len(x), -1)
        h = np.maximum(f @ p["W3"] + p["b3"], 0)
        keep = None
        if train and drop > 0:
            keep = (rng.random(h.shape) >= drop).astype(np.float32) / (1 - drop)
            h = h * keep
        logits = h @ p["W4"] + p["b4"]
        cache = (x, c1, cols1, m1, p1, c2, cols2, m2, p2, f, h, keep)
        return logits, cache

    def _backward(self, dlogits, cache):
        p = self.p
        x, c1, cols1, m1, p1, c2, cols2, m2, p2, f, h, keep = cache
        g = {}
        g["W4"] = _tmat(h, dlogits)
        g["b4"] = dlogits.sum(0)
        dh = dlogits @ p["W4"].T
        if keep is not None:
            dh = dh * keep
        dh = dh * (h > 0)
        g["W3"] = _tmat(f, dh)
        g["b3"] = dh.sum(0)
        dp2 = (dh @ p["W3"].T).reshape(p2.shape)
        dr2 = _pool_backward(dp2, m2) * (c2 > 0)
        dp1, g["W2"], g["b2"] = _conv_backward(dr2, cols2, p1.shape, p["W2"])
        dr1 = _pool_backward(dp1, m1) * (c1 > 0)
        _, g["W1"], g["b1"] = _conv_backward(dr1, cols1, x.shape, p["W1"], need_dx=False)
        return g

    # ---------------- training
    def train_batch(self, x, y, lr=1e-3, wd=1e-4, rng=None):
        """One Adam step on a batch. x: (N,1,28,28) float32 0..1, y: (N,) int."""
        logits, cache = self._forward(x, train=True, rng=rng)
        logits = logits - logits.max(1, keepdims=True)
        e = np.exp(logits)
        prob = e / e.sum(1, keepdims=True)
        n = len(y)
        loss = -np.log(prob[np.arange(n), y] + 1e-9).mean()
        d = prob
        d[np.arange(n), y] -= 1
        d /= n
        g = self._backward(d.astype(np.float32), cache)
        self.t += 1
        b1, b2 = 0.9, 0.999
        for k in self.p:
            gk = g[k] + (wd * self.p[k] if k.startswith("W") else 0)
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        return float(loss)

    # ---------------- inference
    def predict_proba(self, x, batch=512):
        out = []
        for i in range(0, len(x), batch):
            logits, _ = self._forward(x[i:i + batch])
            logits = logits - logits.max(1, keepdims=True)
            e = np.exp(logits)
            out.append(e / e.sum(1, keepdims=True))
        return np.concatenate(out) if out else np.zeros((0, 10), np.float32)

    # ---------------- storage
    def save(self, path):
        np.savez_compressed(path, **self.p)

    @classmethod
    def load(cls, path):
        net = cls()
        z = np.load(path)
        for k in net.p:
            net.p[k] = z[k].astype(np.float32)
        return net
