"""
Clothes Counter — web server (معسكر العمل 2026 • منه وله)

Run it on the laptop:
    python clothes_server.py

  * Dashboard (laptop):  http://localhost:5000
  * Phones (same Wi-Fi): https://<laptop-ip>:5443

The phone streams its camera to the laptop. The laptop finds the sticker,
waits until it is steady, takes the shot, reads it offline (clothes_counter
engine) and sends the numbers back to the phone for a quick confirm.
"""

import io
import os
import json
import time
import uuid
import base64
import socket
import shutil
import ipaddress
import threading
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory, abort
from werkzeug.serving import make_server

import clothes_counter as cc


HTTP_PORT = 5000      # dashboard, this laptop only
HTTPS_PORT = 5443     # phones on the Wi-Fi (camera needs https)

WEB_DIR = Path(__file__).resolve().parent / "web"
CERT_DIR = cc.DATA_DIR / "https"

app = Flask(__name__, static_folder=None)

ENGINE_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]


def ensure_mnist():
    """On a new laptop, fetch the handwritten-digit examples once (~10 MB).
    Without them the reader still works, only weaker on handwriting."""
    import urllib.request
    base = cc.DATA_DIR / "mnist"
    missing = [f for f in MNIST_FILES if not (base / f).exists()]
    if not missing:
        return
    print("Downloading handwriting examples (first run only, ~10 MB)...", flush=True)
    base.mkdir(parents=True, exist_ok=True)
    for f in missing:
        try:
            tmp = base / (f + ".part")
            urllib.request.urlretrieve(MNIST_URL + f, tmp)
            tmp.replace(base / f)
        except Exception as e:
            print(f"  could not download {f} ({e}) — continuing without it", flush=True)
            return


print("Getting the sticker reader ready... (the first start on a new laptop "
      "takes a few minutes — wait for the links)", flush=True)
cc.DATA_DIR.mkdir(parents=True, exist_ok=True)
detector = cc.StickerDetector(cv2.imread(str(cc.TEMPLATE_FILE)))
config = cc.load_config()
if config["digits"] == "english":
    ensure_mnist()
reader = cc.DigitReader(config["digits"])

sessions = {}          # phone id -> Session
saved_readings = {}    # box id -> reading (to learn from later fixes)
box_owner = {}         # box id -> phone that saved it
SIG_DIR = cc.DATA_DIR / "signatures"   # handwriting fingerprint per saved box
SAME_BOX = 0.76        # fingerprint similarity above this = the same sticker
                       # (real photos: different stickers 0.68 max)
MIN_INK = 150          # too little handwriting to tell boxes apart
NEW_BOX = 0.85         # same box in the next frames: 0.95+; another box: 0.68 max
signatures = {}        # box id -> fingerprint


def load_signatures():
    signatures.clear()
    if SIG_DIR.exists():
        for f in SIG_DIR.glob("*.npy"):
            try:
                signatures[int(f.stem)] = np.load(f)
            except Exception:
                pass


load_signatures()
SAVE_LOCK = threading.Lock()


# ============================================================
# AUTO-CAPTURE PER PHONE
# ============================================================

class Session:

    SEARCH, HOLD, REVIEW, CLEAR = "search", "hold", "review", "clear"

    def __init__(self, cid):
        self.cid = cid
        self.state = self.SEARCH
        self.history = []
        self.misses = 0
        self.last_center = None
        self.last_sig = None
        self.seen = time.time()
        self.name = ""

    def step(self, frame, det):
        """Advance the state machine; returns (message, capture or None)."""
        diag = float(np.hypot(*frame.shape[:2]))

        # Never capture a sticker that is partly outside the picture:
        # the cut-off numbers would be read wrong.
        if det is not None and not whole_sticker(det, frame):
            if self.state in (self.SEARCH, self.HOLD):
                self.state, self.history = self.SEARCH, []
                return "Move back — the whole sticker must be in the picture", None

        if self.state == self.SEARCH:
            if det:
                self.history, self.misses = [(frame, det)], 0
                self.state = self.HOLD
                return "Hold the box steady…", None
            return "Point the camera at the sticker", None

        if self.state == self.HOLD:
            if det is None:
                self.misses += 1
                if self.misses > 2:
                    self.state = self.SEARCH
                return "Hold the box steady…", None
            self.misses = 0
            self.history = (self.history + [(frame, det)])[-4:]
            if len(self.history) < 3:
                return "Hold the box steady…", None
            ref = self.history[-1][1]["corners"]
            moved = max(float(np.abs(h[1]["corners"] - ref).max()) for h in self.history)
            if moved > 0.03 * diag:
                return "Hold the box steady…", None

            best = max(
                self.history,
                key=lambda fd: cv2.Laplacian(
                    cv2.cvtColor(fd[0], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(),
            )
            canon = detector.warp(best[0], best[1]["H"])
            self.last_center = best[1]["corners"].mean(axis=0) / diag
            # remember what this box looks like, to spot the next one
            self.last_sig = cc.ink_signature(canon, detector.printed)[0]
            self.history = []
            self.state = self.REVIEW
            return "Captured ✔", canon

        if self.state == self.CLEAR:
            if det is None:
                self.misses += 1
                if self.misses >= 3:
                    self.state = self.SEARCH
                    return "Point the camera at the sticker", None
            else:
                self.misses = 0
                c = det["corners"].mean(axis=0) / diag
                if self.last_center is not None and np.hypot(*(c - self.last_center)) > 0.25:
                    self.state = self.SEARCH       # a new sticker slid in
                    return self.step(frame, det)
                # A different box put in the same spot (phone on a stand,
                # boxes swapped quickly): its handwriting doesn't match the
                # box just saved, so read it now.
                if self.last_sig is not None:
                    with ENGINE_LOCK:
                        sig, ink = cc.ink_signature(detector.warp(frame, det["H"]), detector.printed)
                    if ink >= MIN_INK and cc.signature_similarity(sig, self.last_sig) < NEW_BOX:
                        self.state = self.SEARCH
                        return self.step(frame, det)
            return "Saved ✔ — show the next box", None

        return "Check the numbers on the phone", None


def whole_sticker(det, frame, margin=4):
    """True when all four corners of the sticker are inside the frame."""
    h, w = frame.shape[:2]
    c = det["corners"]
    return bool((c[:, 0] >= margin).all() and (c[:, 0] <= w - margin).all()
                and (c[:, 1] >= margin).all() and (c[:, 1] <= h - margin).all())


def get_session(cid):
    cid = (cid or "anon")[:64]
    with STATE_LOCK:
        s = sessions.get(cid)
        if s is None:
            s = sessions[cid] = Session(cid)
        s.seen = time.time()
        return s


# ============================================================
# HELPERS
# ============================================================

def is_local():
    return request.remote_addr in ("127.0.0.1", "::1")


def local_only():
    if not is_local():
        abort(403)


def jpeg_b64(img, max_w=None, quality=80):
    if max_w and img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def decode_image(data):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def values_from_reading(r):
    """
    Turn a sticker reading into what gets saved.
    Only numbers actually written are kept; an empty box stays empty ("").
    Returns (values, total, needs_check, error).
    """
    check = False
    values = {}
    for key, _, _ in cc.CATEGORIES:
        c = r["cats"][key]
        if not c["glyphs"]:
            values[key] = ""                 # nothing written
        elif c["value"] is None:
            values[key] = ""                 # ink, but unreadable
            check = True
        else:
            values[key] = c["value"]
            check |= c["conf"] < cc.CONF_OK
    written = [v for v in values.values() if v != ""]
    cats = sum(written)

    # The total comes from the tick on the top boxes (٣٠ / ٤٠ / ٥٠, or a
    # number written on the dots). Without a clear tick it is the sum of
    # the numbers below; a tick that disagrees with them is flagged ⚠.
    choice, other = r["total_choice"], r["other"]
    other_val = other["value"] if other["glyphs"] and other["value"] else 0
    clear_tick = r["total_conf"] >= 1.0
    if clear_tick and choice in (30, 40, 50):
        total = choice
    elif clear_tick and choice == "other" and other_val:
        total = other_val
        check |= other["conf"] < cc.CONF_OK
    elif choice is not None and not clear_tick:
        total, check = cats, True            # several boxes inked: unclear
    elif other_val and not written:          # number on the dots, no tick
        total, check = other_val, True
    else:                                    # no tick: sum of the numbers
        total = cats

    if total <= 0:
        return values, 0, check, "Couldn't read any number — take the photo again"
    if cats and cats != total:
        check = True
    return values, total, check, None


def store_box(values, total, check, location, source, reading=None, owner=""):
    """Append one box; returns its id."""
    with SAVE_LOCK:
        rows = cc.load_boxes()
        box_id = max([cc.to_int(x.get("id")) for x in rows] + [0]) + 1

        image_name = ""
        if reading is not None:
            cc.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            image_name = f"box_{box_id:05d}.jpg"
            cv2.imwrite(str(cc.IMAGES_DIR / image_name), reading["image"],
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            if reading.get("ink_amount", 0) >= MIN_INK:
                SIG_DIR.mkdir(parents=True, exist_ok=True)
                np.save(SIG_DIR / f"{box_id}.npy", reading["signature"])
                signatures[box_id] = reading["signature"]
            saved_readings[box_id] = reading
            for k in list(saved_readings)[:-60]:
                saved_readings.pop(k, None)

        cc.append_box({
            "id": box_id,
            "time": datetime.now().isoformat(timespec="seconds"),
            "total": total,
            "location": location,
            "source": source,
            "image": image_name,
            "check": "1" if check else "",
            **values,
        })
        box_owner[box_id] = owner

    if location:
        locs = cc.load_locations()
        if location in locs:
            locs.remove(location)
        cc.save_locations([location] + locs[:49])
    return box_id


def auto_save(canon, session, location):
    """Read a straightened sticker and save it straight away."""
    with ENGINE_LOCK:
        r = cc.read_sticker(canon, detector, reader)

    # The same sticker again (even much later, or after a restart)?
    if r.get("ink_amount", 0) >= MIN_INK and signatures:
        best_id, best = max(
            ((i, cc.signature_similarity(r["signature"], sg))
             for i, sg in list(signatures.items())),
            key=lambda t: t[1],
        )
        if best >= SAME_BOX:
            return {"skipped": f"Already saved as box #{best_id} — not counted again"}

    values, total, check, error = values_from_reading(r)
    if error:
        return {"error": error}

    box_id = store_box(values, total, check, location, "camera", r, session.cid)
    return {"saved": {"id": box_id, "total": total, "check": check,
                      "location": location, **values}}


def learn_from(r, values):
    """Numbers the operator corrected become handwriting samples."""
    learned = 0
    for key, _, _ in cc.CATEGORIES:
        read, final = r["cats"][key], values.get(key, "")
        if final == "" or cc.to_int(final) <= 0:
            continue
        final = cc.to_int(final)
        # Only learn digits someone actually changed — an unchanged
        # (possibly wrong) read must never become a training sample.
        if read["value"] != final:
            learned += reader.learn(read["glyphs"], final)
    return learned


def retrain_async():
    def job():
        with ENGINE_LOCK:
            reader.train()
    threading.Thread(target=job, daemon=True).start()


def clean_values(data):
    """Form values -> category numbers, keeping empty fields empty."""
    out = {}
    for k, _, _ in cc.CATEGORIES:
        v = str(data.get(k, "")).strip()
        out[k] = "" if v == "" else max(0, cc.to_int(v))
    return out


def total_for(values, data):
    """The total typed/ticked by the person; the sum of the written
    numbers when no total is given."""
    total = max(0, cc.to_int(data.get("total")))
    if total > 0:
        return total
    return sum(v for v in values.values() if v != "")


def phone_location():
    return unquote(request.headers.get("X-Location", "")).strip()[:80]


# ============================================================
# PAGES & FILES
# ============================================================

@app.route("/")
def phone_page():
    if is_local() and request.scheme == "http":
        return send_from_directory(WEB_DIR, "dashboard.html")
    return send_from_directory(WEB_DIR, "phone.html")


@app.route("/phone")
def phone_page_direct():
    return send_from_directory(WEB_DIR, "phone.html")


@app.route("/display")
def display_page():
    # Numbers only, for the big screen
    return send_from_directory(WEB_DIR, "display.html")


@app.route("/dashboard")
def dashboard_page():
    local_only()
    return send_from_directory(WEB_DIR, "dashboard.html")


@app.route("/web/<path:name>")
def web_file(name):
    return send_from_directory(WEB_DIR, name)


@app.route("/assets/<path:name>")
def asset_file(name):
    if name not in ("clothes_stamp.png", "clothes_logo.png", "app.ico"):
        abort(404)
    return send_from_directory(cc.ASSETS, name)


@app.route("/images/<path:name>")
def box_image(name):
    local_only()
    return send_from_directory(cc.IMAGES_DIR, name)


# ============================================================
# PHONE API
# ============================================================

@app.post("/api/frame")
def api_frame():
    s = get_session(request.headers.get("X-Client"))
    if not config.get("phone_camera", True):
        return jsonify(state="off", message="Camera stopped from the dashboard — use manual entry")
    frame = decode_image(request.get_data())
    if frame is None:
        return jsonify(error="bad image"), 400

    with ENGINE_LOCK:
        det = detector.detect(frame)
    msg, canon = s.step(frame, det)

    out = {"state": s.state, "message": msg}
    if det is not None:
        h, w = frame.shape[:2]
        out["quad"] = (det["corners"] / [w, h]).round(4).tolist()
    if canon is not None:
        out.update(auto_save(canon, s, phone_location()))
        s.state, s.misses = Session.CLEAR, 0
        out["state"] = s.state
    return jsonify(out)


@app.post("/api/photo")
def api_photo():
    """One still photo (manual shot, or phones without live camera)."""
    s = get_session(request.headers.get("X-Client"))
    if not config.get("phone_camera", True):
        return jsonify(error="Camera stopped from the dashboard — use manual entry"), 423
    f = request.files.get("photo")
    frame = decode_image(f.read() if f else request.get_data())
    if frame is None:
        return jsonify(error="The photo is not clear"), 400
    h, w = frame.shape[:2]
    if max(h, w) > 2000:
        sc = 2000 / max(h, w)
        frame = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    with ENGINE_LOCK:
        det = detector.detect(frame)
    if det is None:
        return jsonify(error="Sticker not found — take the photo again with the whole sticker visible"), 422
    if not whole_sticker(det, frame):
        return jsonify(error="Part of the sticker is outside the photo — move back and take it again"), 422
    out = auto_save(detector.warp(frame, det["H"]), s, phone_location())
    s.state, s.misses = Session.CLEAR, 0
    return jsonify(out), (422 if "error" in out else 200)


@app.post("/api/save")
def api_save():
    """Manual box (no sticker photo)."""
    data = request.get_json(force=True) or {}
    s = get_session(request.headers.get("X-Client"))
    values = clean_values(data)
    total = total_for(values, data)
    if total <= 0:
        return jsonify(error="The total must be greater than zero"), 400
    location = str(data.get("location") or "").strip()[:80]
    box_id = store_box(values, total, False, location,
                       "manual" if is_local() else "phone", None, s.cid)
    return jsonify(ok=True, id=box_id, total=total)


@app.post("/api/update")
def api_update():
    """Fix a saved box. Phones may only fix boxes they saved themselves."""
    data = request.get_json(force=True) or {}
    box_id = cc.to_int(data.get("id"), -1)
    s = get_session(request.headers.get("X-Client"))
    if not is_local() and box_owner.get(box_id) != s.cid:
        abort(403)

    values = clean_values(data)
    total = total_for(values, data)
    if total <= 0:
        return jsonify(error="The total must be greater than zero"), 400

    with SAVE_LOCK:
        rows = cc.load_boxes()
        row = next((r for r in rows if cc.to_int(r.get("id")) == box_id), None)
        if row is None:
            return jsonify(error="Box not found"), 404
        row.update(values)
        row["total"] = total
        row["check"] = ""
        if "location" in data:
            row["location"] = str(data.get("location") or "").strip()[:80]
        cc.write_boxes(rows)

    r = saved_readings.get(box_id)
    if r is not None:
        with ENGINE_LOCK:
            learned = learn_from(r, values)
        if learned:
            retrain_async()
    return jsonify(ok=True, id=box_id, total=total)


@app.post("/api/hello")
def api_hello():
    s = get_session(request.headers.get("X-Client"))
    s.name = str((request.get_json(silent=True) or {}).get("name", ""))[:30]
    if s.state == Session.REVIEW:
        s.state = Session.CLEAR      # page reloaded mid-review
    return jsonify(ok=True)


# ============================================================
# SHARED API
# ============================================================

@app.get("/api/stats")
def api_stats():
    rows = cc.load_boxes()
    st = cc.box_stats(rows)
    today = datetime.now().date().isoformat()
    today_rows = [r for r in rows if r.get("time", "").startswith(today)]
    out = {
        "boxes": st["boxes"],
        "pieces": st["pieces"],
        "unsorted": st["unsorted"],
        "cats": {k: st[k] for k, _, _ in cc.CATEGORIES},
        "today": {"boxes": len(today_rows),
                  "pieces": sum(cc.to_int(r.get("total")) for r in today_rows)},
        "locations": cc.load_locations(),
        "config": {"digits": config["digits"], "auto_save": config["auto_save"],
                   "phone_camera": bool(config.get("phone_camera", True))},
    }
    if is_local():
        out["by_location"] = sorted(st["by_location"].items(), key=lambda kv: -kv[1])
        out["rows"] = list(reversed(rows[-200:]))
        now = time.time()
        with STATE_LOCK:
            out["phones"] = [
                {"id": s.cid[:6], "name": s.name, "state": s.state}
                for s in sessions.values() if now - s.seen < 8
            ]
        out["urls"] = phone_urls()
        out["learned"] = reader.user_count
    return jsonify(out)


@app.delete("/api/box/<box_id>")
def api_delete(box_id):
    local_only()
    rows = cc.load_boxes()
    keep = [r for r in rows if str(r.get("id")) != str(box_id)]
    if len(keep) == len(rows):
        return jsonify(error="not found"), 404
    cc.write_boxes(keep)
    # A deleted box may be scanned again
    signatures.pop(cc.to_int(box_id), None)
    (SIG_DIR / f"{cc.to_int(box_id)}.npy").unlink(missing_ok=True)
    return jsonify(ok=True)


@app.post("/api/learn")
def api_learn():
    """Teach the reader the handwriting on every confirmed saved box."""
    local_only()
    rows = cc.load_boxes()
    with ENGINE_LOCK:
        n = cc.learn_from_boxes(reader, detector, rows)
        if n:
            reader.train()
    confirmed = sum(1 for r in rows if r.get("check") != "1" and r.get("image"))
    return jsonify(ok=True, learned=n, boxes=confirmed)


@app.post("/api/config")
def api_config():
    global reader
    local_only()
    data = request.get_json(force=True) or {}
    if data.get("digits") in ("arabic", "english") and data["digits"] != config["digits"]:
        config["digits"] = data["digits"]
        with ENGINE_LOCK:
            reader = cc.DigitReader(config["digits"])
    if "auto_save" in data:
        config["auto_save"] = bool(data["auto_save"])
    if "phone_camera" in data:
        config["phone_camera"] = bool(data["phone_camera"])
    cc.save_config(config)
    return jsonify(ok=True, config=config)


@app.post("/api/reset")
def api_reset():
    local_only()
    backup = cc.DATA_DIR / "backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
    n = 2
    while backup.exists():
        backup = backup.with_name(f"{backup.name.split('-')[0]}-{n}")
        n += 1
    try:
        backup.mkdir(parents=True)
        if cc.BOXES_FILE.exists():
            shutil.copy2(cc.BOXES_FILE, backup / cc.BOXES_FILE.name)
        if cc.IMAGES_DIR.exists():
            shutil.copytree(cc.IMAGES_DIR, backup / "images")
    except Exception as e:
        return jsonify(error=f"Backup failed, nothing was deleted: {e}"), 500
    cc.BOXES_FILE.unlink(missing_ok=True)
    shutil.rmtree(cc.IMAGES_DIR, ignore_errors=True)
    shutil.rmtree(SIG_DIR, ignore_errors=True)
    signatures.clear()
    return jsonify(ok=True, backup=str(backup))


@app.get("/api/export")
def api_export():
    local_only()
    buf = io.BytesIO()
    cc.export_workbook(buf, cc.load_boxes())
    buf.seek(0)
    return send_file(
        buf, as_attachment=True,
        download_name=f"Clothes_{datetime.now():%Y%m%d_%H%M}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ============================================================
# HTTPS CERTIFICATE (self-signed, for the phone camera)
# ============================================================

def local_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            ips.add(ip)
    except Exception:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def phone_urls():
    return [f"https://{ip}:{HTTPS_PORT}" for ip in local_ips()]


def ensure_cert():
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert_f, key_f, ips_f = CERT_DIR / "cert.pem", CERT_DIR / "key.pem", CERT_DIR / "ips.json"
    ips = local_ips()
    try:
        if cert_f.exists() and key_f.exists() and \
                set(json.loads(ips_f.read_text())) >= set(ips):
            return str(cert_f), str(key_f)
    except Exception:
        pass

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Working Camp Clothes Counter")])
    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    san += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=3 * 365))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_f.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    cert_f.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    ips_f.write_text(json.dumps(ips))
    return str(cert_f), str(key_f)


# ============================================================
# MAIN
# ============================================================

def already_running():
    try:
        with socket.create_connection(("127.0.0.1", HTTP_PORT), timeout=1):
            return True
    except OSError:
        return False


def main():
    # Windows lets a second copy bind the same port; requests would then
    # be split between old and new servers. Run one copy only.
    if already_running():
        print(f"The server is already running — opening http://localhost:{HTTP_PORT}")
        if not os.environ.get("CLOTHES_NO_BROWSER"):
            webbrowser.open(f"http://localhost:{HTTP_PORT}")
        return

    cert, key = ensure_cert()

    http = make_server("127.0.0.1", HTTP_PORT, app, threaded=True)
    https = make_server("0.0.0.0", HTTPS_PORT, app, threaded=True, ssl_context=(cert, key))

    threading.Thread(target=https.serve_forever, daemon=True).start()

    print("=" * 56)
    print(f"  {cc.TITLE_AR} — {cc.CAMP_AR}")
    print(f"  Dashboard (this laptop):  http://localhost:{HTTP_PORT}")
    for u in phone_urls():
        print(f"  Phones (same Wi-Fi):      {u}")
    print("  Close this window to stop the server.")
    print("=" * 56, flush=True)

    if not os.environ.get("CLOTHES_NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{HTTP_PORT}")).start()
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
