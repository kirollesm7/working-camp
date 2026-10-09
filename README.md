# Working Camp 2026 — counters

Two apps for **معسكر العمل 2026 (منه وله)**:

- **Clothes counter** — phones scan the sticker on each box; the laptop reads the
  ticks and the handwritten numbers (offline) and keeps the totals.
- **Carton counter** — counts cartons from a laser sensor on a TTL-to-USB adapter (COM11).

## Run it on a new laptop (Windows)

1. Install **Python 3** from <https://www.python.org/downloads/> —
   in the installer tick **"Add python.exe to PATH"**.
2. Download this project (green **Code** button → **Download ZIP**) and unzip it.
3. Double-click **`Run_Clothes.bat`**.
   - The first time it installs the libraries and prepares the reader.
     This needs internet and takes **a few minutes** — wait until the black window
     shows the links. `localhost` only opens after that.
   - If Windows asks about the firewall, choose **Allow** on **Private networks**
     (otherwise the phones can't connect).
4. Open the links shown in the black window:
   - Dashboard on the laptop: `http://localhost:5000`
   - Phones (same Wi-Fi): `https://<laptop-ip>:5443` — accept the security warning once
     (**Advanced → Proceed**) and allow the camera.
   - Big screen: `Clothes_Display.bat` or `http://localhost:5000/display`

Keep the black window open while working — closing it stops the server.

**Carton counter:** double-click **`Run.bat`**.

## Notes

- Saved boxes, photos, learned handwriting and the HTTPS key live in `data/`,
  which is **not** in git. Copy the `data/` folder too if you want to move the counts
  and the learned handwriting to another laptop.
- Train the CNN reader (optional): `python app/train_cnn.py`
