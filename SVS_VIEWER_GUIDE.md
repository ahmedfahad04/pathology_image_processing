# SVS Viewer Guide — TCGA-FJ-A871 Whole Slide Image

View the `1.1 GB` SVS slide `TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs` in your browser without loading the full `38 GB` uncompressed image into RAM.

> Stack: `episphere/svs` + `OpenSeadragon 4.1.1` + `GeoTIFFTileSource-JPEG2k` + `RangeHTTPServer`

---

## 1. Quick Start — View the Local Image

### Prerequisites

- Python `3.8+` (tested on `3.12`)
- Modern browser (Chrome/Firefox)
- `1.1 GB` SVS file at `Cell_Count/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs`

### Setup

```bash
# clone the viewer (optional — already at /tmp/svs)
git clone https://github.com/episphere/svs.git /tmp/svs

# install Python deps (only needed for efficient viewing + inspection)
pip install RangeHTTPServer tifffile imagecodecs Pillow
```

> Why `RangeHTTPServer`? Stock `python -m http.server` does **not** support `HTTP Range` (`206 Partial Content`). The viewer fetches `~50 KB` tiles on demand — without Range it would try to download the full `1.1 GB`.

### Run the Server

```bash
cd "/home/fahad/Documents/PROJECTS/Cell_Count"

# foreground (Ctrl+C to stop)
python3 -m RangeHTTPServer 8765

# or background
nohup python3 -m RangeHTTPServer 8765 > /tmp/svs_server.log 2>&1 &
echo "http://localhost:8765/viewer.html"
```

### Open the Viewer

Open in browser:

- **Main viewer (local image):** http://localhost:8765/viewer.html
- **Thumbnail:** http://localhost:8765/thumbnail.jpg
- **Raw file:** http://localhost:8765/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs

Controls: **scroll = zoom**, **drag = pan**, **F12 → Network** to see tile requests.

### Stop / Change Port

```bash
ss -tlnp | grep 8765        # find PID
kill <PID>                  # e.g. kill 13292

# different port
python3 -m RangeHTTPServer 8000
# then open http://localhost:8000/viewer.html
```

---

## 2. What Files Are Involved

| File | Purpose |
|------|---------|
| `viewer.html` | Local viewer — loads `./TCGA-...svs` via `OpenSeadragon.GeoTIFFTileSource.getAllTileSources(fileUrl)` |
| `TCGA-...svs` | Your local slide (`1.1 GB` compressed, `38 GB` uncompressed) |
| `/tmp/svs/index.html` | Original `episphere/svs` demo — loads **cloud** images from `https://storage.googleapis.com/imagebox_test` (`svs.js:26`) + `mapping.json` |
| `thumbnail.jpg` | Extracted `1024×648` preview (generated via `tifffile`) |

> `viewer.html` = local. `episphere/svs` demo = cloud. They share the same `OpenSeadragon` + `GeoTIFFTileSource` logic (`svs.js:171-182`).

---

## 3. How This SVS Is Constructed

### Raw numbers (from `tifffile`, tag `ImageDescription:270`)

```
Aperio Image Library v12.0.15
144272x89685 [0,100 141432x89585] (240x240) JPEG/RGB Q=30
|AppMag = 40|MPP = 0.2525|ScanScope ID = SS1763
```

- **Full resolution (Level 0):** `141432 × 89585 × 3` px = `12.7 gigapixels`
  - Uncompressed: `141432 * 89585 * 3 ≈ 38 GB`
  - Compressed (JPEG `Q=30`, `Compression:259=7`): `1.1 GB` on disk
  - Physical size: `35.7 mm × 22.6 mm` (`MPP 0.2525 µm/px` → `~3960 dpi`, `40×` magnification)
- **Pyramid levels** (downsampled copies in same file) — all tiled `240×240` px (`TileWidth:322=240`, `TileLength:323=240`, `ImageDescription:270` `(240x240)`):

  | Page | Level | Dimensions (W×H) | Downsample | Tile size | Grid (cols×rows) | Tiles | `tif.series` |
  |------|-------|------------------|------------|-----------|------------------|-------|--------------|
  | 0 | 0 (full) | `141432 × 89585` | 1× | `240×240` | `590×374` | `220,660` | `series 0, level 0` |
  | 2 | 1 | `35358 × 22396` | 4× | `240×240` | `148×94` | `13,912` | `series 0, level 1` |
  | 3 | 2 | `8839 × 5599` | 16× | `240×240` | `37×24` | `888` | `series 0, level 2` |
  | 4 | 3 (overview) | `2209 × 1399` | 64× | `240×240` | `10×6` | `60` | `series 0, level 3` |
  | 1 | thumbnail | `1024 × 648` | — | **not tiled** (`RowsPerStrip:278=16`, stripped JPEG) | — | `1` strip | `series 1, level 0` |

  Total tiled pyramid: `~235,520` JPEG tiles. Other SVS files may use `256×256` or `512×512` — scanner-dependent (`Aperio SS1763` → `240` here).
- No `XResolution:282`/`YResolution:283` — scale is in `MPP` inside `ImageDescription`.

### Parts of an SVS (TIFF) file

```
[ TIFF Header ]
[ IFD 0: Level 0 ] → 141432×89585, 240×240 tiles, 220660 tiles (TileOffsets:324, TileByteCounts:325)
[ IFD 1: Thumbnail ] → 1024×648, stripped (RowsPerStrip:278=16), not tiled
[ IFD 2: Level 1 ] → 35358×22396, 240×240 tiles, 13912 tiles
[ IFD 3: Level 2 ] → 8839×5599, 240×240 tiles, 888 tiles
[ IFD 4: Level 3 ] → 2209×1399, 240×240 tiles, 60 tiles
[ JPEG tile data ........... 1.1 GB ........... ]
```

Each `IFD` is a directory describing one image level. Tiled levels store `TileWidth:322`/`TileLength:323` + `TileOffsets:324`/`TileByteCounts:325`; the thumbnail is stripped. Tiles are JPEG-compressed individually so any tile can be decoded alone. Verify with: `python3 -c "import tifffile; p=tifffile.TiffFile('x.svs').pages[0]; print(p.tags[322].value, p.tags[323].value, len(p.tags[324].value))"`

---

## 4. How Navigation Works (Why No 15 GB RAM Needed)

Think **Google Maps** for pathology.

1. **Parse header only:** Browser does `Range: bytes=0-16383` → reads TIFF header + IFDs, learns where every tile lives. No pixels yet.
2. **Show overview:** Render Level 3 (`2209×1399`) — entire slide fits screen, `<5 MB` RAM.
3. **On zoom/pan:** Compute viewport → request **only visible tiles** at needed level:
   ```
   GET /TCGA-...svs  Range: bytes=1048576-1123456 → 206 Partial Content (~50 KB)
   ```
   Decode that one `240×240` JPEG tile → draw. A typical view = `10-20` tiles (`2-5 MB` RAM).
4. **Never loads full `141k×89k` bitmap.** That would be `38 GB`.

Verify: open `viewer.html` → `F12 → Network` → filter `svs` → you see dozens of `206` requests of `20-80 KB`, not one `1.1 GB` download.

```bash
# proof Range works
curl -H "Range: bytes=0-1023" http://localhost:8765/TCGA-...svs -o /tmp/x.bin
ls -lh /tmp/x.bin  # 1.0K with RangeHTTPServer, 1.1G with stock http.server
```

### Do you see the FULL image?

Yes — the canvas always represents the **full `35.7×22.6 mm` slide extent**. At first you see it at low-res (Level 3). Zoom in and the viewer swaps to Level 0 tiles for that region, so you see cells at `40×` (`0.25 µm/px`) but only for the area on screen.

---

## 5. Inspect Without Browser

```bash
python3 -c "
import tifffile
with tifffile.TiffFile('TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs') as tif:
    for i,p in enumerate(tif.pages):
        print(i, p.shape, p.compression)
"

# extract thumbnail
python3 -c "
import tifffile; from PIL import Image
with tifffile.TiffFile('TCGA-...svs') as tif:
    Image.fromarray(tif.pages[1].asarray()).save('thumbnail.jpg')
"
```

---

## 6. Troubleshooting

| Problem | Fix |
|---------|-----|
| `curl Range` returns `200` + `1.1 GB` | You're using `http.server` — switch to `RangeHTTPServer` |
| Black viewer / `GeoTIFFTileSource not found` | Wait 2s, hard-refresh `Ctrl+Shift+R`, check CDN online |
| `Address already in use` | `ss -tlnp \| grep 8765` → `kill <PID>` or use `8766` |
| `imagecodecs` error on thumbnail | `pip install imagecodecs` |

---

*Generated for `Cell_Count` — local viewer at `viewer.html` serves the TCGA slide via `http://localhost:8765/viewer.html`.*
