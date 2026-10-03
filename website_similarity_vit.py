#!/usr/bin/env python3
"""
site_clone_detector.py - multi-signal website clone / impersonation detector.

Compares a REFERENCE (legit) site against one or more CANDIDATE sites using
several independent signals, because no single signal is reliable:

  visual   : DINOv2 embeddings (CLS + mean-patch) on above-the-fold + page tiles
  hash     : perceptual dHash of the above-the-fold screenshot
  color    : colour histogram intersection
  layout   : DOM-box occupancy grid (survives re-skinning / image swaps)
  text     : visible-text shingle Jaccard + containment (survives re-theming)
  assets   : shared asset file names (clones often copy images/js/css verbatim)
  classes  : shared CSS class vocabulary
  favicon  : favicon exact / perceptual match
Plus context (not part of the score): domain lookalike checks and credential forms.

Install:
    pip install playwright torch transformers pillow numpy tldextract
    playwright install chromium

Usage:
    python site_clone_detector.py https://real-bank.com https://r3al-bank-login.xyz
    python site_clone_detector.py REF CAND1 CAND2 CAND3 --json out.json --save-dir shots

SAFETY: run suspected phishing URLs in a sandbox/VM. This script never submits forms.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
VIEWPORT_W, VIEWPORT_H = 1440, 900
MAX_TILES = 3                      # above-the-fold + up to 2 more screens
MODEL_NAME = "facebook/dinov2-small"
EMB_W, EMB_H = 308, 196            # multiples of 14, ~16:10 aspect ratio
NAV_TIMEOUT_MS = 45_000
CHROME_PATH = None                 # e.g. r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CONCURRENCY = 3
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Calibration: raw similarity of *unrelated* pages (lo) -> 0, near-identical (hi) -> 1.
# These are sensible defaults; tune them on a labelled set of your own pairs.
CAL = {
    "visual": (0.40, 0.92),
    "hash": (0.50, 0.95),
    "color": (0.60, 1.00),
    "layout": (0.20, 0.90),
    "text": (0.00, 0.70),
    "assets": (0.00, 0.50),
    "classes": (0.10, 0.70),
}
WEIGHTS = {
    "visual": 0.25, "layout": 0.15, "text": 0.20, "hash": 0.10,
    "assets": 0.10, "favicon": 0.10, "color": 0.05, "classes": 0.05,
}
GENERIC_ASSETS = {
    "favicon.ico", "index.js", "main.js", "app.js", "bundle.js", "style.css",
    "styles.css", "main.css", "app.css", "jquery.min.js", "logo.png", "logo.svg",
}


def cal(x: float, key: str) -> float:
    lo, hi = CAL[key]
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def find_system_browser() -> str | None:
    for browser_name in ("chrome", "msedge", "chromium", "chromium-browser"):
        browser_path = shutil.which(browser_name)
        if browser_path:
            return browser_path

    program_files = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    program_files_x86 = Path(
        os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    )
    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = (
        program_files / "Google/Chrome/Application/chrome.exe",
        program_files_x86 / "Google/Chrome/Application/chrome.exe",
        local_app_data / "Google/Chrome/Application/chrome.exe",
        program_files / "Microsoft/Edge/Application/msedge.exe",
        program_files_x86 / "Microsoft/Edge/Application/msedge.exe",
        local_app_data / "Microsoft/Edge/Application/msedge.exe",
    )
    return next((str(path) for path in candidates if path.is_file()), None)


# ============================================================
# DATA
# ============================================================
@dataclass
class Site:
    url: str
    final_url: str = ""
    title: str = ""
    tiles: list = field(default_factory=list)       # list[PIL.Image]
    text: str = ""
    classes: set = field(default_factory=set)
    assets: set = field(default_factory=set)
    boxes: list = field(default_factory=list)
    forms: list = field(default_factory=list)
    has_password: bool = False
    favicon_md5: str | None = None
    favicon_hash: np.ndarray | None = None
    emb: np.ndarray | None = None
    error: str | None = None
    warnings: list = field(default_factory=list)


# ============================================================
# CAPTURE
# ============================================================
EXTRACT_JS = r"""
() => {
  const W = innerWidth;
  const visible = el => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return r.width > 4 && r.height > 4 && s.visibility !== 'hidden' &&
           s.display !== 'none' && parseFloat(s.opacity) > 0.05;
  };
  const kindOf = t => {
    if (['img','svg','video','canvas','picture'].includes(t)) return 'media';
    if (['input','select','textarea','button'].includes(t)) return 'control';
    if (t === 'a') return 'link';
    if (['p','h1','h2','h3','h4','li','span','label'].includes(t)) return 'text';
    return 'block';
  };
  const boxes = [];
  document.querySelectorAll(
    'h1,h2,h3,h4,p,span,li,a,button,input,select,textarea,img,svg,video,canvas,nav,header,footer,form,label'
  ).forEach(el => {
    if (boxes.length >= 2500 || !visible(el)) return;
    const r = el.getBoundingClientRect();
    boxes.push([kindOf(el.tagName.toLowerCase()), r.left + scrollX, r.top + scrollY, r.width, r.height]);
  });
  const classes = new Set();
  document.querySelectorAll('[class]').forEach(el => {
    if (typeof el.className === 'string')
      el.className.split(/\s+/).forEach(c => { if (c.length > 2 && classes.size < 5000) classes.add(c); });
  });
  const assets = new Set();
  document.querySelectorAll('img[src],script[src],link[rel=stylesheet][href],source[src],video[src]')
    .forEach(el => assets.add(el.src || el.href));
  let favicon = null;
  const link = document.querySelector('link[rel~="icon"]');
  favicon = link ? link.href : (location.origin + '/favicon.ico');
  return {
    title: document.title || '',
    text: (document.body ? document.body.innerText : '').slice(0, 30000),
    boxes, classes: [...classes], assets: [...assets], favicon,
    forms: [...document.forms].map(f => ({
      action: f.action || '', password: !!f.querySelector('input[type=password]')
    })),
    hasPassword: !!document.querySelector('input[type=password]'),
  };
}
"""

LAZY_SCROLL_JS = """
async (limit) => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const maxY = Math.min(document.documentElement.scrollHeight, limit);
  for (let y = 0; y < maxY; y += 500) { window.scrollTo(0, y); await sleep(80); }
  window.scrollTo(0, 0);
  await sleep(600);
}
"""

NO_ANIM_CSS = """*,*::before,*::after{animation:none!important;transition:none!important;
caret-color:transparent!important;scroll-behavior:auto!important}"""


def dhash(img: Image.Image, size: int = 16) -> np.ndarray:
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


async def capture(context, site: Site) -> Site:
    page = await context.new_page()
    try:
        try:
            await page.goto(site.url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        except Exception as e:  # keep going with whatever rendered
            site.warnings.append(f"navigation: {type(e).__name__}")
        try:
            await page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        await page.wait_for_timeout(1500)
        await page.add_style_tag(content=NO_ANIM_CSS)
        limit = MAX_TILES * VIEWPORT_H
        await page.evaluate(LAZY_SCROLL_JS, limit)

        site.final_url = page.url
        data = await page.evaluate(EXTRACT_JS)
        site.title = data["title"]
        site.text = data["text"]
        site.classes = set(data["classes"])
        site.assets = set(data["assets"])
        site.boxes = data["boxes"]
        site.forms = data["forms"]
        site.has_password = data["hasPassword"]

        total_h = await page.evaluate("document.documentElement.scrollHeight")
        height = int(min(max(total_h, VIEWPORT_H), limit))
        png = await page.screenshot(
            full_page=True, animations="disabled",
            clip={"x": 0, "y": 0, "width": VIEWPORT_W, "height": height},
        )
        full = Image.open(io.BytesIO(png)).convert("RGB")
        for i in range(MAX_TILES):
            top = i * VIEWPORT_H
            if top >= full.height or (i > 0 and full.height - top < VIEWPORT_H // 2):
                break
            tile = full.crop((0, top, full.width, min(top + VIEWPORT_H, full.height)))
            if tile.height < VIEWPORT_H:
                canvas = Image.new("RGB", (VIEWPORT_W, VIEWPORT_H), "white")
                canvas.paste(tile, (0, 0))
                tile = canvas
            site.tiles.append(tile)

        if np.asarray(site.tiles[0].convert("L")).std() < 3:
            site.warnings.append("above-the-fold looks blank (blocked, cloaked or still loading?)")

        # favicon
        try:
            resp = await context.request.get(data["favicon"], timeout=8000)
            if resp.ok:
                body = await resp.body()
                if body:
                    site.favicon_md5 = hashlib.md5(body).hexdigest()
                    site.favicon_hash = dhash(Image.open(io.BytesIO(body)).convert("RGBA")
                                              .convert("RGB"), 16)
        except Exception:
            pass
    except Exception as e:
        site.error = f"{type(e).__name__}: {e}"
    finally:
        await page.close()
    return site


async def capture_all(sites: list[Site]) -> None:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        opts: dict[str, str | bool] = {"headless": True}
        if CHROME_PATH:
            if not Path(CHROME_PATH).is_file():
                raise FileNotFoundError(f"Chrome executable not found: {CHROME_PATH}")
            opts["executable_path"] = CHROME_PATH
        elif not Path(p.chromium.executable_path).is_file():
            system_browser = find_system_browser()
            if not system_browser:
                raise RuntimeError(
                    "Playwright's Chromium browser is not installed and no system "
                    "Chrome or Edge browser was found. Install Chromium with:\n"
                    f'  "{sys.executable}" -m playwright install chromium\n'
                    "Alternatively, pass --chrome-path with an installed "
                    "Chrome or Edge executable."
                )
            print(
                "Playwright Chromium is not installed; using system browser: "
                f"{system_browser}",
                file=sys.stderr,
            )
            opts["executable_path"] = system_browser
        browser = await p.chromium.launch(**opts)
        sem = asyncio.Semaphore(CONCURRENCY)

        async def run(site: Site):
            ctx = await browser.new_context(
                viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
                device_scale_factor=1, ignore_https_errors=True,
                user_agent=USER_AGENT, locale="en-US", accept_downloads=False,
            )
            async with sem:
                print(f"Capturing {site.url}", file=sys.stderr)
                await capture(ctx, site)
            await ctx.close()

        await asyncio.gather(*(run(s) for s in sites))
        await browser.close()


# ============================================================
# SIGNALS
# ============================================================
class Embedder:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {name} on {self.device}...", file=sys.stderr)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        torch = self.torch
        arr = np.stack([
            (np.asarray(im.convert("RGB").resize((EMB_W, EMB_H), Image.BICUBIC),
                        dtype=np.float32) / 255.0 - self.mean) / self.std
            for im in images
        ]).transpose(0, 3, 1, 2)
        x = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            h = self.model(pixel_values=x).last_hidden_state
        F = torch.nn.functional
        cls = F.normalize(h[:, 0], dim=-1)               # global semantics
        patch = F.normalize(h[:, 1:].mean(dim=1), dim=-1)  # layout/texture
        return (torch.cat([cls, patch], dim=-1) / math.sqrt(2)).cpu().numpy()


def sig_visual(a: Site, b: Site):
    sims = a.emb @ b.emb.T
    first = float(sims[0, 0])
    best = 0.5 * (sims.max(axis=1).mean() + sims.max(axis=0).mean())
    raw = 0.6 * first + 0.4 * float(best)
    return cal(raw, "visual"), f"cos(top)={first:.3f} cos(tiles)={best:.3f}"


def sig_hash(a: Site, b: Site):
    sim = 1.0 - float(np.mean(dhash(a.tiles[0]) ^ dhash(b.tiles[0])))
    return cal(sim, "hash"), f"dHash sim={sim:.3f}"


def _hist(img: Image.Image) -> np.ndarray:
    q = (np.asarray(img.convert("RGB").resize((160, 100)), dtype=np.int32) // 64).reshape(-1, 3)
    idx = q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]
    h = np.bincount(idx, minlength=64).astype(np.float32)
    return h / h.sum()


def sig_color(a: Site, b: Site):
    inter = float(np.minimum(_hist(a.tiles[0]), _hist(b.tiles[0])).sum())
    return cal(inter, "color"), f"hist intersection={inter:.3f}"


KINDS = ["text", "media", "control", "link", "block"]
GRID_W, GRID_H = 24, 15


def layout_vector(boxes) -> np.ndarray:
    grid = np.zeros((len(KINDS), GRID_H, GRID_W), np.float32)
    cw, ch = VIEWPORT_W / GRID_W, VIEWPORT_H / GRID_H
    for kind, x, y, w, h in boxes:
        if y >= VIEWPORT_H or x >= VIEWPORT_W or w * h > 0.4 * VIEWPORT_W * VIEWPORT_H:
            continue
        x0, x1 = int(max(x, 0) // cw), int(min(x + w, VIEWPORT_W - 1) // cw)
        y0, y1 = int(max(y, 0) // ch), int(min(y + h, VIEWPORT_H - 1) // ch)
        grid[KINDS.index(kind), y0:y1 + 1, x0:x1 + 1] = 1
    pad = np.pad(grid, ((0, 0), (1, 1), (1, 1)))  # blur: tolerate small shifts
    blur = sum(pad[:, dy:dy + GRID_H, dx:dx + GRID_W] for dy in range(3) for dx in range(3)) / 9
    return blur.flatten()


def sig_layout(a: Site, b: Site):
    va, vb = layout_vector(a.boxes), layout_vector(b.boxes)
    if va.std() < 1e-6 or vb.std() < 1e-6:
        return None
    r = float(np.corrcoef(va, vb)[0, 1])
    return cal(r, "layout"), f"DOM layout correlation={r:.3f}"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\u00c0-\uffff]+", text.lower())


def sig_text(a: Site, b: Site):
    wa, wb = _words(a.text), _words(b.text)
    if len(wa) < 20 or len(wb) < 20:
        return None  # image-only pages: rely on visual signals
    sh = lambda w: {" ".join(w[i:i + 3]) for i in range(len(w) - 2)}
    sa, sb = sh(wa), sh(wb)
    inter = len(sa & sb)
    jac = inter / max(len(sa | sb), 1)
    cont = inter / max(min(len(sa), len(sb)), 1)  # clone of a subset still counts
    raw = max(jac, 0.8 * cont)
    return cal(raw, "text"), f"jaccard={jac:.3f} containment={cont:.3f}"


def _basenames(urls) -> set[str]:
    out = set()
    for u in urls:
        name = urlparse(u).path.rsplit("/", 1)[-1].lower()
        if len(name) >= 5 and name not in GENERIC_ASSETS and not name.startswith("data:"):
            out.add(name)
    return out


def sig_assets(a: Site, b: Site):
    aa, ab = _basenames(a.assets), _basenames(b.assets)
    if len(aa) < 3 or len(ab) < 3:
        return None
    inter = aa & ab
    raw = len(inter) / len(aa | ab)
    return cal(raw, "assets"), f"{len(inter)} shared asset file names (jaccard={raw:.3f})"


def sig_classes(a: Site, b: Site):
    if len(a.classes) < 15 or len(b.classes) < 15:
        return None
    raw = len(a.classes & b.classes) / len(a.classes | b.classes)
    return cal(raw, "classes"), f"class jaccard={raw:.3f} (utility CSS like Tailwind inflates this)"


def sig_favicon(a: Site, b: Site):
    if a.favicon_hash is None or b.favicon_hash is None:
        return None
    if a.favicon_md5 == b.favicon_md5:
        return 1.0, "byte-identical favicon"
    sim = 1.0 - float(np.mean(a.favicon_hash ^ b.favicon_hash))
    return float(np.clip((sim - 0.6) / 0.4, 0, 1)), f"favicon dHash sim={sim:.3f}"


SIGNALS = {
    "visual": sig_visual, "layout": sig_layout, "text": sig_text, "hash": sig_hash,
    "assets": sig_assets, "favicon": sig_favicon, "color": sig_color, "classes": sig_classes,
}


# ============================================================
# DOMAIN CONTEXT
# ============================================================
def registered_domain(host: str) -> str:
    try:
        import tldextract
        e = tldextract.extract(host)
        rd = getattr(e, "top_domain_under_public_suffix", None) or e.registered_domain
        if rd:
            return rd.lower()
    except Exception:
        pass
    parts = host.lower().split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "gov", "ac"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _deleet(s: str) -> str:
    for k, v in {"0": "o", "1": "l", "3": "e", "5": "s", "vv": "w", "rn": "m"}.items():
        s = s.replace(k, v)
    return s


def domain_context(ref: Site, cand: Site) -> dict:
    hr = urlparse(ref.final_url or ref.url).hostname or ""
    hc = urlparse(cand.final_url or cand.url).hostname or ""
    rr, rc = registered_domain(hr), registered_domain(hc)
    lr, lc = rr.split(".")[0], rc.split(".")[0]
    ratio = SequenceMatcher(None, _deleet(lr), _deleet(lc)).ratio()
    notes = []
    same = rr == rc
    if not same:
        if lr and lr in hc:
            notes.append(f"candidate host contains reference brand '{lr}'")
        if ratio >= 0.8:
            notes.append(f"registered-domain label lookalike (similarity {ratio:.2f})")
        if "xn--" in hc:
            notes.append("punycode / IDN host (possible homograph)")
    return {"ref_domain": rr, "cand_domain": rc, "same_site": same, "lookalike_notes": notes}


# ============================================================
# COMPARE
# ============================================================
def compare(ref: Site, cand: Site) -> dict:
    signals = {}
    for name, fn in SIGNALS.items():
        res = fn(ref, cand)
        if res is None:
            continue
        score, detail = res
        signals[name] = {"score": float(score), "weight": WEIGHTS[name], "detail": detail}

    wsum = sum(s["weight"] for s in signals.values())
    wavg = sum(s["score"] * s["weight"] for s in signals.values()) / wsum
    top3 = float(np.mean(sorted(s["score"] for s in signals.values())[-3:]))
    # blend: weighted average, plus credit for strong independent evidence
    final = 100 * (0.75 * wavg + 0.25 * top3)

    ctx = domain_context(ref, cand)
    if final >= 75:
        verdict = "VERY LIKELY CLONE"
    elif final >= 55:
        verdict = "LIKELY CLONE / COPY"
    elif final >= 35:
        verdict = "PARTIAL SIMILARITY"
    else:
        verdict = "DIFFERENT"

    flags = []
    if ctx["same_site"]:
        flags.append("same registered domain: likely the same organisation, not impersonation")
    elif final >= 55:
        flags.append("HIGH similarity on a DIFFERENT registered domain: possible impersonation")
        if cand.has_password:
            flags.append("candidate has a password field")
        for f in cand.forms:
            fh = urlparse(f["action"]).hostname if f["action"] else None
            if f["password"] and fh and registered_domain(fh) != ctx["cand_domain"]:
                flags.append(f"login form posts to a third-party host: {fh}")
        flags += ctx["lookalike_notes"]

    return {
        "reference": ref.final_url or ref.url, "candidate": cand.final_url or cand.url,
        "score": round(final, 1), "verdict": verdict, "signals": signals,
        "domain": ctx, "flags": flags, "warnings": cand.warnings,
    }


def print_report(r: dict) -> None:
    print("=" * 72)
    print(f"Reference : {r['reference']}")
    print(f"Candidate : {r['candidate']}")
    print("-" * 72)
    for name, s in sorted(r["signals"].items(), key=lambda kv: -kv[1]["weight"]):
        print(f"  {name:<8} {s['score'] * 100:5.1f}   w={s['weight']:.2f}   {s['detail']}")
    print("-" * 72)
    print(f"  SCORE: {r['score']:.1f}/100   VERDICT: {r['verdict']}")
    for f in r["flags"]:
        print(f"  ! {f}")
    for w in r["warnings"]:
        print(f"  ~ {w}")


# ============================================================
# MAIN
# ============================================================
def main() -> int:
    global MODEL_NAME, CHROME_PATH
    ap = argparse.ArgumentParser(description="Website clone / impersonation detector")
    ap.add_argument("reference", help="URL of the legitimate site")
    ap.add_argument("candidates", nargs="+", help="URL(s) to test against the reference")
    ap.add_argument("--model", default=MODEL_NAME, help="HF vision model (DINOv2 recommended)")
    ap.add_argument("--chrome-path", default=CHROME_PATH)
    ap.add_argument("--json", help="write full results to this JSON file")
    ap.add_argument("--save-dir", help="save screenshots here for manual review")
    args = ap.parse_args()
    MODEL_NAME, CHROME_PATH = args.model, args.chrome_path

    sites = [Site(u) for u in [args.reference, *args.candidates]]
    asyncio.run(capture_all(sites))

    ref, cands = sites[0], sites[1:]
    if ref.error or not ref.tiles:
        print(f"Could not capture reference: {ref.error}", file=sys.stderr)
        return 2

    ok = [s for s in sites if s.tiles]
    embedder = Embedder(MODEL_NAME)
    for s in ok:
        s.emb = embedder.embed(s.tiles)

    if args.save_dir:
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(ok):
            for j, t in enumerate(s.tiles):
                t.save(out / f"site{i}_tile{j}.png")

    results = []
    for c in cands:
        if c.error or not c.tiles:
            print(f"\nSkipping {c.url}: {c.error or 'no screenshot'}")
            continue
        r = compare(ref, c)
        results.append(r)
        print_report(r)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())