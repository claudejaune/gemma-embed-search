#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["pillow>=10"]
# ///
"""Index a folder of images, audio, or text with a multimodal embedding model and search it.

All modalities land in one vector space, so a text query retrieves matching images,
audio clips, or documents. Works with any OpenAI-compatible /v1/embeddings server that
accepts multimodal content parts (for example llama.cpp's llama-server running
EmbeddingGemma 2 or another multimodal embedding model).

Usage:
  embed_search.py doctor    [--url URL] [--model NAME] [--api-key KEY]
  embed_search.py init      --url URL --model NAME [--api-key KEY] [--root DIR]
  embed_search.py index     --root DIR [--modality auto] [--limit N] [--force] [--compact]
  embed_search.py search    "query text" [--also P] [--must P] [--color yellow]
                            [--min-color 0.03] [--top 20] [--json] [--html out.html]
  embed_search.py stats     [--root DIR]

Options (all subcommands):
  --url URL        Embeddings server base URL (default http://127.0.0.1:8080)
  --model NAME     Model name the server routes to (see `doctor` for the list)
  --api-key KEY    Bearer token; prefer the EMBED_SEARCH_API_KEY environment variable
  --root DIR       Folder to index or search (also settable with `init --root`)
  --index FILE     Index file (default ~/.cache/gemma-embed-search/<root-hash>.jsonl)
  --config FILE    Config file (default ~/.config/gemma-embed-search/config.json)
  --timeout SECS   Per-request timeout (default 300)

Precedence for url/model/api-key/root/index: flag > environment
(EMBED_SEARCH_URL, EMBED_SEARCH_MODEL, EMBED_SEARCH_API_KEY, EMBED_SEARCH_ROOT) > config file.

Exit codes:
  0 success            2 usage error           3 server unreachable or rejected auth
  4 index missing      5 nothing to do         6 server error mid-run

Diagnostics go to stderr; `--json` output goes to stdout so it pipes into jq.
"""

import argparse
import base64
import colorsys
import fnmatch
import hashlib
import io
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request

VERSION = "1.0.0"
DEFAULT_URL = "http://127.0.0.1:8080"
CONFIG_PATH = os.path.expanduser("~/.config/gemma-embed-search/config.json")
CACHE_DIR = os.path.expanduser("~/.cache/gemma-embed-search")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TEXT_EXTS = {".txt", ".md", ".rst", ".csv", ".json", ".log", ".py", ".js", ".ts", ".c",
             ".cpp", ".h", ".java", ".go", ".rs", ".html", ".xml", ".yaml", ".yml"}

# Hue ranges in degrees. Bins in the stored histogram are 30 degrees wide.
COLOR_RANGES = {
    "red": [(345, 360), (0, 15)],
    "orange": [(15, 42)],
    "yellow": [(42, 70)],
    "green": [(70, 160)],
    "cyan": [(160, 200)],
    "blue": [(200, 260)],
    "purple": [(260, 300)],
    "pink": [(300, 345)],
}

QUERY_PREFIX = "task: search result | query: {q}"
DOC_PREFIX = "title: none | text: {t}"


def die(code, msg):
    print(f"Error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def log(msg):
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------- configuration

def load_config(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        die(2, f"cannot read config {path}: {exc}. Fix or delete it, or pass --config PATH.")


def resolve(args, cfg, key, env=None, default=None):
    """flag > environment > config file > default"""
    val = getattr(args, key.replace("-", "_"), None)
    if val is not None:
        return val
    if env and os.environ.get(env):
        return os.environ[env]
    if key in cfg and cfg[key] not in (None, ""):
        return cfg[key]
    return default


def index_path_for(root):
    digest = hashlib.sha256(os.path.abspath(root).encode()).hexdigest()[:12]
    name = "".join(c if c.isalnum() else "-" for c in os.path.basename(os.path.abspath(root)))
    return os.path.join(CACHE_DIR, f"{name or 'root'}-{digest}.jsonl")


def resolve_paths(args, cfg, need_root=False, need_index=True):
    root = resolve(args, cfg, "root", "EMBED_SEARCH_ROOT")
    idx = resolve(args, cfg, "index")
    if not idx:
        if not root:
            if need_root:
                die(2, "--root is required (or run `init --root DIR` to remember it).")
            die(2, "--index is required (or pass --root to derive it).")
        idx = index_path_for(root)
    return root, idx


# ------------------------------------------------------------------- HTTP layer

class Client:
    def __init__(self, url, model, api_key, timeout):
        self.url = url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def post(self, path, payload):
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
            method="POST",
        )
        return self._send(req)

    def get(self, path):
        req = urllib.request.Request(
            f"{self.url}{path}",
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        return self._send(req, json_body=False)

    def _send(self, req, json_body=True):
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in (401, 403):
                die(3, f"server rejected the credentials (HTTP {exc.code}). Pass --api-key or set "
                       f"EMBED_SEARCH_API_KEY. Server said: {body}")
            if exc.code in (404, 400):
                die(3, f"HTTP {exc.code} from {req.full_url}. Check --model matches a model the "
                       f"server knows (run `doctor`). Server said: {body}")
            die(3, f"HTTP {exc.code} from {req.full_url}. Server said: {body}")
        except urllib.error.URLError as exc:
            die(3, f"cannot reach {self.url} ({exc.reason}). Start your embedding server, or pass "
                   f"--url. See references/setup.md in this skill.")
        except TimeoutError:
            die(3, f"timed out after {self.timeout}s talking to {self.url}. Raise --timeout or "
                   f"reduce --batch.")
        return json.loads(raw) if json_body else raw

    def embed(self, items, batch=16):
        """Embed a list of items (str, or {'content': [...]}). Returns list of vectors."""
        out = []
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            out.extend(self._embed_chunk(chunk))
        return out

    def _embed_chunk(self, chunk):
        payload = {"model": self.model, "input": chunk, "encoding_format": "float"}
        data = self.post("/v1/embeddings", payload)
        rows = data.get("data")
        if not isinstance(rows, list):
            die(6, f"unexpected response, expected a 'data' list. Got: {str(data)[:200]}")
        rows = sorted(rows, key=lambda r: r.get("index", 0))
        return [r["embedding"] for r in rows]

    def embed_one(self, item):
        return self._embed_chunk([item])[0]


# ------------------------------------------------------------- media processing

def image_item(path, resize):
    """Re-encode to JPEG so the server never sees an unsupported container (WebP, 16-bit TIFF)."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if resize and max(im.size) > resize:
            im.thumbnail((resize, resize))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return {"content": [{"type": "image_url",
                         "image_url": {"url": "data:image/jpeg;base64,"
                                              + base64.b64encode(buf.getvalue()).decode()}}]}


def audio_item(path):
    with open(path, "rb") as fh:
        return {"content": [{"type": "input_audio",
                             "input_audio": {"data": base64.b64encode(fh.read()).decode()}}]}


def text_item(path, max_chars):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            body = fh.read(max_chars)
    except OSError as exc:
        raise ValueError(f"cannot read: {exc}")
    if not body.strip():
        raise ValueError("file is empty")
    return DOC_PREFIX.format(t=body)


def color_stats(path, bins=12):
    """Hue histogram plus lightness buckets, from a 64x64 thumbnail (~5 ms)."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((64, 64))
        raw = im.tobytes()
    hist = [0.0] * bins
    gray = dark = bright = 0
    total = max(len(raw) // 3, 1)
    for i in range(0, total * 3, 3):
        h, s, v = colorsys.rgb_to_hsv(raw[i] / 255, raw[i + 1] / 255, raw[i + 2] / 255)
        if v < 0.20:
            dark += 1
        elif v > 0.85 and s < 0.20:
            bright += 1
        elif s < 0.20:
            gray += 1
        else:
            hist[min(int(h * bins), bins - 1)] += 1
    return {"hue": [round(x / total, 4) for x in hist],
            "gray": round(gray / total, 4),
            "dark": round(dark / total, 4),
            "bright": round(bright / total, 4)}


def hue_ratio(rec, color):
    """Fraction of pixels inside a named color's hue range, from the stored histogram."""
    ranges = COLOR_RANGES.get(color)
    if not ranges:
        die(2, f"--color must be one of: {', '.join(sorted(COLOR_RANGES))}. Received: \"{color}\"")
    hist = rec.get("hue") or []
    if not hist:
        return 0.0
    bins = len(hist)
    width = 360.0 / bins
    total = 0.0
    for lo, hi in ranges:
        for i, val in enumerate(hist):
            a, b = i * width, (i + 1) * width
            overlap = max(0.0, min(b, hi) - max(a, lo))
            total += val * (overlap / width)
    return total


def modality_of(path, mode):
    ext = os.path.splitext(path)[1].lower()
    if mode != "auto":
        return mode if ext in {"image": IMAGE_EXTS, "audio": AUDIO_EXTS, "text": TEXT_EXTS}[mode] else None
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in TEXT_EXTS:
        return "text"
    return None


# ------------------------------------------------------------------- index file

def pack(vec):
    return base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode()


def unpack(blob):
    raw = base64.b64decode(blob)
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def read_index(path):
    """Returns (meta, records). Records are dicts; vectors stay packed until needed."""
    try:
        fh = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        die(4, f"no index at {path}. Run: embed_search.py index --root DIR")
    meta, recs = {}, []
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("type") == "meta":
                meta = rec
            else:
                recs.append(rec)
    return meta, recs


def load_existing(path):
    try:
        _, recs = read_index(path)
        return {r["p"]: r for r in recs}
    except SystemExit as exc:
        if exc.code != 4:
            raise
        return {}


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def truncate(vec, dims):
    if not dims or dims >= len(vec):
        return vec
    head = vec[:dims]
    n = math.sqrt(sum(x * x for x in head))
    return [x / n for x in head] if n else head


# ------------------------------------------------------------------- subcommands

def cmd_doctor(args, cfg, client):
    out = {"url": client.url, "model": client.model, "auth": bool(client.api_key)}
    try:
        raw = client.get("/models")
        data = json.loads(raw).get("data", [])
        out["reachable"] = True
        out["models"] = [m.get("id") for m in data if isinstance(m, dict)]
    except SystemExit as exc:
        if exc.code != 3:
            raise
        out["reachable"] = False
        out["error"] = "unreachable or rejected auth"
        print(json.dumps(out, indent=2))
        raise SystemExit(3)
    try:
        vec = client.embed_one(QUERY_PREFIX.format(q="a red square"))
        out["text_embedding"] = {"ok": True, "dim": len(vec)}
    except SystemExit as exc:
        out["text_embedding"] = {"ok": False, "error": "server rejected the request"}
        log(f"  note: text embedding failed (exit {exc.code})")
    print(json.dumps(out, indent=2))
    if client.model not in out["models"] and out["models"]:
        log(f"\nNote: '{client.model}' is not in the server's model list. "
            f"Pick one of: {', '.join(out['models'])}")
    return 0


def cmd_init(args, cfg, client):
    path = args.config or CONFIG_PATH
    data = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    if args.url:
        data["url"] = args.url
    if args.model:
        data["model"] = args.model
    if args.api_key:
        data["api_key"] = args.api_key
    if args.root:
        data["root"] = os.path.abspath(args.root)
    if args.index:
        data["index"] = os.path.abspath(args.index)
    if not data.get("url") or not data.get("model"):
        die(2, "init needs at least --url and --model (plus --api-key if the server needs one).")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.chmod(path, 0o600)
    log(f"wrote {path}")
    print(json.dumps({"config": path, "url": data["url"], "model": data["model"],
                      "api_key": bool(data.get("api_key")), "root": data.get("root")}, indent=2))
    return 0


def iter_files(root, include, exclude_dirs):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".")]
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = os.path.join(dirpath, name)
            if include:
                rel = os.path.relpath(path, root)
                if not any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat)
                           for pat in include):
                    continue
            yield path


def cmd_index(args, cfg, client):
    root, idx = resolve_paths(args, cfg, need_root=True)
    if not root:
        die(2, "index needs --root DIR (the folder to embed).")
    if not os.path.isdir(root):
        die(2, f"--root must be an existing directory. Received: {root}")
    os.makedirs(os.path.dirname(idx) or ".", exist_ok=True)

    existing = {} if args.force else load_existing(idx)
    exclude = set(args.exclude_dir.split(",")) if args.exclude_dir else set()
    include = args.include.split(",") if args.include else None

    todo = []
    for path in iter_files(root, include, exclude):
        mod = modality_of(path, args.modality)
        if not mod:
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        key = f"{st.st_mtime_ns}-{st.st_size}"
        if existing.get(path, {}).get("k") == key:
            continue
        todo.append((path, mod, key))

    if args.limit:
        todo = todo[: args.limit]

    if args.compact:
        kept = [r for r in existing.values() if os.path.exists(r["p"])]
        with open(idx, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "meta", "model": client.model, "ts": int(time.time())}) + "\n")
            for rec in kept:
                fh.write(json.dumps(rec) + "\n")
        log(f"compacted {idx}: kept {len(kept)} of {len(existing)} entries")
        if not todo:
            print(json.dumps({"indexed": 0, "compacted_to": len(kept), "index": idx}))
            return 0

    if not todo:
        log(f"nothing to do: {len(existing)} entries already current in {idx}")
        print(json.dumps({"indexed": 0, "total": len(existing), "index": idx}))
        return 5 if not existing else 0

    mode = "a" if existing and not args.force else "w"
    fh = open(idx, mode, encoding="utf-8")
    if mode == "w":
        fh.write(json.dumps({"type": "meta", "model": client.model, "ts": int(time.time())}) + "\n")

    done = errors = 0
    started = time.time()
    try:
        for start in range(0, len(todo), args.batch):
            group = todo[start:start + args.batch]
            items, kept = [], []
            for path, mod, key in group:
                try:
                    if mod == "image":
                        items.append(image_item(path, args.resize))
                    elif mod == "audio":
                        items.append(audio_item(path))
                    else:
                        items.append(text_item(path, args.max_chars))
                    kept.append((path, mod, key))
                except Exception as exc:  # unreadable or corrupt media: log and move on
                    errors += 1
                    log(f"  skip {path}: {exc}")
            if not items:
                continue
            try:
                vectors = client.embed(items, batch=args.batch)
            except SystemExit:
                raise
            for (path, mod, key), vec in zip(kept, vectors):
                rec = {"p": path, "k": key, "m": mod, "dim": len(vec), "v": pack(vec)}
                if mod == "image":
                    try:
                        rec.update(color_stats(path))
                    except Exception as exc:
                        log(f"  note: no color stats for {path}: {exc}")
                fh.write(json.dumps(rec) + "\n")
                done += 1
            fh.flush()
            rate = done / max(time.time() - started, 1e-6)
            eta = (len(todo) - done) / rate if rate else 0
            log(f"  {done}/{len(todo)}  {rate:.1f} files/s  eta {eta:.0f}s")
    finally:
        fh.close()

    summary = {"indexed": done, "skipped_errors": errors, "total": len(existing) + done,
               "index": idx, "seconds": round(time.time() - started, 1)}
    print(json.dumps(summary))
    return 0


def build_query(client, args):
    """Main query plus --also paraphrases averaged; --must facets scored separately."""
    main = client.embed_one(QUERY_PREFIX.format(q=args.query))
    if args.also:
        parts = [main] + [client.embed_one(QUERY_PREFIX.format(q=p)) for p in args.also]
        dim = len(main)
        merged = [sum(p[i] for p in parts) / len(parts) for i in range(dim)]
        n = math.sqrt(sum(x * x for x in merged)) or 1.0
        main = [x / n for x in merged]
    musts = [client.embed_one(QUERY_PREFIX.format(q=p)) for p in (args.must or [])]
    return main, musts


def cmd_search(args, cfg, client):
    _, idx = resolve_paths(args, cfg, need_root=False)
    _, records = read_index(idx)
    if not records:
        die(4, f"index {idx} is empty. Run: embed_search.py index --root DIR")

    qvec, musts = build_query(client, args)
    qvec = truncate(qvec, args.dims)

    results, filtered = [], 0
    for rec in records:
        if args.modality and rec.get("m") != args.modality:
            continue
        ratio = hue_ratio(rec, args.color) if args.color else None
        if args.color:
            if args.min_color is not None and ratio < args.min_color:
                filtered += 1
                continue
            if args.max_color is not None and ratio > args.max_color:
                filtered += 1
                continue
        vec = truncate(unpack(rec["v"]), args.dims)
        score = cosine(qvec, vec)
        if musts:
            score = min([score] + [cosine(m, vec) for m in musts])
        results.append((score, ratio, rec))

    results.sort(key=lambda r: r[0], reverse=True)
    page = results[args.offset: args.offset + args.top]
    rows = [{"rank": i + 1 + args.offset, "score": round(s, 4),
             **({"color": round(c, 4)} if c is not None else {}),
             "modality": r.get("m"), "path": r["p"]}
            for i, (s, c, r) in enumerate(page)]

    if args.json:
        print(json.dumps({"query": args.query, "index": idx, "total": len(results),
                          "filtered_out": filtered, "results": rows}, indent=2))
    else:
        log(f"{len(results)} matches, {filtered} filtered by color, showing {len(rows)}")
        for row in rows:
            extra = f"  {row['color']:.3f}" if "color" in row else ""
            print(f"{row['score']:.4f}{extra}  [{row['modality']}]  {row['path']}")

    if args.html:
        write_html(args.html, args.query, rows)
        log(f"wrote contact sheet: {args.html}")
    return 0


def write_html(path, query, rows):
    from PIL import Image
    cells = []
    for row in rows:
        thumb = ""
        if row.get("modality") == "image" and os.path.exists(row["path"]):
            try:
                with Image.open(row["path"]) as im:
                    im = im.convert("RGB")
                    im.thumbnail((256, 256))
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=72)
                thumb = (f"<img src='data:image/jpeg;base64,"
                         f"{base64.b64encode(buf.getvalue()).decode()}'>")
            except Exception:
                thumb = "<div class='miss'>no preview</div>"
        else:
            thumb = f"<div class='miss'>{row.get('modality')}</div>"
        color = f"<span class='c'>{row['color']:.3f}</span>" if "color" in row else ""
        cells.append(f"<figure>{thumb}<figcaption><b>{row['score']:.4f}</b> {color}"
                     f"<div>{os.path.basename(row['path'])}</div></figcaption></figure>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><meta charset='utf-8'><title>" + query +
                 "</title><style>body{font:14px system-ui;margin:2em;background:#111;color:#ddd}"
                 "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px}"
                 "figure{margin:0;background:#1c1c1c;padding:8px;border-radius:8px}"
                 "img{width:100%;border-radius:4px;display:block}.miss{height:160px;display:grid;"
                 "place-items:center;color:#888}figcaption{margin-top:6px;word-break:break-all}"
                 ".c{color:#8cf}</style><h1>" + query + f"</h1><main>{''.join(cells)}</main>")


def build_parser():
    p = argparse.ArgumentParser(prog="embed_search.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", help=f"server base URL (default {DEFAULT_URL})")
    p.add_argument("--model", help="model name the server routes to")
    p.add_argument("--api-key", help="bearer token (prefer EMBED_SEARCH_API_KEY)")
    p.add_argument("--config", help=f"config file (default {CONFIG_PATH})")
    p.add_argument("--root", help="folder to index or search")
    p.add_argument("--index", help="index file path")
    p.add_argument("--timeout", type=int, default=300, help="request timeout in seconds")
    p.add_argument("--version", action="version", version=f"embed_search.py {VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check the server, list models, probe an embedding")

    def add_scope(sp):
        # SUPPRESS keeps a value given before the subcommand when it is omitted here
        sp.add_argument("--root", default=argparse.SUPPRESS, help="folder to index or search")
        sp.add_argument("--index", default=argparse.SUPPRESS, help="index file path")
        return sp

    ini = add_scope(sub.add_parser("init", help="save url/model/api-key/root to the config file"))

    idx = add_scope(sub.add_parser("index", help="embed new or changed files into the index"))
    idx.add_argument("--modality", choices=["auto", "image", "audio", "text"], default="auto")
    idx.add_argument("--include", help="comma-separated glob patterns, e.g. '*.png,sub/*.jpg'")
    idx.add_argument("--exclude-dir", help="comma-separated directory names to skip")
    idx.add_argument("--limit", type=int, help="index at most N files (for a pilot run)")
    idx.add_argument("--batch", type=int, default=16, help="items per request (default 16)")
    idx.add_argument("--resize", type=int, default=768,
                     help="longest image side before embedding, 0 disables (default 768)")
    idx.add_argument("--max-chars", type=int, default=4000, help="text files: characters to embed")
    idx.add_argument("--force", action="store_true", help="re-embed everything")
    idx.add_argument("--compact", action="store_true", help="drop entries for deleted files")

    sr = add_scope(sub.add_parser("search", help="rank indexed files against a text query"))
    sr.add_argument("query")
    sr.add_argument("--also", action="append", help="extra phrasing, averaged with the query")
    sr.add_argument("--must", action="append", help="second facet; score is the minimum across facets")
    sr.add_argument("--color", choices=sorted(COLOR_RANGES), help="keep only images with this hue")
    sr.add_argument("--min-color", type=float, help="minimum fraction of pixels in --color (e.g. 0.03)")
    sr.add_argument("--max-color", type=float, help="maximum fraction of pixels in --color")
    sr.add_argument("--modality", choices=["image", "audio", "text"], help="restrict to one modality")
    sr.add_argument("--dims", type=int, help="truncate vectors to N dims (Matryoshka: 512, 256, 128)")
    sr.add_argument("--top", type=int, default=20)
    sr.add_argument("--offset", type=int, default=0)
    sr.add_argument("--json", action="store_true", help="machine-readable output on stdout")
    sr.add_argument("--html", metavar="FILE", help="also write an HTML contact sheet")

    add_scope(sub.add_parser("stats", help="summarize the index"))
    return p


def cmd_stats(args, cfg, client):
    _, idx = resolve_paths(args, cfg, need_root=False)
    meta, recs = read_index(idx)
    by = {}
    for rec in recs:
        by[rec.get("m", "?")] = by.get(rec.get("m", "?"), 0) + 1
    print(json.dumps({"index": idx, "entries": len(recs), "by_modality": by,
                      "model": meta.get("model"), "last_write": meta.get("ts"),
                      "bytes": os.path.getsize(idx) if os.path.exists(idx) else 0}, indent=2))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config or CONFIG_PATH)
    url = resolve(args, cfg, "url", "EMBED_SEARCH_URL", DEFAULT_URL)
    model = resolve(args, cfg, "model", "EMBED_SEARCH_MODEL")
    key = resolve(args, cfg, "api_key", "EMBED_SEARCH_API_KEY")
    if args.cmd != "doctor" and not model:
        die(2, "no model set. Run: embed_search.py init --url URL --model NAME "
               "(see `doctor` for the names the server offers).")
    client = Client(url, model or "unused", key, args.timeout)
    return {"doctor": cmd_doctor, "init": cmd_init, "index": cmd_index,
            "search": cmd_search, "stats": cmd_stats}[args.cmd](args, cfg, client)


if __name__ == "__main__":
    sys.exit(main())
