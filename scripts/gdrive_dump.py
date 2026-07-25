"""Chunked downloader for link-shared Google Drive folders (no auth needed).

Downloads in VALUE order (chat exports -> audio -> docs -> media) and stops when the
chunk cap is hit, so the twin trains on the most useful 30GB first.

  uv run python scripts/gdrive_dump.py inventory      # scrape folder trees -> manifest
  uv run python scripts/gdrive_dump.py download --max-gb 30
  uv run python scripts/gdrive_dump.py status
  uv run python scripts/gdrive_dump.py sort           # file the dump into data/raw/*

Re-runnable: finished files are skipped (state json), partial files restart.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DUMP_DIR = REPO_ROOT / "data" / "raw" / "_dump"
MANIFEST = REPO_ROOT / "data" / "processed" / "drive_manifest.json"
STATE = REPO_ROOT / "data" / "processed" / "drive_dump_state.json"

# Folder IDs live in data/drive_roots.json (gitignored) — they are link-share
# secrets granting access to the raw personal archives. Never commit them.
_ROOTS_FILE = REPO_ROOT / "data" / "drive_roots.json"
ROOTS: dict[str, str] = (
    json.loads(_ROOTS_FILE.read_text(encoding="utf-8")) if _ROOTS_FILE.exists() else {}
)

# roots whose photos/videos ARE the payload (face + voice of the user)
WANT_MEDIA_ROOTS = {"my_vids"}
# path patterns of personal-video folders inside otherwise-skipped media roots
WANT_MEDIA_RE = re.compile(r"Captured Videos|rec smj5|(^|/)video(/|$)", re.IGNORECASE)

AUDIO_EXTS = {".opus", ".ogg", ".mp3", ".m4a", ".wav", ".aac", ".flac", ".wma", ".amr", ".3ga"}
TEXT_EXTS = {".txt", ".json", ".csv", ".md", ".zip", ".html", ".vcf", ".xml"}
DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".xlsx"}

# 0 downloads first
def priority(name: str) -> int:
    ext = Path(name).suffix.lower()
    low = name.lower()
    if "whatsapp chat" in low or ext == ".txt":
        return 0
    if ext in AUDIO_EXTS:
        return 1
    if ext in TEXT_EXTS:
        return 2
    if ext in DOC_EXTS:
        return 3
    return 4  # photos, videos, apks, everything else


_ENTRY = re.compile(
    r'<div class="flip-entry"[^>]*id="entry-(?P<id>[\w-]+)".*?'
    r'href="(?P<href>[^"]+)".*?'
    r'<div class="flip-entry-title">(?P<title>.*?)</div>',
    re.DOTALL,
)


def list_folder(session: requests.Session, folder_id: str) -> list[dict]:
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    r = session.get(url, timeout=60)
    r.raise_for_status()
    entries = []
    for m in _ENTRY.finditer(r.text):
        entries.append(
            {
                "id": m.group("id"),
                "name": html.unescape(m.group("title")).strip(),
                "is_folder": "/folders/" in m.group("href"),
            }
        )
    return entries


def build_inventory() -> list[dict]:
    session = requests.Session()
    files: list[dict] = []
    for root_name, root_id in ROOTS.items():
        queue: list[tuple[str, str]] = [(root_id, root_name)]
        while queue:
            folder_id, rel = queue.pop(0)
            try:
                entries = list_folder(session, folder_id)
            except requests.RequestException as exc:
                print(f"  !! failed to list {rel}: {exc}")
                continue
            print(f"  {rel}: {len(entries)} entries")
            for e in entries:
                if e["is_folder"]:
                    queue.append((e["id"], f"{rel}/{e['name']}"))
                else:
                    files.append({"id": e["id"], "name": e["name"], "path": rel})
            time.sleep(0.3)  # be polite
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(files, indent=1, ensure_ascii=False), encoding="utf-8")
    return files


def download_file(session: requests.Session, file_id: str, dest: Path) -> int:
    """Returns bytes written. Handles the large-file 'virus scan' confirm page."""
    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "confirm": "t"}
    r = session.get(url, params=params, stream=True, timeout=120)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "text/html" in ctype:  # confirm form page -> resubmit with its hidden fields
        page = r.text
        fields = dict(re.findall(r'name="(\w+)" value="([^"]*)"', page))
        if not fields:
            raise RuntimeError("download blocked (no confirm fields — file may need sign-in)")
        r = session.get(url, params=fields, stream=True, timeout=120)
        r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    written = 0
    with open(part, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            written += len(chunk)
    part.replace(dest)
    return written


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=0), encoding="utf-8")


def cmd_download(max_gb: float, skip_media: bool) -> None:
    files = json.loads(MANIFEST.read_text(encoding="utf-8"))
    state = load_state()
    files.sort(key=lambda f: (priority(f["name"]), f["path"]))
    cap = int(max_gb * (1 << 30))
    session = requests.Session()
    new_bytes = 0
    done = skipped = failed = 0
    for f in files:
        if state.get(f["id"], {}).get("done"):
            skipped += 1
            continue
        root = f["path"].split("/", 1)[0]
        if (
            skip_media
            and priority(f["name"]) == 4
            and root not in WANT_MEDIA_ROOTS
            and not WANT_MEDIA_RE.search(f["path"])
        ):
            continue
        if new_bytes >= cap:
            print(f"\nChunk cap reached ({new_bytes / (1<<30):.1f} GB) — stopping.")
            break
        dest = DUMP_DIR / f["path"] / f["name"]
        try:
            n = download_file(session, f["id"], dest)
        except Exception as exc:
            print(f"  !! {f['path']}/{f['name']}: {exc}")
            failed += 1
            continue
        state[f["id"]] = {"done": True, "bytes": n}
        new_bytes += n
        done += 1
        if done % 25 == 0:
            save_state(state)
            print(f"  {done} files, {new_bytes / (1<<30):.2f} GB this chunk")
    save_state(state)
    print(f"\ndownloaded={done}  already-had={skipped}  failed={failed}  new={new_bytes / (1<<30):.2f} GB")


def cmd_status() -> None:
    files = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else []
    state = load_state()
    got = sum(1 for f in files if state.get(f["id"], {}).get("done"))
    total_bytes = sum(v.get("bytes", 0) for v in state.values() if v.get("done"))
    by_prio: dict[int, list[int]] = {}
    for f in files:
        p = priority(f["name"])
        s = state.get(f["id"], {})
        by_prio.setdefault(p, [0, 0])
        by_prio[p][0] += 1
        by_prio[p][1] += 1 if s.get("done") else 0
    names = {0: "chats/txt", 1: "audio", 2: "text/zip", 3: "docs", 4: "media/other"}
    print(f"manifest: {len(files)} files | downloaded: {got} ({total_bytes / (1<<30):.2f} GB)")
    for p in sorted(by_prio):
        t, d = by_prio[p]
        print(f"  {names[p]:<12} {d}/{t}")


SORT_RULES = [
    (re.compile(r"whatsapp chat.*\.txt$", re.I), "whatsapp"),
    (re.compile(r"(^|/)(PTT-\d{8}|WhatsApp Voice Notes)", re.I), "voice_notes"),
    (re.compile(r"(call ?rec|recordings|\bcall\b|callrecord)", re.I), "calls"),
]


def cmd_sort(move: bool) -> None:
    counts: dict[str, int] = {}
    for path in DUMP_DIR.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(DUMP_DIR).as_posix()
        target = None
        for rx, bucket in SORT_RULES:
            if rx.search(rel):
                target = bucket
                break
        if target is None and path.suffix.lower() in AUDIO_EXTS:
            target = "voice_notes" if "voice" in rel.lower() else "calls"
        if target is None:
            continue  # leave non-matching files in _dump
        dest = REPO_ROOT / "data" / "raw" / target / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            (shutil.move if move else shutil.copy2)(str(path), dest)
        counts[target] = counts.get(target, 0) + 1
    print("sorted:", counts or "nothing matched")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory")
    d = sub.add_parser("download")
    d.add_argument("--max-gb", type=float, default=30.0)
    d.add_argument("--include-media", action="store_true", help="also download photos/videos/other")
    sub.add_parser("status")
    s = sub.add_parser("sort")
    s.add_argument("--move", action="store_true", help="move instead of copy")
    args = ap.parse_args()

    if args.cmd == "inventory":
        files = build_inventory()
        prios: dict[int, int] = {}
        for f in files:
            prios[priority(f["name"])] = prios.get(priority(f["name"]), 0) + 1
        print(f"\nmanifest: {len(files)} files -> {MANIFEST}")
        print("by priority:", prios)
    elif args.cmd == "download":
        cmd_download(args.max_gb, skip_media=not args.include_media)
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "sort":
        cmd_sort(args.move)


if __name__ == "__main__":
    sys.exit(main())
