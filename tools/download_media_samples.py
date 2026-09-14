#!/usr/bin/env python3
"""Download small, auditable samples of public-domain media.

The script keeps NASA and Wikimedia Commons in separate trees and records the
source metadata for every downloaded file. It intentionally uses a byte cap
per source and media type so a source dump cannot fill the data volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import time
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


NASA_API = "https://images-api.nasa.gov/search"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "FirstAgent-media-sampler/1.0"
NASA_TYPES = ("image", "video", "audio")
COMMONS_CATEGORIES = {
    "image": "Public domain images",
    "video": "Videos of films in the public domain",
    "audio": "LibriVox recordings",
}
EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"},
    "video": {".mp4", ".webm", ".ogv", ".mov"},
    "audio": {".mp3", ".ogg", ".oga", ".wav", ".m4a", ".flac"},
}


def request_json(url: str, params: dict[str, str]) -> dict:
    request = Request(
        f"{url}?{urlencode(params)}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def request_size(url: str) -> int | None:
    request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def extension(url: str, media_type: str) -> str | None:
    path = url.split("?", 1)[0].lower()
    suffix = Path(path).suffix
    return suffix if suffix in EXTENSIONS[media_type] else None


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value[:160] or "media"


def download_file(url: str, destination: Path, remaining: int) -> int:
    size = request_size(url)
    if size is not None and size > remaining:
        return 0
    request = Request(url, headers={"User-Agent": USER_AGENT})
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > remaining:
                return 0
            total = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > remaining:
                    return 0
                output.write(block)
        temporary.replace(destination)
        return total
    except Exception:
        return 0
    finally:
        if temporary.exists():
            temporary.unlink()


def record(path: Path, manifest: Path, source: str, media_type: str, metadata: dict) -> None:
    hasher = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    item = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "source": source,
        "media_type": media_type,
        "metadata": metadata,
    }
    with manifest.open("a", encoding="utf-8") as output:
        output.write(json.dumps(item, ensure_ascii=False) + "\n")


def nasa_candidates(media_type: str):
    for page in range(1, 101):
        payload = request_json(
            NASA_API,
            {
                "q": "NASA",
                "media_type": media_type,
                "page": str(page),
                "page_size": "100",
            },
        )
        items = payload.get("collection", {}).get("items", [])
        if not items:
            break
        for item in items:
            data = (item.get("data") or [{}])[0]
            href = item.get("href", "")
            if not href.endswith("collection.json"):
                nasa_id = data.get("nasa_id")
                if not nasa_id:
                    continue
                href = (
                    "https://images-assets.nasa.gov/"
                    f"{media_type}/{quote(nasa_id, safe='')}/collection.json"
                )
            description = (data.get("description") or "").lower()
            if "third-party copyright" in description or "copyright protected" in description:
                continue
            try:
                request = Request(href, headers={"User-Agent": USER_AGENT})
                with urlopen(request, timeout=60) as response:
                    assets = json.load(response)
            except Exception:
                continue
            candidates = [
                asset for asset in assets
                if extension(asset, media_type) is not None
                and "metadata.json" not in asset
            ]
            candidates.sort(key=lambda asset: ("~orig" not in asset, len(asset)))
            if candidates:
                yield candidates[0], data


def commons_files(category: str):
    pending = [f"Category:{category}"]
    visited: set[str] = set()
    titles: list[str] = []
    while pending and len(visited) < 100 and len(titles) < 5000:
        current = pending.pop(0)
        if current in visited:
            continue
        visited.add(current)
        continuation: dict[str, str] = {}
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": current,
                "cmtype": "file|subcat",
                "cmlimit": "500",
                "format": "json",
                **continuation,
            }
            payload = request_json(COMMONS_API, params)
            for member in payload.get("query", {}).get("categorymembers", []):
                title = member["title"]
                if member["ns"] == 14:
                    pending.append(title)
                elif member["ns"] == 6:
                    titles.append(title)
            continuation = payload.get("continue", {})
            if not continuation:
                break
    for start in range(0, len(titles), 50):
        batch = titles[start : start + 50]
        payload = request_json(
            COMMONS_API,
            {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "imageinfo|info",
                "iiprop": "url|size|mime|extmetadata",
                "format": "json",
            },
        )
        for page in payload.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            license_name = (
                info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
            ).lower()
            if "public domain" not in license_name and "cc0" not in license_name:
                continue
            url = info.get("url", "")
            if url:
                yield url, {
                    "title": page.get("title"),
                    "license": license_name,
                    "mime": info.get("mime"),
                    "source_page": "https://commons.wikimedia.org/wiki/" + quote(
                        page.get("title", "").replace(" ", "_"), safe=":_"
                    ),
                }


def download_source(root: Path, source: str, media_type: str, cap: int) -> None:
    destination = root / source / media_type
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "manifest.jsonl"
    used = sum(
        path.stat().st_size
        for path in destination.iterdir()
        if path.is_file() and path.name != manifest.name and not path.name.endswith(".part")
    )
    iterator = nasa_candidates(media_type) if source == "nasa" else commons_files(COMMONS_CATEGORIES[media_type])
    for index, (url, metadata) in enumerate(iterator):
        # Avoid spending a long time searching for a file that fits a tiny
        # remainder after the byte cap has effectively been reached.
        if used >= cap or cap - used < 16 * 1024 * 1024:
            break
        suffix = extension(url, media_type)
        if suffix is None:
            continue
        identifier = metadata.get("nasa_id") or metadata.get("title") or str(index)
        destination_file = destination / f"{index:06d}_{safe_name(str(identifier))}{suffix}"
        if destination_file.exists():
            continue
        remaining = cap - used
        size = download_file(url.replace("http://", "https://"), destination_file, remaining)
        if not size:
            continue
        used += size
        record(destination_file, manifest, source, media_type, {"url": url, **metadata})
        time.sleep(0.05)
    print(f"{source}/{media_type}: {used:,} bytes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cap-gib", type=float, default=1.0)
    args = parser.parse_args()
    if args.cap_gib <= 0:
        parser.error("--cap-gib must be positive")
    cap = int(args.cap_gib * 1024**3)
    for source in ("nasa", "commons"):
        for media_type in NASA_TYPES:
            download_source(args.root, source, media_type, cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
