"""Low-level byte reading, text decoding and binary detection.

Kept in one tiny module so tests can patch ``read_bytes`` deterministically to
simulate unreadable files without touching filesystem ACLs (see
``tests/test_source_partition.py::test_unreadable_file_is_skipped``).

Binary detection is *content-first* (the spec forbids relying on extension
alone): a NUL byte in the first sniffed chunk marks binary — except when a
UTF-8/UTF-16/UTF-32 BOM is present (those encodings legitimately contain NUL
bytes).  A conservative extension hint is applied afterwards, only to skip
obvious asset/binary formats without reading them (mirroring the suffix skips
the ``source_inspect_many`` search walker already applies).
"""

from __future__ import annotations

import codecs
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "ASSET_BINARY_SUFFIXES",
    "SNIFF_BYTES",
    "decode_text",
    "has_binary_extension",
    "is_binary_head",
    "read_bytes",
]

SNIFF_BYTES = 8192

#: Suffixes of asset/binary formats skipped without a read (extension hint,
#: applied *after* the content sniff would have caught real binaries anyway).
ASSET_BINARY_SUFFIXES: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".svg",  # asset (matches source_inspect_many's search skip list)
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".zip",
        ".gz",
        ".tar",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".jar",
        ".war",
        ".class",
        ".so",
        ".dll",
        ".exe",
        ".dylib",
        ".o",
        ".a",
        ".lib",
        ".obj",
        ".wasm",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wav",
        ".flac",
        ".ogg",
        ".webm",
        ".iso",
        ".img",
        ".psd",
        ".ai",
        ".bin",
    }
)

_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def read_bytes(path: Path, limit: int | None = None) -> bytes:
    """Read ``path`` fully (or up to ``limit`` bytes) as bytes."""
    with path.open("rb") as handle:
        return handle.read() if limit is None else handle.read(limit)


def is_binary_head(data: bytes) -> bool:
    """Content sniff on the first chunk: NUL byte ⇒ binary (BOMs are text)."""
    if not data:
        return False
    if any(data.startswith(bom) for bom, _encoding in _BOMS):
        return False
    return b"\x00" in data


def has_binary_extension(name: str) -> bool:
    """Extension hint (see module docstring — never the sole binary check)."""
    folded = name.casefold()
    return any(folded.endswith(suffix) for suffix in ASSET_BINARY_SUFFIXES)


def decode_text(data: bytes) -> str:
    """Decode file bytes for line counting.

    BOM-aware for UTF-8/16/32; everything else is decoded as UTF-8 with
    replacement (non-UTF-8 text like Latin-1 still counts its lines — decoding
    is for counting only, not for content fidelity).
    """
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return data.decode(encoding, errors="replace")
    return data.decode("utf-8", errors="replace")
