"""Privacy-conscious local source packaging for managed scans."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import stat
import subprocess  # nosec B404
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from strix.interface.cloud import http


MAX_FILES = 20_000
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_TOTAL_BYTES = 250 * 1024 * 1024
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024

_ALWAYS_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "vendor",
        "venv",
        "__pycache__",
        "dist",
        "build",
        "coverage",
        "target",
    }
)
_SENSITIVE_NAMES = frozenset(
    {
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "credentials.json",
        "service-account.json",
        "service_account.json",
        ".env",
        ".npmrc",
        ".pypirc",
        ".netrc",
    }
)
_SENSITIVE_PATTERNS = (
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "*.jks",
    "secrets.*",
    "secret.*",
    ".env.*",
)
_ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tgz",
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".7z",
    ".rar",
    ".gz",
    ".bz2",
    ".xz",
)


@dataclass(frozen=True)
class SelectedFile:
    path: Path
    archive_name: str
    size: int


@dataclass(frozen=True)
class SourceManifest:
    source: Path
    files: tuple[SelectedFile, ...]
    excluded: Counter[str]
    include_hidden: bool
    include_sensitive: bool
    include_archives: bool

    @property
    def total_bytes(self) -> int:
        return sum(item.size for item in self.files)

    def as_dict(
        self,
        *,
        show_files: bool,
        archive_bytes: int | None = None,
        archive_sha256: str | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "source": str(self.source),
            "file_count": len(self.files),
            "uncompressed_bytes": self.total_bytes,
            "excluded_count": sum(self.excluded.values()),
            "excluded_by_reason": dict(sorted(self.excluded.items())),
            "include_hidden": self.include_hidden,
            "include_sensitive": self.include_sensitive,
            "include_archives": self.include_archives,
        }
        if archive_bytes is not None:
            result["archive_bytes"] = archive_bytes
        if archive_sha256 is not None:
            result["archive_sha256"] = archive_sha256
        if show_files:
            result["files"] = [item.archive_name for item in self.files]
        return result


@dataclass(frozen=True)
class SourceBundle:
    manifest: SourceManifest
    archive_path: Path
    archive_bytes: int
    archive_sha256: str

    def summary(self, *, show_files: bool) -> dict[str, object]:
        return self.manifest.as_dict(
            show_files=show_files,
            archive_bytes=self.archive_bytes,
            archive_sha256=self.archive_sha256,
        )


def prepare_source(
    value: str,
    *,
    include_hidden: bool,
    include_sensitive: bool,
    include_archives: bool,
    exclude: list[str],
) -> SourceBundle:
    """Select safe source files and build a bounded temporary ZIP archive."""
    source = Path(value).expanduser().resolve()
    if not source.is_dir():
        raise http.CloudError(f"--source must be a directory: {source}")
    manifest = select_source(
        source,
        include_hidden=include_hidden,
        include_sensitive=include_sensitive,
        include_archives=include_archives,
        exclude=exclude,
    )
    if not manifest.files:
        raise http.CloudError("no files remain after applying source upload exclusions.")

    with tempfile.NamedTemporaryFile(prefix="strix-source-", suffix=".zip", delete=False) as handle:
        archive_path = Path(handle.name)
    try:
        _write_archive(archive_path, manifest.files)
    except BaseException:
        archive_path.unlink(missing_ok=True)
        raise
    archive_bytes = archive_path.stat().st_size
    if archive_bytes > MAX_ARCHIVE_BYTES:
        archive_path.unlink(missing_ok=True)
        raise http.CloudError(
            "source archive is larger than the 50 MB upload limit; narrow --source or "
            "add --exclude patterns."
        )
    digest = _sha256(archive_path)
    return SourceBundle(manifest, archive_path, archive_bytes, digest)


def select_source(
    source: Path,
    *,
    include_hidden: bool = False,
    include_sensitive: bool = False,
    include_archives: bool = False,
    exclude: list[str] | None = None,
) -> SourceManifest:
    excluded: Counter[str] = Counter()
    selected: list[SelectedFile] = []
    patterns = [*_load_ignore_patterns(source), *(exclude or [])]
    total_bytes = 0
    for relative in _candidate_paths(source):
        archive_name = relative.as_posix()
        reason = _exclusion_reason(
            relative,
            include_hidden=include_hidden,
            include_sensitive=include_sensitive,
            include_archives=include_archives,
            patterns=patterns,
        )
        if reason:
            excluded[reason] += 1
            continue
        path = source / relative
        try:
            info = path.lstat()
        except OSError:
            excluded["unreadable"] += 1
            continue
        if not stat.S_ISREG(info.st_mode):
            excluded["symlink_or_non_file"] += 1
            continue
        if info.st_size > MAX_FILE_BYTES:
            raise http.CloudError(
                f"{archive_name} is larger than the 25 MB per-file limit; exclude it explicitly."
            )
        selected.append(SelectedFile(path, archive_name, info.st_size))
        total_bytes += info.st_size
        if len(selected) > MAX_FILES:
            raise http.CloudError(
                f"source contains more than {MAX_FILES:,} files; narrow --source or add exclusions."
            )
        if total_bytes > MAX_TOTAL_BYTES:
            raise http.CloudError(
                "selected source is larger than the 250 MB expanded-size limit; narrow --source "
                "or add --exclude patterns."
            )
    selected.sort(key=lambda item: item.archive_name)
    return SourceManifest(
        source,
        tuple(selected),
        excluded,
        include_hidden,
        include_sensitive,
        include_archives,
    )


def remove_bundle(bundle: SourceBundle) -> None:
    bundle.archive_path.unlink(missing_ok=True)


def _candidate_paths(source: Path) -> list[Path]:
    git_root = _git_root(source)
    if git_root is not None:
        git = shutil.which("git")
        if git is None:
            return [path.relative_to(source) for path in source.rglob("*")]
        relative_source = source.relative_to(git_root)
        command = [
            git,
            "-C",
            str(git_root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
        ]
        if relative_source != Path():
            command.append(relative_source.as_posix())
        result = subprocess.run(  # noqa: S603  # nosec B603
            command, check=False, capture_output=True
        )
        if result.returncode == 0:
            paths: list[Path] = []
            for raw in result.stdout.split(b"\0"):
                if not raw:
                    continue
                repo_relative = Path(os.fsdecode(raw))
                try:
                    paths.append(repo_relative.relative_to(relative_source))
                except ValueError:
                    continue
            return paths
    return [path.relative_to(source) for path in source.rglob("*")]


def _git_root(source: Path) -> Path | None:
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(  # noqa: S603  # nosec B603
        [git, "-C", str(source), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return Path(result.stdout.strip()).resolve()
    except OSError:
        return None


def _exclusion_reason(  # noqa: PLR0911
    relative: Path,
    *,
    include_hidden: bool,
    include_sensitive: bool,
    include_archives: bool,
    patterns: list[str],
) -> str | None:
    parts = relative.parts
    if any(part == ".git" for part in parts):
        return "git_metadata"
    if any(part in _ALWAYS_EXCLUDED_DIRS for part in parts[:-1]):
        return "dependency_or_build_output"
    if not include_hidden and any(part.startswith(".") for part in parts):
        return "hidden"
    posix = PurePosixPath(relative.as_posix())
    if any(
        posix.match(pattern) or fnmatch.fnmatch(relative.as_posix(), pattern)
        for pattern in patterns
    ):
        return "user_pattern"
    name = relative.name.lower()
    if not include_sensitive and (
        name in _SENSITIVE_NAMES
        or any(fnmatch.fnmatch(name, pattern) for pattern in _SENSITIVE_PATTERNS)
    ):
        return "sensitive_filename"
    if not include_archives and name.endswith(_ARCHIVE_SUFFIXES):
        return "nested_archive"
    return None


def _write_archive(destination: Path, files: tuple[SelectedFile, ...]) -> None:
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for item in files:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(item.path, flags)
            except OSError as exc:
                raise http.CloudError(f"could not safely read {item.archive_name}: {exc}") from exc
            with os.fdopen(descriptor, "rb") as source_file:
                current = os.fstat(source_file.fileno())
                if not stat.S_ISREG(current.st_mode) or current.st_size != item.size:
                    raise http.CloudError(
                        f"{item.archive_name} changed while the source archive was being built; "
                        "retry."
                    )
                info = zipfile.ZipInfo(item.archive_name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with archive.open(info, "w", force_zip64=True) as target:
                    shutil.copyfileobj(source_file, target, length=1024 * 1024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ignore_patterns(source: Path) -> list[str]:
    path = source / ".strixignore"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise http.CloudError(f"could not read {path}: {exc}") from exc
    patterns: list[str] = []
    for line_number, raw in enumerate(lines, start=1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if value.startswith("!"):
            raise http.CloudError(
                f"{path}:{line_number}: negated patterns are not supported; use exclude-only globs."
            )
        patterns.append(value)
    return patterns
