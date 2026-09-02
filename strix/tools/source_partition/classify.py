"""Ignore-layer 3 - source classification (tests / schemas / generated code /
vendor trees / data / plain text).

This is a *weight-policy* concern, separate from exclusion (layers 1-2 in
``exclusions.py`` / ``gitignore.py``).  It only ever sees paths that survived
those layers - the classification layer consults the exclusion layer's
results, never the other way around.

``GENERATED`` / ``VENDOR`` files are classified here and then dropped by the
unit builder (they are excluded *entirely* from the effective weight).
Everything else keeps a kind with its own weight factor
(``PartitionConfig.weight``).

Heuristics are deliberately conservative and name/path driven (no content
sniffing): misclassification changes a weight, never correctness of the
inventory.  Case-folded segment markers, NFC-normalized paths.
"""

from __future__ import annotations

from strix.tools.source_partition.models import FileKind


__all__ = ["classify_file", "suffix_of"]

#: Directory basenames (any depth, case-folded) that mark a vendored tree.
VENDOR_DIR_NAMES: frozenset[str] = frozenset(
    {
        "vendor",
        "third_party",
        "thirdparty",
        "vendored",
        "bower_components",
        "jspm_packages",
    }
)

#: Directory basenames that mark generated output.
GENERATED_DIR_NAMES: frozenset[str] = frozenset({"generated", "gen", "codegen"})

#: Directory basenames that mark tests.
TEST_DIR_NAMES: frozenset[str] = frozenset({"test", "tests", "testing", "__tests__"})

#: Directory basenames that mark schema/contract trees.
SCHEMA_DIR_NAMES: frozenset[str] = frozenset(
    {"schema", "schemas", "openapi", "swagger", "proto", "protos", "graphql"}
)

#: File suffixes that are schemas/contracts regardless of name.
SCHEMA_SUFFIXES: frozenset[str] = frozenset(
    {".proto", ".avsc", ".thrift", ".graphql", ".gql", ".graphqls"}
)

#: File suffixes treated as data artifacts (dumps, fixtures, columnar stores).
DATA_SUFFIXES: frozenset[str] = frozenset(
    {
        ".csv",
        ".tsv",
        ".psv",
        ".jsonl",
        ".ndjson",
        ".parquet",
        ".avro",
        ".orc",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".sql",
        ".dump",
        ".feather",
        ".h5",
        ".hdf5",
        ".arrow",
    }
)

#: File suffixes recognized as hand-written source code.
SOURCE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".pyi",
        ".pyw",
        ".js",
        ".mjs",
        ".cjs",
        ".jsx",
        ".ts",
        ".mts",
        ".cts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".kts",
        ".scala",
        ".sc",
        ".cs",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".cxx",
        ".hpp",
        ".hxx",
        ".hh",
        ".m",
        ".mm",
        ".swift",
        ".php",
        ".rb",
        ".pl",
        ".pm",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ksh",
        ".dart",
        ".vue",
        ".svelte",
        ".erl",
        ".hrl",
        ".ex",
        ".exs",
        ".clj",
        ".cljs",
        ".cljc",
        ".hs",
        ".lhs",
        ".lua",
    }
)

#: Suffixes a schema *name marker* is allowed to apply to (keeps marker checks
#: from misclassifying e.g. "schema-less.md").
_SCHEMA_NAME_SUFFIXES: frozenset[str] = frozenset(
    {".json", ".yaml", ".yml", ".xml", ".sql", ".proto", ".graphql", ".gql", ".txt"}
)


def suffix_of(path: str) -> str:
    """Lower-cased final suffix including the dot (``""`` when absent)."""
    _, dot, ext = path.rpartition(".")
    return f".{ext.casefold()}" if dot else ""


def _is_generated_name(name: str) -> bool:
    folded = name.casefold()
    if ".generated" in folded:
        return True
    return folded.endswith(
        (
            ".min.js",
            ".min.css",
            "_pb2.py",
            "_pb.py",
            "_pb.js",
            "_pb.ts",
            "_pb.dart",
            ".pb.go",
            ".grpc.go",
            ".grpc.py",
            ".d.ts",
        )
    )


def _is_test_name(name: str) -> bool:
    folded = name.casefold()
    if folded == "conftest.py":
        return True
    stem = folded[: folded.rfind(".")] if "." in folded else folded
    if stem.startswith(("test_", "test-")) or stem.endswith(("_test", "-test")):
        return True
    return ".test" in stem or ".spec" in stem


def _is_schema_name(name: str) -> bool:
    folded = name.casefold()
    if suffix_of(folded) not in _SCHEMA_NAME_SUFFIXES:
        return False
    return "openapi" in folded or "swagger" in folded or "schema" in folded


def _kind_from_dirs(dirs: list[str]) -> FileKind | None:
    """Directory-marker classification (vendor/generated/test/schema dirs)."""
    for directory in dirs:
        segment = directory.casefold()
        if segment in VENDOR_DIR_NAMES:
            return FileKind.VENDOR
        if segment in GENERATED_DIR_NAMES:
            return FileKind.GENERATED
        if segment in TEST_DIR_NAMES:
            return FileKind.TEST
        if segment in SCHEMA_DIR_NAMES:
            return FileKind.SCHEMA
    return None


def _kind_from_name(name: str) -> FileKind | None:
    """Basename-marker classification (generated/test file names)."""
    if _is_generated_name(name):
        return FileKind.GENERATED
    if _is_test_name(name):
        return FileKind.TEST
    return None


def classify_file(rel: str) -> FileKind:
    """Classify one surviving file (normalized root-relative path)."""
    segments = rel.split("/")
    kind = _kind_from_dirs(segments[:-1])
    if kind is not None:
        return kind
    kind = _kind_from_name(segments[-1])
    if kind is not None:
        return kind
    suffix = suffix_of(segments[-1])
    if suffix in SCHEMA_SUFFIXES or _is_schema_name(segments[-1]):
        return FileKind.SCHEMA
    if suffix in DATA_SUFFIXES:
        return FileKind.DATA
    if suffix in SOURCE_SUFFIXES:
        return FileKind.SOURCE
    return FileKind.TEXT
