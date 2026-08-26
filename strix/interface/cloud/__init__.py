"""`strix cloud` — the managed Strix platform (app.strix.ai) from the terminal.

Every command maps to one operation of the public REST API. Output is JSON
when stdout is not a terminal, so agents can parse every result. Exit codes:
0 success, 1 error, 2 invalid usage, 4 authentication required, 5 payment
required.
"""

from __future__ import annotations

from rich.console import Console

from strix.interface.cloud.runner import resolve, run
from strix.interface.cloud.spec import DEFAULT_VERBS, GROUP_HELP, SPEC
from strix.interface.platform_cli import run_login


_USAGE_HEADER = """[bold]Usage:[/] strix cloud <command> [arguments]

[bold]Session commands:[/]
  login       Sign in to the managed platform and store an API token
  logout      Remove the stored API token
  whoami      Show the stored account, workspace, and token state
  credits     Show the credit balance of the workspace

[bold]Resource commands:[/]"""

_USAGE_FOOTER = """
Run [bold]strix cloud <command>[/] without a verb to list its verbs.
Every command accepts [bold]--json[/] and [bold]--token[/]. Write commands
accept [bold]--data[/] with a JSON object of extra request fields.
API reference: https://docs.app.strix.ai"""


def run_cloud(argv: list[str]) -> int:
    """Entry point for ``strix cloud …``. Returns a process exit code."""
    console = Console()
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_usage(console)
        return 0

    group, rest = argv[0], argv[1:]
    if group in ("login", "logout", "whoami"):
        session_argv = {"login": rest, "logout": ["logout"], "whoami": ["status"]}
        return run_login(session_argv[group])
    if group == "credits":
        group, rest = "billing", ["credits", *rest]

    if group not in SPEC:
        console.print(f"[red]Unknown command:[/] {group}")
        _print_usage(console)
        return 2
    resolved = resolve(group, rest)
    if resolved is None:
        _print_verbs(console, group)
        return 0 if not rest or rest[0] in ("-h", "--help", "help") else 2
    cmd, remaining = resolved
    verb_label = " ".join(rest[: len(rest) - len(remaining)]) or DEFAULT_VERBS.get(group, "")
    return run(group, verb_label, cmd, remaining)


def _print_usage(console: Console) -> None:
    console.print(_USAGE_HEADER)
    for group in SPEC:
        console.print(f"  {group:<14}{GROUP_HELP.get(group, '')}")
    console.print(_USAGE_FOOTER)


def _print_verbs(console: Console, group: str) -> None:
    console.print(f"[bold]strix cloud {group}[/] verbs:")
    for verb, cmd in SPEC[group].items():
        console.print(f"  {verb:<28}{cmd.help}")
