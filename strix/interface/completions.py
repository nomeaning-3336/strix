"""Shell completion scripts and candidates for the Strix CLI."""

from __future__ import annotations

import sys
from typing import Any

from strix.interface.cloud.spec import SPEC, Cmd


_ROOT_COMMANDS = ("cloud", "auth", "view", "completions", "completion")
_SESSION_COMMANDS = ("login", "logout", "whoami", "credits")
_COMMON_FLAGS = ("--json", "--token", "--app-url", "--timeout", "--help")


def run_completions(argv: list[str]) -> int:
    """Print a shell integration script or hidden completion candidates."""
    if argv and argv[0] == "--candidates":
        for candidate in completion_candidates(argv[1:]):
            sys.stdout.write(candidate + "\n")
        return 0
    if not argv or argv[0] in ("-h", "--help", "help"):
        sys.stdout.write(
            "Usage: strix completions <zsh|bash|fish>\n\n"
            "Enable tab completion for the current shell:\n"
            "  zsh:  source <(strix completions zsh)\n"
            "  bash: source <(strix completions bash)\n"
            "  fish: strix completions fish | source\n"
        )
        return 0
    shell = argv[0].lower()
    scripts = {"zsh": _zsh_script, "bash": _bash_script, "fish": _fish_script}
    generator = scripts.get(shell)
    if generator is None:
        sys.stderr.write(f"Unknown shell: {shell}. Choose zsh, bash, or fish.\n")
        return 2
    sys.stdout.write(generator())
    return 0


def completion_candidates(words: list[str]) -> list[str]:
    """Return candidates for words after the ``strix`` executable."""
    prior, current = _split_cursor(words)
    if not prior:
        return _matching(_ROOT_COMMANDS, current)
    if prior[0] != "cloud":
        return []
    return _cloud_candidates(prior[1:], current)


def _split_cursor(words: list[str]) -> tuple[list[str], str]:
    if not words:
        return [], ""
    return words[:-1], words[-1]


def _cloud_candidates(prior: list[str], current: str) -> list[str]:
    groups = (*_SESSION_COMMANDS, *SPEC, "workspace")
    if not prior:
        return _matching(groups, current)
    group = "workspaces" if prior[0] == "workspace" else prior[0]
    rest = prior[1:]
    if group in _SESSION_COMMANDS:
        return _matching(_session_flags(group), current)
    commands = SPEC.get(group)
    if commands is None:
        return _matching(groups, current)

    verb_paths = [verb.split() for verb in commands]
    if group == "workspaces":
        verb_paths.append(["use"])
    matching_paths = [path for path in verb_paths if path[: len(rest)] == rest]
    if not matching_paths:
        return []
    next_words = sorted({path[len(rest)] for path in matching_paths if len(path) > len(rest)})
    exact_verbs = [" ".join(path) for path in matching_paths if len(path) == len(rest)]
    candidates: list[str] = list(next_words)
    for verb in exact_verbs:
        if verb == "use" and group == "workspaces":
            candidates.extend(_COMMON_FLAGS)
        else:
            candidates.extend(_command_flags(commands[verb]))
    return _matching(candidates, current)


def _session_flags(group: str) -> tuple[str, ...]:
    if group == "login":
        return ("--no-browser", "--scopes", "--workspace", "--help")
    if group == "whoami":
        return ("--json", "--help")
    return ("--help",)


def _command_flags(cmd: Cmd) -> tuple[str, ...]:
    flags: list[str] = list(_COMMON_FLAGS)
    for param in cmd.query + cmd.body:
        flag = "--" + (param.flag or _kebab(param.name))
        flags.append(flag)
        if param.kind == "bool":
            flags.append("--no-" + flag.removeprefix("--"))
    if cmd.method in ("POST", "PUT", "PATCH"):
        flags.append("--data")
    if cmd.binary:
        flags.append("--output")
    if cmd.link:
        flags.append("--no-browser")
    if cmd.wait_path or cmd.wait_self:
        flags.append("--wait")
    if cmd.path == "/billing/topup":
        flags.extend(("--yes", "--no-pay", "--payment-method"))
    if cmd.path == "/billing/auto-topup" and cmd.method == "PUT":
        flags.append("--no-monthly-cap")
    return tuple(dict.fromkeys(flags))


def _kebab(value: str) -> str:
    output: list[str] = []
    for char in value:
        if char.isupper():
            output.extend(("-", char.lower()))
        else:
            output.append("-" if char == "_" else char)
    return "".join(output)


def _matching(candidates: Any, prefix: str) -> list[str]:
    return sorted({str(candidate) for candidate in candidates if str(candidate).startswith(prefix)})


def _zsh_script() -> str:
    return r"""#compdef strix
_strix() {
  local -a candidates
  candidates=("${(@f)$($words[1] completions --candidates "${words[@]:2}")}")
  _describe 'strix' candidates
}
compdef _strix strix
"""


def _bash_script() -> str:
    return r"""_strix_completion() {
  local -a candidates
  mapfile -t candidates < <(strix completions --candidates "${COMP_WORDS[@]:1:$COMP_CWORD}")
  COMPREPLY=( $(compgen -W "${candidates[*]}" -- "${COMP_WORDS[$COMP_CWORD]}") )
}
complete -F _strix_completion strix
"""


def _fish_script() -> str:
    return r"""function __strix_candidates
  set -l words (commandline -opc)
  set -e words[1]
  command strix completions --candidates $words (commandline -ct)
end
complete -c strix -f -a '(__strix_candidates)'
"""
