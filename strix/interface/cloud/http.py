"""HTTP client for the managed Strix platform API (app.strix.ai)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import requests

from strix.config import load_settings
from strix.interface.platform_cli import read_record


if TYPE_CHECKING:
    from pathlib import Path


_DEFAULT_TIMEOUT_S = 120
_app_url_override: str | None = None
_timeout_s: float = _DEFAULT_TIMEOUT_S

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_AUTH = 4
EXIT_PAYMENT = 5


class CloudError(Exception):
    """A failed cloud command. Carries the process exit code."""

    def __init__(self, message: str, *, exit_code: int = EXIT_ERROR, payload: Any = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.payload = payload


def configure(*, base_url: str | None = None, timeout: float | None = None) -> None:
    """Set the platform URL and the request timeout for this process."""
    global _app_url_override, _timeout_s  # noqa: PLW0603
    if base_url:
        _app_url_override = base_url.rstrip("/")
    if timeout:
        _timeout_s = timeout


def app_url() -> str:
    if _app_url_override:
        return _app_url_override
    return load_settings().viewer.app_url.rstrip("/")


def api_token(override: str | None = None) -> str:
    token = override or os.environ.get("STRIX_API_TOKEN")
    if not token:
        record = read_record()
        if record is not None:
            stored = record.get("api_token")
            if isinstance(stored, str):
                token = stored
    if not token or not token.strip():
        raise CloudError(
            "not signed in. Run `strix cloud login`, or set STRIX_API_TOKEN.",
            exit_code=EXIT_AUTH,
        )
    return token.strip()


def request(
    method: str,
    path: str,
    *,
    token: str | None = None,
    query: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> requests.Response:
    url = f"{app_url()}/api/v1{path}"
    headers = {"Authorization": f"Bearer {api_token(token)}"}
    try:
        response = requests.request(
            method,
            url,
            headers=headers,
            params={k: v for k, v in (query or {}).items() if v is not None} or None,
            json=body,
            timeout=_timeout_s,
        )
    except requests.RequestException as exc:
        raise CloudError(f"could not reach {app_url()}: {exc}") from exc
    return response


def upload_file(signed_url: str, upload_token: str, path: Path) -> None:
    """Stream a file to a platform-issued storage URL."""
    try:
        with path.open("rb") as stream:
            response = requests.put(
                signed_url,
                data=stream,
                headers={
                    "Authorization": f"Bearer {upload_token}",
                    "Content-Type": "application/zip",
                },
                timeout=_timeout_s,
            )
    except (OSError, requests.RequestException) as exc:
        raise CloudError(f"source upload failed: {exc}") from exc
    if not response.ok:
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                fields = cast("dict[str, Any]", payload)
                detail = str(fields.get("message") or fields.get("error") or "")
        except ValueError:
            pass
        raise CloudError(detail or f"source upload failed (HTTP {response.status_code})")


def parsed(response: requests.Response) -> Any:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            return response.json()
        except ValueError:
            return response.text
    return response.text


def check(response: requests.Response) -> Any:
    data = parsed(response)
    if response.ok:
        content_type = response.headers.get("content-type", "").lower()
        if "application/json" not in content_type:
            raise CloudError(
                "the server returned a non-JSON response. Check STRIX_APP_URL and preview "
                "access, then retry."
            )
        return data
    detail = ""
    if isinstance(data, dict):
        raw = cast("dict[str, Any]", data)
        detail = str(raw.get("detail") or raw.get("error") or "")
    message = detail or f"HTTP {response.status_code}"
    if response.status_code in (401, 403):
        raise CloudError(message, exit_code=EXIT_AUTH, payload=data)
    if response.status_code == 402:
        hint = detail or (
            "not enough credits. Run `strix cloud billing topup --credits N` to buy credits."
        )
        raise CloudError(hint, exit_code=EXIT_PAYMENT, payload=data)
    raise CloudError(message, exit_code=EXIT_ERROR, payload=data)
