"""Encrypted local storage for browser state and answer memory.

On macOS ``keyring`` uses Keychain. The package is optional so normal offline
CI never needs browser dependencies. Live browser setup fails closed when the
Keychain-backed key cannot be obtained.
"""

from __future__ import annotations

import os
from pathlib import Path


class BrowserSecurityError(RuntimeError):
    """Protected local storage is unavailable."""


def _fernet():
    try:
        import keyring
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise BrowserSecurityError("Install the optional application extra for Keychain-protected browser state.") from exc
    service, account = "jobvis", f"browser-state-{os.getuid()}"
    secret = keyring.get_password(service, account)
    if not secret:
        secret = Fernet.generate_key().decode("ascii")
        keyring.set_password(service, account, secret)
    return Fernet(secret.encode("ascii"))


def save_encrypted(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_fernet().encrypt(payload))
    path.chmod(0o600)


def load_encrypted(path: Path) -> bytes:
    if not path.exists():
        raise BrowserSecurityError(f"Protected state does not exist: {path}")
    try:
        return _fernet().decrypt(path.read_bytes())
    except Exception as exc:  # noqa: BLE001 - never leak key material
        raise BrowserSecurityError("Could not decrypt protected local state.") from exc
