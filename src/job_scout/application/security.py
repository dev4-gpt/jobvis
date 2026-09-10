"""Encrypted local storage for browser state and answer memory.

On macOS ``keyring`` uses Keychain. The package is optional so normal offline
CI never needs browser dependencies. Live browser setup fails closed when the
Keychain-backed key cannot be obtained.
"""

from __future__ import annotations

import os
import tempfile
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
    try:
        secret = keyring.get_password(service, account)
        if not secret:
            secret = Fernet.generate_key().decode("ascii")
            keyring.set_password(service, account, secret)
        return Fernet(secret.encode("ascii"))
    except Exception as exc:
        raise BrowserSecurityError("Encrypted storage is unavailable; check Keychain access.") from exc


def save_encrypted(path: Path, payload: bytes) -> None:
    encrypted = _fernet().encrypt(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(prefix=".encrypted-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encrypted)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_encrypted(path: Path) -> bytes:
    if not path.exists():
        raise BrowserSecurityError(f"Protected state does not exist: {path}")
    try:
        return _fernet().decrypt(path.read_bytes())
    except Exception as exc:  # noqa: BLE001 - never leak key material
        raise BrowserSecurityError("Could not decrypt protected local state.") from exc
