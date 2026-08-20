"""Local port preflight behavior."""

from __future__ import annotations

import socket

import job_scout.app as app_module


def test_port_preflight_allows_reusable_local_socket(monkeypatch):
    calls: list[tuple[int, int, int]] = []

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def setsockopt(self, level, option, value):
            calls.append((level, option, value))

        def bind(self, address):
            assert address == ("127.0.0.1", 12345)

    monkeypatch.setattr(app_module.socket, "socket", lambda *args, **kwargs: FakeSocket())

    app_module._assert_port_available(12345, "wizard")

    assert calls == [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)]
