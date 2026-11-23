"""Stable posting identity shared by discovery and the application tracker."""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def posting_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not host or parsed.username is not None:
        return ""
    if port is not None and port != (443 if parsed.scheme == "https" else 80):
        host = parsed.netloc.lower()
    query = sorted(
        (key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"gclid", "fbclid"}
    )
    path = parsed.path.rstrip("/")
    if not path and not query:
        return ""
    return urlunsplit(("https", host, path, urlencode(query), ""))
