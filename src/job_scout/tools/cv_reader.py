"""Extract plain text from a CV PDF with pypdf.

Text extraction only — the LLM does the structuring downstream. No OCR, no
layout analysis (out of scope by design).
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from job_scout.graph.schemas import CVLink


class CVReadError(ValueError):
    """Raised when a PDF cannot be read or yields no usable text."""


def extract_cv_document(path: str | Path) -> tuple[str, list[CVLink]]:
    """Return the concatenated text of a CV PDF.

    Raises ``CVReadError`` if the file is missing, unreadable, or empty of text
    (e.g. a scanned image with no text layer) so the caller can surface a clean
    message instead of an opaque parser traceback.
    """
    path = Path(path)
    if not path.exists():
        raise CVReadError(f"CV file not found: {path}")

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - normalize any parser failure
        raise CVReadError(f"Could not read PDF {path.name}: {exc}") from exc

    text = "\n".join(pages).strip()
    if not text:
        raise CVReadError(f"No extractable text in {path.name}. Is it a scanned image? OCR is out of scope for this project.")
    links: list[CVLink] = []
    seen: set[tuple[str, int]] = set()
    for page_number, page in enumerate(reader.pages, start=1):
        annotations = page.get("/Annots") or []
        for annotation_ref in annotations:
            try:
                annotation = annotation_ref.get_object()
                action = annotation.get("/A")
                action = action.get_object() if action else None
                url = str(action.get("/URI")) if action and action.get("/URI") else ""
                url = url.strip()
                if not url or not (url.startswith("http://") or url.startswith("https://") or url.startswith("mailto:")):
                    continue
                key = (url, page_number)
                if key in seen:
                    continue
                seen.add(key)
                links.append(CVLink(label=_link_label(url), url=url, page=page_number))
            except Exception:  # noqa: BLE001, S112 - one malformed annotation must not lose the CV
                continue
    return text, links


def extract_cv_text(path: str | Path) -> str:
    """Backward-compatible text-only wrapper."""
    return extract_cv_document(path)[0]


def _link_label(url: str) -> str:
    """Give an annotation a useful editable label when PDF text has no anchor metadata."""
    if url.startswith("mailto:"):
        return "Email"
    host = url.split("/", 3)[2].removeprefix("www.") if "://" in url else url
    if "linkedin" in host:
        return "LinkedIn"
    if "github" in host:
        return "GitHub"
    if "portfolio" in url or "vercel.app" in host:
        return "Portfolio / project"
    return host or "Link"
