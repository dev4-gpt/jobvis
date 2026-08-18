"""Offline-testable ATS detection, field discovery, and safe mappings.

The HTML is untrusted reference data. This module only reads labels and
attributes; it never executes page instructions or submits a form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from html.parser import HTMLParser
from urllib.parse import urlparse


class ATSName(StrEnum):
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    UNKNOWN = "unknown"


class _FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.fields: list[FormField] = []
        self._labels: dict[str, str] = {}
        self._label_for: str | None = None
        self._label_text: list[str] = []
        self._select: FormField | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "label":
            self._label_for = values.get("for")
            self._label_text = []
        if tag not in {"input", "textarea", "select"}:
            return
        field_id = values.get("id") or values.get("name") or f"field-{len(self.fields) + 1}"
        field_type = values.get("type", "text") if tag == "input" else tag
        field = FormField(
            field_id=field_id,
            label=values.get("aria-label") or values.get("placeholder") or values.get("name") or field_id,
            name=values.get("name") or "",
            input_type=field_type.lower(),
            required="required" in values or values.get("aria-required") == "true",
            selector=f"#{field_id}" if values.get("id") else f"[name='{values.get('name')}']" if values.get("name") else "",
            sensitive=_is_sensitive(values.get("name", "") + " " + values.get("aria-label", "") + " " + field_id),
        )
        self.fields.append(field)
        if tag == "select":
            self._select = field

    def handle_endtag(self, tag: str) -> None:
        if tag == "label" and self._label_for:
            label = " ".join(self._label_text).strip()
            if label:
                self._labels[self._label_for] = label
                for field in self.fields:
                    if field.field_id == self._label_for:
                        field.label = label
            self._label_for = None
            self._label_text = []
        if tag == "select":
            self._select = None

    def handle_data(self, data: str) -> None:
        if self._label_for is not None:
            self._label_text.append(data)


@dataclass
class FormField:
    field_id: str
    label: str
    name: str = ""
    input_type: str = "text"
    required: bool = False
    selector: str = ""
    sensitive: bool = False
    options: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ApplicantFacts:
    """Explicit facts approved for safe autofill; no free-form CV inference."""

    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin_url: str = ""
    portfolio_url: str = ""
    location: str = ""


@dataclass
class FieldProposal:
    field_id: str
    label: str
    value: str = ""
    confidence: float = 0.0
    sensitive: bool = False
    approved: bool = False
    reason: str = ""


@dataclass
class FormInspection:
    ats: ATSName
    url: str
    fields: list[FormField]
    proposals: list[FieldProposal] = field(default_factory=list)
    pause_reason: str = ""


def detect_ats(url: str, html: str = "") -> ATSName:
    host = urlparse(url).netloc.lower()
    haystack = f"{host} {html[:20000].lower()}"
    if "greenhouse" in haystack:
        return ATSName.GREENHOUSE
    if "lever" in haystack:
        return ATSName.LEVER
    if "ashby" in haystack:
        return ATSName.ASHBY
    return ATSName.UNKNOWN


def discover_fields(url: str, html: str) -> FormInspection:
    parser = _FormParser()
    parser.feed(html)
    ats = detect_ats(url, html)
    proposals = propose_mappings(parser.fields, ApplicantFacts())
    pause_reason = "Manual login, MFA, or CAPTCHA may be required." if "login" in html.lower() else ""
    return FormInspection(ats=ats, url=url, fields=parser.fields, proposals=proposals, pause_reason=pause_reason)


def propose_mappings(fields: list[FormField], facts: ApplicantFacts) -> list[FieldProposal]:
    proposals: list[FieldProposal] = []
    for form_field in fields:
        key = _field_key(form_field.label, form_field.name, form_field.field_id)
        value = {
            "name": facts.name,
            "email": facts.email,
            "phone": facts.phone,
            "linkedin": facts.linkedin_url,
            "portfolio": facts.portfolio_url,
            "location": facts.location,
        }.get(key, "")
        if form_field.input_type == "file" or "resume" in key or "cv" in key or "cover" in key:
            reason = "File upload must be explicitly approved."
            proposals.append(FieldProposal(form_field.field_id, form_field.label, sensitive=False, reason=reason))
        elif form_field.sensitive:
            proposals.append(
                FieldProposal(
                    form_field.field_id,
                    form_field.label,
                    sensitive=True,
                    reason="Sensitive question: ask the user one application at a time.",
                )
            )
        elif value:
            proposals.append(
                FieldProposal(
                    form_field.field_id, form_field.label, value=value, confidence=0.98, reason="Explicit candidate fact."
                )
            )
        else:
            proposals.append(
                FieldProposal(
                    form_field.field_id, form_field.label, sensitive=False, reason="Unknown field: pause for human input."
                )
            )
    return proposals


def _field_key(*values: str) -> str:
    text = " ".join(values).lower()
    if re.search(r"linkedin", text):
        return "linkedin"
    if re.search(r"portfolio|personal website|website", text):
        return "portfolio"
    if re.search(r"e.?mail", text):
        return "email"
    if re.search(r"phone|mobile|telephone", text):
        return "phone"
    if re.search(r"name", text):
        return "name"
    if re.search(r"city|location|address", text):
        return "location"
    return text.strip()


def _is_sensitive(text: str) -> bool:
    pattern = r"sponsor|authorization|visa|salary|criminal|disability|gender|race|password|ssn|security"
    return bool(re.search(pattern, text.lower()))
