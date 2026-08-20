"""Gradio UI for Job Scout.

A four-step wizard with a refined editorial look: warm-paper background with a
faint gradient-and-grain wash, a display serif (Fraunces) paired with a mono for
data (IBM Plex Mono), emerald accent. The flow is (1) drop your resume,
(2) review the extracted profile and choose where to search (the human decides
locations and remote, not the model), (3) find jobs — ranked as cards with a
conic-gauge fit score, matched-skill chips, and honest gaps — and (4) tailor an
application for a selected job: cover letter + reworded CV, every claim checked
against the resume, downloadable as PDF/.tex.

Jobvis, the voice concierge, is a separate surface now (job_scout.api serves it
on :8000) — the wizard just links to it. What stays here is the sync: a run
started by voice lands on these pages through the 1s Timer, and a click here
whispers a screen event to whoever is listening on the console.
"""

from __future__ import annotations

import re
import socket
import tempfile
from datetime import date
from html import escape, unescape
from pathlib import Path
from uuid import uuid4

import gradio as gr

from job_scout import candidate_store
from job_scout.graph.schemas import CandidatePreferences, CVLink, Profile, RankedJob
from job_scout.profile import extract_profile
from job_scout.renderer import render_cover_letter_pdf, render_pdf
from job_scout.runner import RunResult, TailorResult, stream_search, stream_tailor
from job_scout.tools.cv_reader import CVReadError
from job_scout.tracing import opik_url, register_prompts
from job_scout.voice import bridge as voice_bridge
from job_scout.voice import is_voice_available

CAPTION = "Prepares applications — never submits them."

INTRO_HTML = """
<p class="js-lead">Drop your resume and let Job Scout find roles that actually match it —
ranked by fit, with the gaps shown honestly.</p>
<ol class="js-steps">
  <li><span class="js-num">1</span><div><b>Drop your resume</b> (PDF) below.</div></li>
  <li><span class="js-num">2</span><div>We <b>read it</b> and show your skills, experience, and locations.</div></li>
  <li><span class="js-num">3</span><div>Hit <b>Find jobs</b> to get openings ranked by fit.</div></li>
</ol>
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=(gr.themes.GoogleFont("Public Sans"), "ui-sans-serif", "system-ui", "sans-serif"),
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#FBFAF8",
    body_background_fill_dark="#141311",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#1E1D1A",
    button_primary_background_fill="#0E6E4A",
    button_primary_background_fill_hover="#0A5539",
    button_primary_text_color="#FFFFFF",
    button_large_radius="12px",
)

# Fine tiled grain, kept as a constant so the CSS block stays within line length.
_GRAIN = "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.5'/%3E%3C/svg%3E\")"  # noqa: E501

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
  --js-accent: #0E6E4A;
  --js-accent-bright: #0E9C68;
  --js-ring-track: rgba(20,19,17,0.08);
}
.dark {
  --js-ring-track: rgba(255,255,255,0.10);
}

/* --- Atmospheric background: faint emerald + amber wash over warm paper --- */
.gradio-container {
  max-width: 760px !important;
  margin: 0 auto !important;
  position: relative;
}
body::before {
  content: "";
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(60% 45% at 12% 4%, rgba(14,156,104,0.10), transparent 70%),
    radial-gradient(55% 45% at 92% 8%, rgba(180,83,9,0.07), transparent 72%);
}
.dark body::before {
  background:
    radial-gradient(60% 45% at 12% 4%, rgba(14,156,104,0.14), transparent 70%),
    radial-gradient(55% 45% at 92% 8%, rgba(180,83,9,0.10), transparent 72%);
}
/* fine grain */
body::after {
  content: "";
  position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.35;
  background-image: __GRAIN__;
}
.dark body::after { opacity: 0.5; mix-blend-mode: screen; }
.gradio-container > * { position: relative; z-index: 1; }

/* --- Header --- */
#js-header { text-align: center; padding: 22px 0 4px; }
#js-header .js-mark {
  display: inline-flex; align-items: center; justify-content: center; gap: 10px;
}
#js-header .js-mark svg { width: 30px; height: 30px; color: var(--js-accent); }
#js-header h1 {
  font-family: 'Fraunces', Georgia, serif; font-weight: 600;
  font-size: 2.85rem; letter-spacing: -0.025em; margin: 0; line-height: 1.02;
}
#js-header .js-tag {
  display: inline-block; margin-top: 13px; font-size: 0.76rem; letter-spacing: 0.01em;
  color: var(--body-text-color-subdued);
  border: 1px solid var(--border-color-primary); border-radius: 999px; padding: 4px 14px;
  background: color-mix(in srgb, var(--block-background-fill) 55%, transparent);
  backdrop-filter: blur(4px);
}

/* --- Stepper --- */
.js-stepper {
  list-style: none; display: flex; align-items: center; justify-content: center;
  gap: 0; margin: 20px auto 6px; padding: 0; max-width: 420px;
}
.js-step { display: flex; align-items: center; gap: 9px; font-size: 0.82rem; }
.js-step-dot {
  width: 26px; height: 26px; border-radius: 50%; flex: none;
  display: grid; place-items: center; font-size: 0.78rem; font-weight: 700;
  font-family: 'IBM Plex Mono', monospace;
  border: 1.5px solid var(--border-color-primary); color: var(--body-text-color-subdued);
  background: var(--block-background-fill); transition: all .2s ease;
}
.js-step-label { color: var(--body-text-color-subdued); font-weight: 500; }
.js-step::after {
  content: ""; width: 34px; height: 1.5px; margin: 0 12px;
  background: var(--border-color-primary); border-radius: 2px;
}
.js-step:last-child::after { display: none; }
.js-step-active .js-step-dot {
  border-color: var(--js-accent); color: #fff; background: var(--js-accent);
  box-shadow: 0 0 0 4px rgba(14,110,74,0.14);
}
.js-step-active .js-step-label { color: var(--body-text-color); font-weight: 600; }
.js-step-done .js-step-dot {
  border-color: var(--js-accent); color: var(--js-accent);
  background: rgba(14,110,74,0.12);
}

/* --- Intro --- */
.js-lead { text-align: center; font-size: 1.05rem; line-height: 1.55; color: var(--body-text-color-subdued);
  margin: 14px auto 4px; max-width: 480px; }
.js-steps { list-style: none; padding: 0; margin: 18px auto 8px; max-width: 470px; }
.js-steps li { display: flex; gap: 13px; align-items: flex-start; padding: 9px 0; font-size: 0.95rem; line-height: 1.45; }
.js-steps .js-num { flex: none; width: 26px; height: 26px; border-radius: 50%;
  background: rgba(14,110,74,0.12); color: var(--js-accent); font-weight: 700; font-size: 0.82rem;
  font-family: 'IBM Plex Mono', monospace; margin-top: 1px;
  display: flex; align-items: center; justify-content: center; }

.js-section-label { font-size: 0.76rem; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--body-text-color-subdued); margin: 4px 0 10px 2px; }
.js-muted { color: var(--body-text-color-subdued); }

/* --- Dropzone polish --- */
.js-drop { border: 1.5px dashed var(--border-color-primary) !important;
  border-radius: 16px !important; background: color-mix(in srgb, var(--block-background-fill) 70%, transparent) !important;
  transition: border-color .2s ease, background .2s ease; }
.js-drop:hover { border-color: var(--js-accent) !important; }

/* --- Cards --- */
.js-card { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-radius: 16px; padding: 20px 22px; box-shadow: 0 1px 2px rgba(20,19,17,0.03); }
.js-profile-name { font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
.js-profile-sub { color: var(--body-text-color-subdued); font-size: 0.92rem; margin: 3px 0 14px; }
.js-profile-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 4px 0 14px;
  border-top: 1px solid var(--border-color-primary); padding-top: 14px; }
.js-stat .js-stat-val { font-family: 'IBM Plex Mono', monospace; font-size: 1.15rem; font-weight: 600; line-height: 1; }
.js-stat .js-stat-key { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--body-text-color-subdued); margin-top: 5px; }
.js-profile-row { font-size: 0.9rem; margin: 3px 0; }
.js-profile-row b { font-weight: 600; }
.js-pill { display: inline-block; font-size: 0.75rem; padding: 3px 11px; border-radius: 999px;
  background: rgba(14,110,74,0.10); color: var(--js-accent); margin: 5px 5px 0 0; }

/* --- Job cards --- */
.js-jobs { display: flex; flex-direction: column; gap: 12px; }
.js-job { border: 1px solid var(--border-color-primary); border-radius: 14px;
  padding: 17px 19px; background: var(--block-background-fill);
  box-shadow: 0 1px 2px rgba(20,19,17,0.03);
  transition: box-shadow .18s ease, transform .18s ease, border-color .18s ease;
  opacity: 0; transform: translateY(8px); animation: js-rise .4s ease forwards; }
.js-job:hover { box-shadow: 0 12px 30px rgba(20,19,17,0.09); transform: translateY(-2px);
  border-color: color-mix(in srgb, var(--js-accent) 40%, var(--border-color-primary)); }
@keyframes js-rise { to { opacity: 1; transform: none; } }

.js-job-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.js-job-title { font-weight: 600; font-size: 1.06rem; text-decoration: none; color: var(--body-text-color);
  line-height: 1.3; }
.js-job-title:hover { color: var(--js-accent); }
.js-job-meta { font-size: 0.85rem; color: var(--body-text-color-subdued); margin-top: 4px;
  display: flex; align-items: center; flex-wrap: wrap; gap: 7px; }
.js-remote, .js-source { font-size: 0.66rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
  border-radius: 5px; padding: 2px 7px; line-height: 1.5; }
.js-remote { color: var(--js-accent); border: 1px solid rgba(14,110,74,0.35); }
.js-source { color: var(--body-text-color-subdued); border: 1px solid var(--border-color-primary); }
.js-job-why { font-size: 0.92rem; line-height: 1.55; margin: 12px 0 0; }

/* chips */
.js-chips { margin-top: 11px; display: flex; flex-wrap: wrap; gap: 6px; }
.js-chip { display: inline-flex; align-items: center; gap: 4px; font-size: 0.75rem;
  padding: 3px 10px; border-radius: 999px; line-height: 1.5; }
.js-chip-match { background: rgba(14,110,74,0.11); color: var(--js-accent); }
.js-chip-gap { background: rgba(100,116,139,0.13); color: var(--body-text-color-subdued); }

/* --- Fit gauge --- */
.js-fit-ring { --size: 56px; width: var(--size); height: var(--size); flex: none;
  border-radius: 50%; display: grid; place-items: center; position: relative;
  background: conic-gradient(var(--ring) calc(var(--val) * 1%), var(--js-ring-track) 0); }
.js-fit-ring::before { content: ""; position: absolute; inset: 5px; border-radius: 50%;
  background: var(--block-background-fill); }
.js-fit-num { position: relative; font-family: 'IBM Plex Mono', monospace; font-weight: 600;
  font-size: 1.05rem; color: var(--tier-fg); }
.js-fit-high { --ring: var(--js-accent-bright); --tier-fg: #0E6E4A; }
.js-fit-mid  { --ring: #D97706; --tier-fg: #B45309; }
.js-fit-low  { --ring: #94A3B8; --tier-fg: #64748B; }
.dark .js-fit-high { --tier-fg: #34D399; }
.dark .js-fit-mid  { --tier-fg: #FBBF24; }
.dark .js-fit-low  { --tier-fg: #94A3B8; }

/* --- States --- */
.js-empty { text-align: center; padding: 40px 20px; color: var(--body-text-color-subdued); }
.js-empty .js-empty-icon { font-size: 1.8rem; opacity: 0.8; margin-bottom: 8px; }
.js-status { font-size: 0.86rem; color: var(--body-text-color-subdued); text-align: center; min-height: 18px; }
.js-status-err { color: #B45309; }
.js-spin { display: inline-block; width: 13px; height: 13px; margin-right: 8px; vertical-align: -1px;
  border: 2px solid rgba(14,110,74,0.25); border-top-color: var(--js-accent); border-radius: 50%;
  animation: js-rotate .7s linear infinite; }
@keyframes js-rotate { to { transform: rotate(360deg); } }

/* --- Footer --- */
.js-footer { text-align: center; font-size: 0.82rem; color: var(--body-text-color-subdued);
  border-top: 1px solid var(--border-color-primary); margin-top: 20px; padding-top: 15px; }
.js-footer .js-meta-mono { font-family: 'IBM Plex Mono', monospace; }
.js-footer a { color: var(--js-accent); text-decoration: none; }
.js-footer a:hover { text-decoration: underline; }

/* --- Tailor pack (Phase 2) --- */
.js-pack { display: flex; flex-direction: column; gap: 14px; }
.js-letter { white-space: pre-wrap; font-size: 0.93rem; line-height: 1.6; }
.js-cvprev-role { font-weight: 600; margin: 10px 0 2px; }
.js-cvprev-dates { color: var(--body-text-color-subdued); font-size: 0.85rem; font-weight: 400; }
.js-cvprev ul { margin: 4px 0 0; padding-left: 1.2em; }
.js-cvprev li { font-size: 0.9rem; line-height: 1.5; margin: 3px 0; }
.js-ref { font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; color: var(--js-accent);
  background: rgba(14,110,74,0.09); border-radius: 5px; padding: 1px 6px; margin-left: 6px;
  vertical-align: 1px; white-space: nowrap; }
.js-honesty { border-left: 3px solid #D97706; background: rgba(217,119,6,0.07);
  border-radius: 0 12px 12px 0; padding: 12px 16px; font-size: 0.9rem; line-height: 1.55; }
.js-honesty b { color: #B45309; }
.js-fab-ok { border: 1px solid rgba(14,110,74,0.35); background: rgba(14,110,74,0.06);
  border-radius: 12px; padding: 11px 16px; font-size: 0.88rem; color: var(--js-accent); }
.js-fab-warn { border: 1px solid rgba(217,119,6,0.45); background: rgba(217,119,6,0.07);
  border-radius: 12px; padding: 12px 16px; font-size: 0.88rem; }
.js-fab-warn b { color: #B45309; }
.js-fab-warn ul { margin: 8px 0 0; padding-left: 1.2em; }
.js-fab-warn li { margin: 4px 0; line-height: 1.5; }
.js-fab-warn .js-fab-reason { color: var(--body-text-color-subdued); font-size: 0.8rem; }

/* --- Strip the Gradio chrome: pages float directly on the paper --- */
footer { display: none !important; }
.js-page { background: transparent !important; border: none !important; box-shadow: none !important; }
.js-page > .styler, .js-page .form { background: transparent !important; border: none !important; box-shadow: none !important; }

/* --- Jobvis: a one-line pointer at the voice console on :8000 --- */
#jv-strip { display: flex; align-items: center; justify-content: center; gap: 12px;
  background: linear-gradient(140deg, #1A241F 0%, #101713 100%);
  border: 1px solid rgba(14,156,104,0.28); border-radius: 18px;
  padding: 12px 18px; margin: 2px 0 10px; box-shadow: 0 14px 40px rgba(10, 20, 16, 0.28); }
#jv-strip .jv-hint { margin: 0; color: rgba(236,242,238,0.72); }
.jv-hint { text-align: center; font-size: 0.78rem; color: var(--body-text-color-subdued); margin: 6px 0 0; }
.jv-link { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: #7DE3B5 !important;
  text-decoration: none; white-space: nowrap; }
.jv-link:hover { text-decoration: underline; }

/* accessibility: visible focus */
a:focus-visible, button:focus-visible, .js-job-title:focus-visible {
  outline: 2px solid var(--js-accent); outline-offset: 2px; border-radius: 4px; }

@media (max-width: 520px) {
  #js-header h1 { font-size: 2.3rem; }
  .js-step-label { display: none; }
  .js-step::after { width: 22px; margin: 0 8px; }
  .js-profile-grid { grid-template-columns: repeat(3, 1fr); gap: 8px; }
}
@media (prefers-reduced-motion: reduce) {
  .js-job { animation: none; opacity: 1; transform: none; }
  .js-spin { animation: none; }
}
""".replace("__GRAIN__", _GRAIN)

# Small compass mark for the wordmark.
_MARK = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<circle cx="12" cy="12" r="9"/><path d="m15.5 8.5-2.1 5-5 2.1 2.1-5z"/></svg>'
)


def _stepper(active: int) -> str:
    """Render the 4-step progress indicator with ``active`` (1-4) highlighted."""
    labels = ("Resume", "Profile", "Jobs", "Tailor")
    items = []
    for i, label in enumerate(labels, 1):
        state = "done" if i < active else "active" if i == active else "todo"
        mark = "✓" if state == "done" else str(i)
        items.append(
            f'<li class="js-step js-step-{state}">'
            f'<span class="js-step-dot">{mark}</span>'
            f'<span class="js-step-label">{label}</span></li>'
        )
    return f'<ol class="js-stepper">{"".join(items)}</ol>'


def _loading_html(text: str) -> str:
    """Centered spinner with a message."""
    return f'<div class="js-empty"><div class="js-empty-icon"><span class="js-spin"></span></div><div>{escape(text)}</div></div>'


def _chips(items: list[str], kind: str, limit: int) -> str:
    """Render a row of matched-skill or gap chips."""
    icon = "✓ " if kind == "match" else ""
    shown = "".join(f'<span class="js-chip js-chip-{kind}">{icon}{escape(s)}</span>' for s in items[:limit])
    return f'<div class="js-chips">{shown}</div>' if shown else ""


def _profile_html(profile: Profile | None, cv_links: list[CVLink] | None = None) -> str:
    """Render the extracted profile as a card."""
    if profile is None:
        return '<div class="js-card js-muted">Could not read a profile from that resume.</div>'
    skills = "".join(f'<span class="js-pill">{escape(s)}</span>' for s in profile.skills) or "—"
    years = f"{profile.years_experience:g}" if profile.years_experience else "—"
    languages = escape(", ".join(profile.languages) or "—")
    links = list(cv_links or [])
    links_html = ""
    if links:
        rows = "".join(
            f'<li><a href="{escape(link.url)}" target="_blank" rel="noopener">{escape(link.label)}</a>'
            f' <span class="js-muted">page {link.page}</span></li>'
            for link in links
        )
        links_html = (
            '<div style="margin-top:14px"><b>Detected resume links</b>'
            '<p class="js-muted" style="margin:4px 0 6px;font-size:0.82rem">'
            "These are preserved in the generated CV. Check the labels before tailoring.</p>"
            f"<ul>{rows}</ul></div>"
        )
    education_summary = (
        "; ".join((entry.degree + " " + entry.field + " at " + entry.institution).strip() for entry in profile.education_history)
        or "Not extracted — review the resume"
    )
    graduation_summary = str(profile.expected_graduation_date or "Unknown")
    return (
        '<div class="js-card">'
        f'<p class="js-profile-name">{escape(profile.name or "Candidate")}</p>'
        f'<p class="js-profile-sub">{escape(profile.seniority.title())} · '
        f"{escape(', '.join(profile.primary_roles) or '—')}</p>"
        '<div class="js-profile-grid">'
        f'<div class="js-stat"><div class="js-stat-val">{years}</div><div class="js-stat-key">Years exp</div></div>'
        f'<div class="js-stat"><div class="js-stat-val">{"Yes" if profile.remote_ok else "No"}</div>'
        '<div class="js-stat-key">Remote</div></div>'
        f'<div class="js-stat"><div class="js-stat-val">{len(profile.skills)}</div><div class="js-stat-key">Skills</div></div>'
        "</div>"
        f'<p class="js-profile-row"><b>Locations</b> &nbsp;{escape(", ".join(profile.locations) or "—")}</p>'
        f'<p class="js-profile-row"><b>Education timeline</b> &nbsp;{escape(education_summary)}</p>'
        f'<p class="js-profile-row"><b>Expected graduation</b> &nbsp;{escape(graduation_summary)}</p>'
        f'<p class="js-profile-row"><b>Languages</b> &nbsp;{languages}</p>'
        f'<div style="margin-top:12px">{skills}</div>{links_html}'
        "</div>"
    )


def _fit_class(score: int) -> str:
    """Return the CSS tier class for a fit score."""
    if score >= 80:
        return "js-fit-high"
    if score >= 60:
        return "js-fit-mid"
    return "js-fit-low"


def _fit_ring(score: int) -> str:
    """Render the fit score as a conic-gradient gauge."""
    return (
        f'<div class="js-fit-ring {_fit_class(score)}" style="--val:{score}" '
        f'role="img" aria-label="Fit score {score} out of 100">'
        f'<span class="js-fit-num">{score}</span></div>'
    )


def _job_card(ranked: RankedJob, index: int) -> str:
    """Render one ranked job as a card with a conic fit gauge and skill chips."""
    job = ranked.job
    title = escape(job.title)
    if job.url:
        title_html = f'<a class="js-job-title" href="{escape(job.url)}" target="_blank" rel="noopener">{title}</a>'
    else:
        title_html = f'<span class="js-job-title">{title}</span>'
    remote = '<span class="js-remote">Remote</span>' if job.remote else ""
    source = f'<span class="js-source">{escape(job.source)}</span>'
    matched = _chips(ranked.matched_skills, "match", 6)
    gaps = _chips(ranked.gaps, "gap", 4)
    status = escape(ranked.eligibility_status)
    bucket = escape(ranked.primary_or_adjacent)
    reasons = ranked.hard_blockers or ranked.eligibility_reasons
    policy = "".join(f"<li>{escape(reason)}</li>" for reason in reasons[:3])
    policy_html = f'<ul class="js-muted" style="margin:8px 0 0;padding-left:18px">{policy}</ul>' if policy else ""
    risk_bits = []
    if job.clearance_required:
        risk_bits.append("clearance required")
    if job.authorization_requirement != "unknown":
        risk_bits.append("authorization language needs confirmation")
    if job.sponsorship_signal != "unknown":
        risk_bits.append("sponsorship language needs confirmation")
    risk_html = (
        f'<p class="js-muted" style="font-size:0.8rem;margin:5px 0 0">⚠ {escape(" · ".join(risk_bits))}</p>' if risk_bits else ""
    )
    freshness = f"posted {job.posted_at}" if job.posted_at else "posted date unknown"
    salary = f" · {job.salary_text}" if job.salary_text else ""
    delay = f"animation-delay:{min(index, 8) * 55}ms"
    return (
        f'<div class="js-job" style="{delay}">'
        '<div class="js-job-head">'
        f"<div>{title_html}"
        f'<div class="js-job-meta">{escape(job.company)} · {escape(job.location)}{remote}{source}'
        f' <span class="js-source">{bucket} · {status}</span></div></div>'
        f"{_fit_ring(ranked.fit_score)}"
        "</div>"
        f'<p class="js-job-why">{escape(ranked.fit_explanation)}</p>'
        f'<p class="js-muted" style="font-size:0.8rem;margin:8px 0 0">'
        f"Role fit {ranked.role_fit_score} · evidence fit {ranked.evidence_fit_score} · "
        f"{escape(ranked.job.employment_type)} · {escape(ranked.job.work_mode)} · start {escape(ranked.start_timing_fit)} · "
        f"{escape(freshness)}{escape(salary)}</p>"
        f"{risk_html}{policy_html}"
        f"{matched}{gaps}"
        "</div>"
    )


def _results_html(result: RunResult) -> str:
    """Render the ranked jobs as cards, or an empty state."""
    if not result.ranked_jobs:
        detail = (
            "The search did not complete. Check the run status below and try again after fixing the provider."
            if result.failed
            else (
                "Jobs were fetched, but none produced a usable ranking. Check source diagnostics below."
                if result.n_jobs_fetched
                else "No sources returned postings for this policy. Check source diagnostics below."
            )
        )
        return f'<div class="js-empty"><div class="js-empty-icon">🔍</div><div>{escape(detail)}</div></div>'
    primary = [r for r in result.ranked_jobs if r.primary_or_adjacent == "primary" and r.eligibility_status != "blocked"]
    adjacent = [r for r in result.ranked_jobs if r.primary_or_adjacent == "adjacent" and r.eligibility_status != "blocked"]
    blocked = [r for r in result.ranked_jobs if r.eligibility_status == "blocked" or r.primary_or_adjacent == "review"]

    def section(label: str, jobs: list[RankedJob], empty: str) -> str:
        cards = "".join(_job_card(r, i) for i, r in enumerate(jobs))
        body = f'<div class="js-jobs">{cards}</div>' if jobs else f'<div class="js-muted">{empty}</div>'
        return f'<p class="js-section-label" style="margin-top:18px">{label} ({len(jobs)})</p>{body}'

    sections = [
        section("Primary matches", primary, "No eligible primary matches yet."),
        section("Adjacent roles", adjacent, "No adjacent roles returned."),
        section("Blocked or review-required", blocked, "None."),
    ]
    summary = (
        f'<div class="js-card" style="margin-bottom:14px"><b>{result.n_jobs_fetched} fetched</b> · '
        f"{len(primary) + len(adjacent)} eligible · {len(primary)} primary · {len(adjacent)} adjacent · "
        f"{len(blocked)} blocked/review</div>"
    )
    return summary + "".join(sections)


def _footer_html(result: RunResult) -> str:
    """Render the run footer: cost, latency, job source, and the Opik link."""
    link = f'<a href="{result.opik_url or opik_url()}" target="_blank" rel="noopener">view traces in Opik ↗</a>'
    if result.failed:
        body = f"⚠ run failed — {escape(result.error_message)}. The trace has details · {link}"
    else:
        sources = escape(", ".join(result.jobs_sources) or "none")
        diagnostics = " · ".join(
            f"{d.source}: {d.latency_ms:.0f}ms/{d.returned}{' timeout' if d.timed_out else ''}" for d in result.source_diagnostics
        )
        sep = ' <span class="js-muted">·</span> '
        body = (
            f'<span class="js-meta-mono">${result.cost_usd:.4f}</span>{sep}'
            f'<span class="js-meta-mono">{result.latency_s}s</span>{sep}'
            f"source: {sources}"
            + (f'{sep}<span class="js-muted">{escape(diagnostics)}</span>' if diagnostics else "")
            + f"{sep}{link}"
        )
    return f'<div class="js-footer">{body}</div>'


def _cv_preview_html(pack) -> str:
    """Render the tailored CV with a human-readable grounding badge."""
    cv = pack.cv
    parts = [f"<p><b>{escape(cv.headline)}</b></p>", f'<p class="js-job-why">{escape(cv.summary)}</p>']
    for entry in cv.experience:
        dates = f' <span class="js-cvprev-dates">{escape(entry.dates)}</span>' if entry.dates else ""
        parts.append(f'<p class="js-cvprev-role">{escape(entry.role)} — {escape(entry.company)}{dates}</p>')
        if entry.bullets:
            bullets = "".join(f"<li>{escape(b.text)}<span class='js-ref'>✓ verified evidence</span></li>" for b in entry.bullets)
            parts.append(f"<ul>{bullets}</ul>")
    if cv.projects:
        parts.append('<p class="js-section-label">Selected projects</p>')
        for entry in cv.projects:
            dates = f' <span class="js-cvprev-dates">{escape(entry.dates)}</span>' if entry.dates else ""
            parts.append(f'<p class="js-cvprev-role">{escape(entry.role)} — {escape(entry.company)}{dates}</p>')
            if entry.bullets:
                bullets = "".join(
                    f"<li>{escape(b.text)}<span class='js-ref'>✓ verified evidence</span></li>" for b in entry.bullets
                )
                parts.append(f"<ul>{bullets}</ul>")
    if cv.skills:
        pills = "".join(f'<span class="js-pill">{escape(s)}</span>' for s in cv.skills)
        parts.append(f'<div style="margin-top:10px">{pills}</div>')
    if cv.education:
        rows = "".join(f"<li>{escape(line)}</li>" for line in cv.education)
        parts.append(f"<ul>{rows}</ul>")
    if cv.links:
        links = " · ".join(
            f'<a href="{escape(link.url)}" target="_blank" rel="noopener">{escape(link.label)}</a>' for link in cv.links
        )
        parts.append(f'<p class="js-muted"><b>Preserved links:</b> {links}</p>')
    return f'<div class="js-card js-cvprev">{"".join(parts)}</div>'


def _fabrication_html(result: TailorResult) -> str:
    """Render the validator verdict — never hidden, whatever it says."""
    if result.fabrication_flags == 0:
        return '<div class="js-fab-ok">✓ Every claim in this application traced back to your CV.</div>'
    report = result.fabrication_report
    summary = ""
    if report:
        summary = (
            f"<div class='js-muted' style='margin-top:6px'>"
            f"{report.confirmed_claims} confirmed · {report.near_miss_claims} near-miss · "
            f"{report.unsupported_claims} unsupported</div>"
        )
    items = ""
    if report:
        items = "".join(
            f"<li>“{escape(f.text[:140])}”<br><span class='js-fab-reason'>{escape(f.reason)}</span></li>" for f in report.flagged
        )
    n = result.fabrication_flags
    return (
        f'<div class="js-fab-warn"><b>⚠ {n} statement{"s" if n != 1 else ""} could not be verified against your CV.</b>'
        f" Review before sending.{summary}<ul>{items}</ul></div>"
    )


def _pack_html(result: TailorResult) -> str:
    """Render the application pack: cover letter, tailored CV, honesty note."""
    if result.pack is None:
        why = escape("; ".join(result.errors) or result.error_message or "unknown error")
        return f'<div class="js-empty"><div class="js-empty-icon">⚠</div><div>Could not tailor this job: {why}</div></div>'
    pack = result.pack
    quality = result.cover_letter_quality
    quality_html = ""
    if quality is not None and not quality.passed:
        policy_detail = ""
        if quality.policy_violations:
            policy_detail = (
                "<br><span class='js-fab-reason'>Unconfirmed policy claim(s): "
                + escape(" | ".join(quality.policy_violations[:2]))
                + "</span>"
            )
        quality_html = (
            '<div class="js-fab-warn"><b>⚠ Cover-letter quality gate needs review.</b>'
            f" The draft has {quality.word_count} words, {quality.evidence_matches} evidence points, and "
            f"{quality.requirement_matches} matched requirements. {escape('; '.join(quality.reasons))}"
            f"<br><span class='js-fab-reason'>Gate targets: {escape(' | '.join(quality.requirement_targets[:2]))}</span>"
            f"{policy_detail}</div>"
        )
    honesty = ""
    if pack.honesty_note:
        honesty = f'<div class="js-honesty"><b>Honesty note</b> — {escape(pack.honesty_note)}</div>'
    letter = re.sub(r"(?is)<br\s*/?>", "\n", pack.cover_letter)
    letter = re.sub(r"(?is)</(?:p|div|li|h[1-6])\s*>", "\n", letter)
    letter = re.sub(r"(?is)<[^>]+>", "", letter)
    letter_html = escape(unescape(letter)).replace("\n", "<br>")
    return (
        '<div class="js-pack">'
        f"{_fabrication_html(result)}"
        f"{quality_html}"
        '<p class="js-section-label">Cover letter</p>'
        f'<div class="js-card js-letter">{letter_html}</div>'
        '<p class="js-section-label">Tailored CV</p>'
        f"{_cv_preview_html(pack)}"
        f"{honesty}"
        "</div>"
    )


def _tailor_footer_html(result: TailorResult) -> str:
    """Run footer for the tailor step: cost, latency, research flag, Opik link."""
    link = f'<a href="{result.opik_url or opik_url()}" target="_blank" rel="noopener">view traces in Opik ↗</a>'
    if result.failed:
        body = f"⚠ run failed — {escape(result.error_message)}. The trace has details · {link}"
    else:
        sep = ' <span class="js-muted">·</span> '
        research = "with company research" if result.research_used else "no company research"
        body = (
            f'<span class="js-meta-mono">${result.cost_usd:.4f}</span>{sep}'
            f'<span class="js-meta-mono">{result.latency_s}s</span>{sep}'
            f"{research}{sep}{link}"
        )
    return f'<div class="js-footer">{body}</div>'


def _status(text: str, error: bool = False) -> str:
    """Render a status line on the start page."""
    cls = "js-status js-status-err" if error else "js-status"
    return f'<div class="{cls}">{escape(text)}</div>'


def _pack_downloads(result: TailorResult, profile: Profile | None) -> tuple[dict, dict, dict, dict, str]:
    """Footer + download-button updates for a finished tailoring run (click or voice)."""
    hidden = gr.update(visible=False)
    footer = _tailor_footer_html(result)
    pdf_btn, tex_btn, letter_pdf_btn, letter_tex_btn = hidden, hidden, hidden, hidden
    if result.pack is not None:
        name = (profile.name if profile else None) or "Candidate"
        out_dir = Path(tempfile.mkdtemp(prefix="job_scout_render_"))
        cv_render = render_pdf(result.pack.cv, name, out_dir)
        letter_render = render_cover_letter_pdf(result.pack.cover_letter, name, out_dir)
        tex_btn = gr.update(value=str(cv_render.tex_path), visible=True)
        letter_tex_btn = gr.update(value=str(letter_render.tex_path), visible=True)
        if cv_render.pdf_path is not None:
            pdf_btn = gr.update(value=str(cv_render.pdf_path), visible=True)
        if letter_render.pdf_path is not None:
            letter_pdf_btn = gr.update(value=str(letter_render.pdf_path), visible=True)
        messages = list(dict.fromkeys(message for message in (cv_render.message, letter_render.message) if message))
        if messages:
            guidance = "PDF downloads are unavailable until Tectonic is installed. "
            guidance += "Run `brew install tectonic`, restart Job Scout, and tailor again. "
            guidance += "The .tex files are still available for Overleaf."
            footer = f'<div class="js-footer">{escape(guidance)}</div>{footer}'
    return pdf_btn, tex_btn, letter_pdf_btn, letter_tex_btn, footer


def _voice_context(text: str) -> None:
    """Tell a listening console what just happened on screen (silent context)."""
    voice_bridge.get_bridge().note_screen_event(text)


def on_run_tick():
    """Mirror a voice-triggered run into the wizard's pages.

    The console holds the conversation, but the click path still owns these
    screens, so a run started by voice has to land here too. While a run is IN
    FLIGHT this shows the runner's live status lines ("ranking 12 jobs…") so
    the page never sits frozen; when the bridge hands over the finished run,
    the SAME renderers the click path uses fill it in. Announcing the result is
    the console's job now — it hears the same run on its own feed.
    """
    no = gr.update()
    bridge = voice_bridge.get_bridge()
    run = bridge.pop_finished_run()
    if run is None:
        progress = bridge.run_status()
        if progress.get("running"):
            loading = _loading_html(str(progress.get("latest_status") or "working…"))
            if progress.get("kind") == "search":
                show_results = (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False))
                return (*show_results, loading, "", no, no, no, no, no, no, no)
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                no,
                no,
                no,
                loading,
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )
    if run is not None and run.kind == "search" and run.search_result is not None and not run.failed:
        result = run.search_result
        choices = [
            (f"{r.job.title} — {r.job.company} (fit {r.fit_score})", r.job.job_id)
            for r in result.ranked_jobs
            if r.eligibility_status != "blocked"
        ]
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            _results_html(result),
            _footer_html(result),
            gr.update(choices=choices, value=None, visible=bool(choices)),
            no,
            no,
            no,
            no,
            no,
            no,
        )
    if run is not None and run.kind == "tailor" and run.tailor_result is not None and not run.failed:
        result = run.tailor_result
        pdf_btn, tex_btn, letter_pdf_btn, letter_tex_btn, footer = _pack_downloads(result, bridge.snapshot().profile)
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            no,
            no,
            no,
            _pack_html(result),
            footer,
            pdf_btn,
            tex_btn,
            letter_pdf_btn,
            letter_tex_btn,
        )
    return (no,) * 12


REMOTE_CHOICE = "Remote (anywhere)"
US_SCOPE_CHOICE = "Anywhere in the United States"


def _is_us_scope(location: str) -> bool:
    """Whether a chooser value means the whole US, not a literal city."""
    return location.strip().lower() in {"us", "usa", "united states", "anywhere in the united states"}


def _selection_to_prefs(selection: list[str]) -> dict:
    """The chooser's ticked boxes as a preferences dict."""
    us_scope = any(_is_us_scope(s) or s == US_SCOPE_CHOICE for s in selection)
    locations = [s for s in selection if s != REMOTE_CHOICE and not _is_us_scope(s) and s != US_SCOPE_CHOICE]
    preferences = {"locations": [] if us_scope else locations, "remote": REMOTE_CHOICE in selection}
    if us_scope:
        preferences["country_scope"] = "us"
    return preferences


def _parse_policy_date(value: object, fallback: date | None) -> date | None:
    """Parse a date control without letting a typo erase the saved policy."""
    if value is None or not str(value).strip():
        return fallback
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return fallback


def _preferences_from_ui(
    selection: list[str],
    raw: dict | None,
    employment_types: list[str] | None = None,
    target_start_min: object = None,
    target_start_max: object = None,
    accepted_work_modes: list[str] | None = None,
    primary_role_families: list[str] | None = None,
    authorization_status: str | None = None,
    sponsorship_policy: str | None = None,
    clearance_status: str | None = None,
) -> tuple[dict, CandidatePreferences | None]:
    """Combine the location chooser with the typed intent editor.

    ``raw is None`` preserves the legacy handler contract used by notebooks and
    tests. The browser UI supplies the typed editor, which activates the
    candidate-aware search path.
    """
    legacy = _selection_to_prefs(selection)
    if not raw:
        return legacy, None
    try:
        typed = CandidatePreferences.model_validate(raw)
    except Exception:
        typed = CandidatePreferences()
    typed = typed.model_copy(
        update={
            "locations": legacy["locations"],
            "country_scope": "us" if legacy.get("country_scope") == "us" else "selected",
        }
    )
    updates: dict[str, object] = {}
    if employment_types is not None:
        updates["employment_types"] = employment_types
    if target_start_min is not None:
        updates["target_start_min"] = _parse_policy_date(target_start_min, typed.target_start_min)
    if target_start_max is not None:
        updates["target_start_max"] = _parse_policy_date(target_start_max, typed.target_start_max)
    if accepted_work_modes is not None:
        updates["accepted_work_modes"] = accepted_work_modes
    if primary_role_families is not None:
        updates["primary_role_families"] = primary_role_families
    if authorization_status is not None:
        updates["authorization_status"] = authorization_status
    if sponsorship_policy is not None:
        updates["sponsorship_policy"] = sponsorship_policy
    if clearance_status is not None:
        updates["clearance_status"] = clearance_status
    if updates:
        typed = typed.model_copy(update=updates)
    return typed.model_dump(mode="json"), typed


def _apply_preferences(profile: Profile, preferences: dict | None) -> Profile:
    """The profile the SEARCH sees: extraction fields overridden by the human's choice.

    The stored profile stays untouched — it is what the extractor measured and
    what evaluation grades. Only the search runs on the chosen locations.
    """
    if not preferences:
        return profile
    locations = list(preferences.get("locations") or [])
    if preferences.get("country_scope") == "us" or any(_is_us_scope(loc) for loc in locations):
        locations = [US_SCOPE_CHOICE]
    return profile.model_copy(update={"locations": locations, "remote_ok": bool(preferences.get("remote"))})


def _preference_selection(profile: Profile, preferences: dict | None) -> tuple[list[str], list[str]]:
    """(choices, selected) for the chooser — stored preferences win, else the extraction's suggestion."""
    if preferences:
        locations = [loc for loc in (preferences.get("locations") or []) if loc]
        remote = bool(preferences.get("remote")) or "remote" in candidate_store.preference_model(preferences).accepted_work_modes
        if preferences.get("country_scope") == "us" or any(_is_us_scope(loc) for loc in locations):
            locations = [US_SCOPE_CHOICE]
    else:
        # New candidates start with the product policy: US-wide relocation and
        # all work modes accepted. The user can narrow this before searching.
        locations, remote = [US_SCOPE_CHOICE], True
    choices = [*dict.fromkeys([*profile.locations, *locations, US_SCOPE_CHOICE, REMOTE_CHOICE])]
    return choices, [*locations, *([REMOTE_CHOICE] if remote else [])]


def on_prefs_change(selection: list[str], profile: Profile | None, cv_text: str, thread_id: str) -> None:
    """Persist the chooser's state and keep the voice bridge on the same page."""
    if profile is None:
        return
    stored = candidate_store.load_candidate()
    links = stored.cv_links if stored else []
    current = stored.candidate_preferences if stored else CandidatePreferences()
    location_prefs = _selection_to_prefs(selection)
    # Location changes must not silently reset the separately confirmed
    # employment, timing, role-family, authorization, or clearance policy.
    modes = [mode for mode in current.accepted_work_modes if mode != "remote"]
    if location_prefs.get("remote"):
        modes.insert(0, "remote")
    current = current.model_copy(
        update={
            "locations": location_prefs["locations"],
            "country_scope": location_prefs.get("country_scope", "selected"),
            "accepted_work_modes": list(dict.fromkeys(modes)),
        }
    )
    prefs = current.model_dump(mode="json")
    voice_bridge.get_bridge().record_profile(candidate_store.effective_profile(profile, prefs), cv_text, thread_id, links)
    candidate_store.save_candidate(profile, cv_text, prefs, links)


def on_add_location(new_location: str, choices: list[str], selection: list[str], profile, cv_text, thread_id):
    """Add a typed location to the chooser, ticked, and persist.

    Outputs: (loc_choices_state, loc_group, loc_new_textbox).
    """
    location = (new_location or "").strip()
    if not location or location in choices:
        return choices, gr.update(), ""
    choices = [*choices[:-1], location, REMOTE_CHOICE] if choices else [location, REMOTE_CHOICE]
    selection = [*selection, location]
    on_prefs_change(selection, profile, cv_text, thread_id)  # programmatic updates don't fire .change
    return choices, gr.update(choices=choices, value=selection), ""


def on_links_change(value, profile: Profile | None, cv_text: str, thread_id: str):
    """Persist human-edited labels while keeping URLs and page metadata typed."""
    if profile is None:
        return gr.update()
    try:
        links = [CVLink.model_validate(item) for item in (value or []) if isinstance(item, dict)]
    except Exception as exc:  # noqa: BLE001 - surface malformed editor data without changing state
        gr.Warning(f"Could not save link edits: {exc}")
        return gr.update()
    stored = candidate_store.load_candidate()
    preferences = stored.preferences if stored else None
    effective = candidate_store.effective_profile(profile, preferences)
    voice_bridge.get_bridge().record_profile(effective, cv_text, thread_id, links)
    candidate_store.save_candidate(profile, cv_text, preferences, links)
    return [link.model_dump() for link in links]


def _on_load(thread_id: str):
    """Register the wizard thread with the voice bridge and restore a saved candidate.

    With a stored candidate the app opens on step 2 — profile shown, chooser
    restored, Find jobs ready, Jobvis pre-seeded — and no LLM call happens
    (the extraction was paid for when the CV was first uploaded). Jobs are NOT
    restored: results go stale, so every session fetches fresh.
    Outputs also restore the typed target-search controls so a restart cannot
    silently replace a candidate's confirmed policy with the defaults.
    """
    voice_bridge.get_bridge().register_thread(thread_id)
    stored = candidate_store.load_candidate()
    if stored is None:
        return (gr.update(),) * 17
    choices, selected = _preference_selection(stored.profile, stored.preferences)
    effective = candidate_store.effective_profile(stored.profile, stored.preferences)
    voice_bridge.get_bridge().record_profile(effective, stored.cv_text, thread_id, stored.cv_links)
    policy = stored.candidate_preferences
    note = (
        '<p class="js-muted" style="text-align:center;font-size:0.8rem;margin-top:10px">'
        "Restored from your last session — “Start over” forgets it.</p>"
    )
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        _profile_html(stored.profile, stored.cv_links) + note,
        stored.cv_text,
        stored.profile,
        choices,
        gr.update(choices=choices, value=selected),
        gr.update(value=[link.model_dump() for link in stored.cv_links]),
        gr.update(value=policy.model_dump(mode="json")),
        gr.update(value=policy.employment_types),
        gr.update(value=str(policy.target_start_min or "")),
        gr.update(value=str(policy.target_start_max or "")),
        gr.update(value=policy.accepted_work_modes),
        gr.update(value=policy.primary_role_families),
        gr.update(value=policy.authorization_status),
        gr.update(value=policy.sponsorship_policy),
        gr.update(value=policy.clearance_status),
    )


def on_upload(file_path: str | None, thread_id: str):
    """Step 1 → 2: read the resume, extract the profile, and reveal the profile page.

    The chooser is seeded with the extraction's locations/remote as a ticked
    SUGGESTION — the human confirms or edits before any search runs.
    Outputs: (page_start, page_profile, profile_html, cv_text_state,
    start_status, profile_state, loc_choices_state, loc_group, links_editor).
    """
    stay = gr.update(visible=True), gr.update(visible=False)
    go = gr.update(visible=False), gr.update(visible=True)
    no = gr.update()

    if not file_path:
        yield (*stay, gr.update(), "", _status("Please drop a PDF resume.", error=True), None, no, no, no)
        return
    try:
        from job_scout.tools.cv_reader import extract_cv_document

        cv_text, cv_links = extract_cv_document(file_path)
    except CVReadError as exc:
        yield (*stay, gr.update(), "", _status(f"Could not read that PDF: {exc}", error=True), None, no, no, no)
        return

    yield (*go, _loading_html("Reading your resume…"), cv_text, "", gr.update(), no, no, no)
    try:
        profile = extract_profile(cv_text, thread_id=thread_id, tags=["phase-2", "ui", "extract"])
    except Exception as exc:  # noqa: BLE001 - show a friendly error and return to start
        msg = f"Couldn't read a profile: {exc}"
        if "api_key" in str(exc).lower() or "credentials" in str(exc).lower():
            msg += " Add an LLM key to .env (OPENAI_API_KEY, or a free SCOUT_MODEL via groq:/ollama:)."
        yield (*stay, gr.update(), "", _status(msg, error=True), None, no, no, no)
        return
    choices, selected = _preference_selection(profile, None)
    initial_preferences = CandidatePreferences().model_copy(update={"locations": _selection_to_prefs(selected)["locations"]})
    voice_bridge.get_bridge().record_profile(profile, cv_text, thread_id, cv_links, initial_preferences)
    candidate_store.save_candidate(profile, cv_text, initial_preferences.model_dump(mode="json"), cv_links)
    _voice_context(f"Screen event: the user just uploaded a CV; a profile was extracted for {profile.name or 'the candidate'}.")
    yield (
        *go,
        _profile_html(profile, cv_links),
        cv_text,
        "",
        profile,
        choices,
        gr.update(choices=choices, value=selected),
        gr.update(value=[link.model_dump() for link in cv_links]),
    )


def on_find(
    cv_text: str,
    profile: Profile | None,
    thread_id: str,
    loc_selection: list[str],
    preference_value: dict | None = None,
    employment_types: list[str] | None = None,
    target_start_min: object = None,
    target_start_max: object = None,
    accepted_work_modes: list[str] | None = None,
    primary_role_families: list[str] | None = None,
    authorization_status: str | None = None,
    sponsorship_policy: str | None = None,
    clearance_status: str | None = None,
):
    """Step 2 → 3: run the job-finding graph for the chosen locations and stream results.

    The search runs on the EFFECTIVE profile — extraction overridden by the
    chooser's ticked locations/remote — so the model builds its query around
    where the human actually wants to work, not where it guessed.
    Outputs: (page_profile, page_results, results_html, footer_html, job_select).
    """
    go = gr.update(visible=False), gr.update(visible=True)
    if profile is None:
        yield (gr.update(visible=True), gr.update(visible=False), gr.update(), "", gr.update())
        return

    prefs, typed_preferences = _preferences_from_ui(
        loc_selection or [],
        preference_value,
        employment_types,
        target_start_min,
        target_start_max,
        accepted_work_modes,
        primary_role_families,
        authorization_status,
        sponsorship_policy,
        clearance_status,
    )
    effective = candidate_store.effective_profile(profile, prefs)
    stored = candidate_store.load_candidate()
    cv_links = stored.cv_links if stored else []
    voice_bridge.get_bridge().record_profile(effective, cv_text, thread_id, cv_links, typed_preferences)
    candidate_store.save_candidate(profile, cv_text, prefs, cv_links)

    yield (*go, _loading_html("Searching for jobs…"), "", gr.update())
    result = RunResult()
    for kind, payload in stream_search(
        effective,
        cv_text=cv_text,
        cv_links=cv_links,
        thread_id=thread_id,
        tags=["phase-2", "ui"],
        preferences=typed_preferences,
    ):
        if kind == "status":
            yield (*go, _loading_html(str(payload)), "", gr.update())
        elif kind == "result":
            result = payload  # type: ignore[assignment]

    choices = [
        (f"{r.job.title} — {r.job.company} (fit {r.fit_score})", r.job.job_id)
        for r in result.ranked_jobs
        if r.eligibility_status != "blocked"
    ]
    select = gr.update(choices=choices, value=None, visible=bool(choices))
    voice_bridge.get_bridge().record_step("results")
    _voice_context(f"Screen event: an on-screen job search just finished — {len(result.ranked_jobs)} ranked jobs are visible.")
    yield (*go, _results_html(result), _footer_html(result), select)


def on_zip(zip_path: str | None) -> str | None:
    """Store the optional LinkedIn export path in session state (path only, never traced)."""
    voice_bridge.get_bridge().record_linkedin_zip(zip_path)
    return zip_path


def on_tailor(selected_job_id: str | None, thread_id: str, linkedin_zip: str | None, profile: Profile | None):
    """Step 3 → 4: tailor an application for the selected job on the SAME thread.

    Only ``selected_job_id`` (+ optional LinkedIn export path) is passed to the
    graph — profile, ranked jobs and CV text come from the thread's checkpoint.
    Outputs: (page_results, page_tailor, tailor_html, tailor_footer, cv_pdf_btn,
    cv_tex_btn, cover_letter_pdf_btn, cover_letter_tex_btn).
    """
    stay = gr.update(visible=True), gr.update(visible=False)
    go = gr.update(visible=False), gr.update(visible=True)
    hidden = gr.update(visible=False)
    if not selected_job_id:
        gr.Warning("Pick a job from the dropdown first.")
        yield (*stay, gr.update(), gr.update(), hidden, hidden, hidden, hidden)
        return

    yield (*go, _loading_html("Drafting the application…"), "", hidden, hidden, hidden, hidden)
    result = TailorResult()
    stream = stream_tailor(
        thread_id=thread_id,
        selected_job_id=selected_job_id,
        tags=["phase-2", "ui", "tailor"],
        linkedin_zip_path=linkedin_zip,
    )
    for kind, payload in stream:
        if kind == "status":
            yield (*go, _loading_html(str(payload)), "", hidden, hidden, hidden, hidden)
        elif kind == "result":
            result = payload  # type: ignore[assignment]

    pdf_btn, tex_btn, letter_pdf_btn, letter_tex_btn, footer = _pack_downloads(result, profile)
    voice_bridge.get_bridge().record_step("tailor")
    _voice_context("Screen event: an application pack was just tailored via the on-screen button and is now visible.")
    yield (*go, _pack_html(result), footer, pdf_btn, tex_btn, letter_pdf_btn, letter_tex_btn)


def reset():
    """Return to step 1, clear the wizard, start a FRESH thread, forget the candidate.

    A new thread_id matters now that the checkpointer is shared: reusing the
    old thread for a different resume would mix two candidates' state. The
    persisted candidate is cleared too — "start over" is the explicit way to
    switch resumes, whereas an app restart keeps you.

    Outputs: (page_start, page_profile, page_results, page_tailor, cv_file,
    cv_text_state, start_status, results_html, footer_html, thread_id,
    linkedin_zip_state, job_select, tailor_out, tailor_footer, pdf_btn, tex_btn,
    cover_letter_pdf_btn, cover_letter_tex_btn,
    loc_choices_state, loc_group).
    """
    new_thread_id = str(uuid4())
    voice_bridge.get_bridge().register_thread(new_thread_id)
    candidate_store.clear_candidate()
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        None,
        "",
        "",
        "",
        "",
        new_thread_id,
        None,
        gr.update(choices=[], value=None),
        "",
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        [],
        gr.update(choices=[], value=[]),
        gr.update(value=[]),
        gr.update(value=CandidatePreferences().model_dump(mode="json")),
        gr.update(value=CandidatePreferences().employment_types),
        gr.update(value=str(CandidatePreferences().target_start_min)),
        gr.update(value=str(CandidatePreferences().target_start_max)),
        gr.update(value=CandidatePreferences().accepted_work_modes),
        gr.update(value=CandidatePreferences().primary_role_families),
        gr.update(value=CandidatePreferences().authorization_status),
        gr.update(value=CandidatePreferences().sponsorship_policy),
        gr.update(value=CandidatePreferences().clearance_status),
    )


def build_app() -> gr.Blocks:
    """Build the four-step wizard app."""
    register_prompts()
    from job_scout.api import CONSOLE_PORT

    with gr.Blocks(title="Job Scout") as demo:
        thread_id = gr.State(lambda: str(uuid4()))
        cv_text_state = gr.State("")
        profile_state = gr.State(None)
        linkedin_zip_state = gr.State(None)

        gr.HTML(
            f'<div id="js-header"><div class="js-mark">{_MARK}<h1>Job Scout</h1></div>'
            f'<div><span class="js-tag">{CAPTION}</span></div></div>'
        )

        voice_ok, voice_hint = is_voice_available()
        if voice_ok:
            gr.HTML(
                '<div id="jv-strip"><span class="jv-hint">Jobvis, the voice concierge, runs in its own console.</span>'
                f'<a class="jv-link" href="http://localhost:{CONSOLE_PORT}" '
                'target="_blank" rel="noopener">Talk to Jobvis ↗</a></div>'
            )
        else:
            gr.HTML(f'<p class="jv-hint">{escape(voice_hint)}</p>')

        with gr.Group(visible=True, elem_classes=["js-page"]) as page_start:
            gr.HTML(_stepper(1))
            gr.HTML(INTRO_HTML)
            cv_file = gr.File(label="", file_types=[".pdf"], type="filepath", height=150, elem_classes=["js-drop"])
            start_status = gr.HTML('<div class="js-status"></div>')

        with gr.Group(visible=False, elem_classes=["js-page"]) as page_profile:
            gr.HTML(_stepper(2))
            gr.HTML('<p class="js-section-label">Your profile</p>')
            profile_out = gr.HTML()
            links_editor = gr.JSON(
                label="Detected resume links — edit labels if needed",
                value=[],
                visible=True,
                every=None,
            )
            gr.HTML('<p class="js-section-label" style="margin-top:14px">Where should we search?</p>')
            gr.HTML(
                '<p class="js-muted" style="font-size:0.86rem">The boxes come from your resume — a suggestion, '
                "not a decision. Tick cities, or choose <b>Anywhere in the United States</b> if you are open to "
                "relocating to any US city; add anywhere we missed.</p>"
            )
            default_policy = CandidatePreferences()
            with gr.Accordion("Target search policy", open=True):
                gr.HTML(
                    '<p class="js-muted" style="font-size:0.86rem">These constraints are authoritative. '
                    "Jobvis will not infer work authorization, sponsorship, or clearance from your resume.</p>"
                )
                with gr.Row():
                    employment_types = gr.CheckboxGroup(
                        label="Employment type",
                        choices=[
                            ("Full-time", "full_time"),
                            ("Contract", "contract"),
                            ("Part-time", "part_time"),
                        ],
                        value=default_policy.employment_types,
                    )
                    accepted_work_modes = gr.CheckboxGroup(
                        label="Accepted work modes",
                        choices=[("Remote", "remote"), ("Hybrid", "hybrid"), ("Onsite", "onsite")],
                        value=default_policy.accepted_work_modes,
                    )
                with gr.Row():
                    target_start_min = gr.Textbox(
                        label="Earliest start (YYYY-MM-DD)",
                        value=str(default_policy.target_start_min),
                    )
                    target_start_max = gr.Textbox(
                        label="Latest start (YYYY-MM-DD)",
                        value=str(default_policy.target_start_max),
                    )
                primary_role_families = gr.CheckboxGroup(
                    label="Primary role families",
                    choices=[
                        ("AI / ML", "ai_ml"),
                        ("Data Science", "data_science"),
                        ("GenAI / LLM / RAG", "genai"),
                        ("Forward-deployed / Solutions", "forward_deployed"),
                        ("Computer Vision", "computer_vision"),
                        ("MLOps / ML Platform", "mlops"),
                    ],
                    value=default_policy.primary_role_families,
                )
                with gr.Row():
                    authorization_status = gr.Dropdown(
                        label="Work authorization",
                        choices=[("Unknown — ask me", "unknown"), ("Confirmed", "confirmed"), ("Not eligible", "ineligible")],
                        value=default_policy.authorization_status,
                    )
                    sponsorship_policy = gr.Dropdown(
                        label="Sponsorship",
                        choices=[
                            ("Unknown — review each job", "unknown"),
                            ("Required", "required"),
                            ("Not required", "not_required"),
                        ],
                        value=default_policy.sponsorship_policy,
                    )
                    clearance_status = gr.Dropdown(
                        label="Security clearance",
                        choices=[("Unknown / none confirmed", "unknown"), ("Confirmed", "confirmed")],
                        value=default_policy.clearance_status,
                    )
            with gr.Accordion("Advanced policy JSON", open=False):
                target_preferences = gr.JSON(
                    label="Target search policy (advanced)",
                    value=default_policy.model_dump(mode="json"),
                    visible=True,
                )
            # Keep the two global choices present even before profile extraction
            # finishes. This prevents Gradio from rejecting a fast click on
            # "Anywhere in the United States" while the dynamic city choices
            # are still being sent to the browser.
            loc_group = gr.CheckboxGroup(
                label="",
                show_label=False,
                choices=[US_SCOPE_CHOICE, REMOTE_CHOICE],
                value=[],
                interactive=True,
            )
            with gr.Row():
                loc_new = gr.Textbox(label="", show_label=False, placeholder="Add another location…", scale=4)
                loc_add = gr.Button("Add", size="sm", scale=0)
            find_btn = gr.Button("Find jobs", variant="primary", size="lg")
            with gr.Accordion("Add your LinkedIn export (optional)", open=False):
                gr.HTML(
                    '<p class="js-muted" style="font-size:0.86rem">Used only to enrich the tailoring step. '
                    "LinkedIn → Settings → “Get a copy of your data”. The ZIP stays on your machine and is "
                    "never attached to traces.</p>"
                )
                linkedin_file = gr.File(label="", file_types=[".zip"], type="filepath", height=90)
            change_btn = gr.Button("Upload a different resume", variant="secondary")

        with gr.Group(visible=False, elem_classes=["js-page"]) as page_results:
            gr.HTML(_stepper(3))
            gr.HTML('<p class="js-section-label">Ranked jobs</p>')
            results_out = gr.HTML()
            footer_out = gr.HTML()
            gr.HTML('<p class="js-section-label" style="margin-top:14px">Prepare an application</p>')
            job_select = gr.Dropdown(label="", choices=[], value=None, visible=False, interactive=True)
            tailor_btn = gr.Button("Tailor application", variant="primary")
            restart_btn = gr.Button("Start over", variant="secondary")

        with gr.Group(visible=False, elem_classes=["js-page"]) as page_tailor:
            gr.HTML(_stepper(4))
            gr.HTML('<p class="js-section-label">Application pack</p>')
            gr.HTML(
                '<div class="js-honesty"><b>Review before sending</b> — this is a draft, not an application submission. '
                "Read the warning list, cover letter, and honesty note; correct anything you cannot support, then "
                "submit it yourself on the employer's site.</div>"
            )
            tailor_out = gr.HTML()
            with gr.Row():
                pdf_btn = gr.DownloadButton("Download tailored CV (PDF)", visible=False, variant="primary")
                tex_btn = gr.DownloadButton("Download tailored CV (.tex)", visible=False, variant="secondary")
            with gr.Row():
                cover_letter_pdf_btn = gr.DownloadButton("Download cover letter (PDF)", visible=False, variant="primary")
                cover_letter_tex_btn = gr.DownloadButton("Download cover letter (.tex)", visible=False, variant="secondary")
            tailor_footer = gr.HTML()
            back_btn = gr.Button("Back to jobs", variant="secondary")
            restart_btn2 = gr.Button("Start over", variant="secondary")

        loc_choices_state = gr.State([])
        cv_file.upload(
            on_upload,
            inputs=[cv_file, thread_id],
            outputs=[
                page_start,
                page_profile,
                profile_out,
                cv_text_state,
                start_status,
                profile_state,
                loc_choices_state,
                loc_group,
                links_editor,
            ],
        )
        links_editor.change(
            on_links_change, inputs=[links_editor, profile_state, cv_text_state, thread_id], outputs=[links_editor]
        )
        linkedin_file.change(on_zip, inputs=[linkedin_file], outputs=[linkedin_zip_state])
        loc_group.input(on_prefs_change, inputs=[loc_group, profile_state, cv_text_state, thread_id])
        loc_add.click(
            on_add_location,
            inputs=[loc_new, loc_choices_state, loc_group, profile_state, cv_text_state, thread_id],
            outputs=[loc_choices_state, loc_group, loc_new],
        )
        loc_new.submit(
            on_add_location,
            inputs=[loc_new, loc_choices_state, loc_group, profile_state, cv_text_state, thread_id],
            outputs=[loc_choices_state, loc_group, loc_new],
        )
        find_btn.click(
            on_find,
            inputs=[
                cv_text_state,
                profile_state,
                thread_id,
                loc_group,
                target_preferences,
                employment_types,
                target_start_min,
                target_start_max,
                accepted_work_modes,
                primary_role_families,
                authorization_status,
                sponsorship_policy,
                clearance_status,
            ],
            outputs=[page_profile, page_results, results_out, footer_out, job_select],
        )
        tailor_btn.click(
            on_tailor,
            inputs=[job_select, thread_id, linkedin_zip_state, profile_state],
            outputs=[
                page_results,
                page_tailor,
                tailor_out,
                tailor_footer,
                pdf_btn,
                tex_btn,
                cover_letter_pdf_btn,
                cover_letter_tex_btn,
            ],
        )
        back_btn.click(
            lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[page_results, page_tailor],
        )
        reset_outputs = [
            page_start,
            page_profile,
            page_results,
            page_tailor,
            cv_file,
            cv_text_state,
            start_status,
            results_out,
            footer_out,
            thread_id,
            linkedin_zip_state,
            job_select,
            tailor_out,
            tailor_footer,
            pdf_btn,
            tex_btn,
            cover_letter_pdf_btn,
            cover_letter_tex_btn,
            loc_choices_state,
            loc_group,
            links_editor,
            target_preferences,
            employment_types,
            target_start_min,
            target_start_max,
            accepted_work_modes,
            primary_role_families,
            authorization_status,
            sponsorship_policy,
            clearance_status,
        ]
        change_btn.click(reset, outputs=reset_outputs)
        restart_btn.click(reset, outputs=reset_outputs)
        restart_btn2.click(reset, outputs=reset_outputs)

        if voice_ok:
            # A run started by voice in the console still has to land on these
            # pages. The Timer must sit at Blocks ROOT level: one created inside
            # a layout container can miss the client render tree, and its null
            # instance kills the frontend at dispatch_load_events (Gradio 6.20).
            gr.Timer(1.0).tick(
                on_run_tick,
                outputs=[
                    page_profile,
                    page_results,
                    page_tailor,
                    results_out,
                    footer_out,
                    job_select,
                    tailor_out,
                    tailor_footer,
                    pdf_btn,
                    tex_btn,
                    cover_letter_pdf_btn,
                    cover_letter_tex_btn,
                ],
            )
        # Restore on page load. A load event, not a Timer: the client only opens
        # its event connection on the first event, so an unprimed timer never
        # fires.
        demo.load(
            _on_load,
            inputs=[thread_id],
            outputs=[
                page_start,
                page_profile,
                profile_out,
                cv_text_state,
                profile_state,
                loc_choices_state,
                loc_group,
                links_editor,
                target_preferences,
                employment_types,
                target_start_min,
                target_start_max,
                accepted_work_modes,
                primary_role_families,
                authorization_status,
                sponsorship_policy,
                clearance_status,
            ],
        )

    return demo


def main() -> None:
    """Serve both surfaces from ONE process on their configured local ports.

    One process is not a convenience here, it is a correctness requirement. The
    voice bridge and the LangGraph ``MemorySaver`` are both process-wide, so
    running the two servers separately gives you two sessions wearing the same
    name: the wizard finds jobs the console cannot see, and Jobvis truthfully
    reports an empty checkpoint. The API goes on a daemon thread first; Gradio
    keeps the main thread, as it prefers to.
    """
    from job_scout.api import CONSOLE_PORT, WIZARD_PORT, serve_in_thread

    if WIZARD_PORT == CONSOLE_PORT:
        raise RuntimeError(f"Jobvis wizard and console must use different ports, got :{WIZARD_PORT} for both")
    _assert_port_available(WIZARD_PORT, "wizard")
    _assert_port_available(CONSOLE_PORT, "console")
    serve_in_thread()
    print(f"Jobvis console: http://localhost:{CONSOLE_PORT}")
    build_app().launch(server_name="127.0.0.1", server_port=WIZARD_PORT, theme=THEME, css=CSS)


def _assert_port_available(port: int, surface: str) -> None:
    """Fail before starting either server when a local port is occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # A recently stopped local server can leave the port in TIME_WAIT.
        # Match the reuse behavior of the servers we are about to start so the
        # preflight does not reject an otherwise free Jobvis port.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(
                f"Jobvis {surface} cannot start: port :{port} is already in use. "
                "Run `make stop` for Jobvis listeners or choose another configured port."
            ) from exc


if __name__ == "__main__":
    main()
