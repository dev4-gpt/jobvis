"use client";

/**
 * The contents of the glass slab. Every value here came from the LangGraph
 * checkpoint via /api/state — the console renders facts, it never derives them.
 *
 * The exception, and the point of the panel: `nextStep` derives nothing about
 * the WORLD, only about what the person should do next. That is a UI concern,
 * so it lives in the UI.
 */

import { coverLetterUrl, packUrl, type State } from "@/lib/api";
import type { OrbMode } from "@/lib/orbScene";
import { useState } from "react";

export type Step = { now: string; next: string; cue?: string };

/** What is happening, and what to say next. The slab's headline. */
export function nextStep(state: State, mode: OrbMode): Step {
  if (state.run.running) {
    return {
      now: state.run.latest_status || "Working…",
      next: "This takes about a minute. Jobvis will speak up when it lands.",
    };
  }
  if (!state.candidate) {
    return {
      now: "No candidate yet",
      next: "Drop a CV in the wizard once — Jobvis remembers it from then on.",
    };
  }
  if (mode === "idle") {
    return {
      // No cue here on purpose: `cue` is a phrase to SAY, and there is nothing
      // to say to an agent that is not listening yet.
      now: `Ready for ${state.candidate.name || "you"}`,
      next: "Engage Jobvis, top right, to start talking.",
    };
  }
  if (state.pack) {
    return {
      now: "Application ready",
      next: "Ask for the highlights, or take the downloads below.",
      cue: "give me the highlights",
    };
  }
  if ((state.source_coverage?.ranked ?? state.jobs.length) > 0) {
    return {
      now: `${state.source_coverage?.ranked ?? state.jobs.length} matches ranked`,
      next: "Ask to run through them, or to tailor one.",
      cue: "tailor an application for the first one",
    };
  }
  return { now: "Listening", next: "Ask for jobs whenever you are ready.", cue: "find me jobs" };
}

export function NextPanel({ step }: { step: Step }) {
  return (
    <section className="block now">
      <p className="label">Now</p>
      <p className="now-line">{step.now}</p>
      <p className="next-line">{step.next}</p>
      {step.cue && <p className="cue">&ldquo;{step.cue}&rdquo;</p>}
    </section>
  );
}

export function JobsPanel({ state, onOpenApplication }: { state: State; onOpenApplication: (jobId: string) => void }) {
  const primary = state.primary_jobs ?? state.jobs.filter((job) => job.primary_or_adjacent === "primary");
  const adjacent = state.adjacent_jobs ?? state.jobs.filter((job) => job.primary_or_adjacent === "adjacent");
  const blocked =
    state.blocked_or_review_jobs ??
    state.jobs.filter((job) => job.eligibility_status === "blocked" || job.primary_or_adjacent === "review");
  if (primary.length === 0 && adjacent.length === 0 && blocked.length === 0) return null;

  const renderJob = (job: (typeof state.jobs)[number], index: number, allowOpen: boolean) => (
    <div className="row" key={`${job.job_id}-${index}`} style={{ animationDelay: `${index * 70}ms` }}>
      <span className="rank">{String(job.rank).padStart(2, "0")}</span>
      <span className="job">
        <b>{job.title}</b>
        <span className="meta">
          {job.company} · {job.location}
        </span>
        <span className="meta">
          {job.primary_or_adjacent ?? "review"} · {job.eligibility_status ?? "unknown"}
          {job.work_mode ? ` · ${job.work_mode}` : ""}
          {job.start_timing_fit ? ` · start ${job.start_timing_fit}` : ""}
        </span>
        {(job.hard_blockers?.length || job.eligibility_reasons?.length) ? (
          <span className="meta warning">{(job.hard_blockers ?? job.eligibility_reasons ?? []).slice(0, 2).join(" · ")}</span>
        ) : null}
      </span>
      <span className="score">{job.fit_score}</span>
      {allowOpen && job.url && (
        <button type="button" className="mini-action" onClick={() => onOpenApplication(job.job_id)}>
          Open application
        </button>
      )}
    </div>
  );

  const lane = (label: string, jobs: typeof primary, allowOpen: boolean) =>
    jobs.length > 0 ? (
      <div className="job-lane" key={label}>
        <p className="label">{label} · {jobs.length}</p>
        {jobs.map((job, index) => renderJob(job, index, allowOpen))}
      </div>
    ) : null;

  return (
    <section className="block">
      {lane("Primary matches", primary, true)}
      {lane("Adjacent roles", adjacent, true)}
      {lane("Blocked or review-required", blocked, false)}
      {state.source_coverage?.diagnostics && state.source_coverage.diagnostics.length > 0 && (
        <div className="meta source-diagnostics">
          Sources: {state.source_coverage.diagnostics.map((item) => `${item.source} ${item.returned}${item.error ? " (error)" : ""}`).join(" · ")}
        </div>
      )}
    </section>
  );
}

export function PackPanel({ state }: { state: State }) {
  const pack = state.pack;
  if (!pack) return null;
  return (
    <section className="block">
      <p className="label">Application</p>
      <b className="headline">{pack.headline}</b>
      <p className={pack.flags === 0 ? "verdict clean" : "verdict flagged"}>
        {pack.flags === 0 ? "✓" : "⚠"} {pack.verdict}
      </p>
      <div className="downloads no-drag">
        <a className="pill solid" href={packUrl("pdf")} download>
          Tailored CV · PDF
        </a>
        <a className="pill" href={packUrl("tex")} download>
          .tex
        </a>
        <a className="pill solid" href={coverLetterUrl("pdf")} download>
          Cover letter · PDF
        </a>
        <a className="pill" href={coverLetterUrl("tex")} download>
          Letter .tex
        </a>
      </div>
    </section>
  );
}

export function ApplicationPanel({ state, onFillSafe }: { state: State; onFillSafe: (ids: string[]) => void }) {
  const application = state.application;
  const [selected, setSelected] = useState<string[]>([]);
  if (!application || application.status === "idle") return null;
  const safeFields = application.fields.filter((field) => !field.sensitive && field.has_value);
  return (
    <section className="block no-drag">
      <p className="label">Application review · {application.ats ?? "unknown ATS"}</p>
      <p className="next-line">{application.message}</p>
      {application.fields.map((field) => (
        <label className="field-row" key={field.field_id}>
          <input
            type="checkbox"
            disabled={field.sensitive || !field.has_value}
            checked={selected.includes(field.field_id)}
            onChange={(event) =>
              setSelected((current) =>
                event.target.checked ? [...current, field.field_id] : current.filter((id) => id !== field.field_id),
              )
            }
          />
          <span>
            <b>{field.label}</b> · {field.sensitive ? "sensitive — ask manually" : field.has_value ? `${Math.round(field.confidence * 100)}% confidence` : field.reason}
          </span>
        </label>
      ))}
      {safeFields.length > 0 && (
        <button type="button" className="pill solid" onClick={() => onFillSafe(selected)} disabled={selected.length === 0}>
          Fill approved safe fields
        </button>
      )}
      <p className="meta">Login, MFA, CAPTCHA, unknown questions, and Submit remain manual.</p>
    </section>
  );
}

export function ActivityPanel({ lines }: { lines: { role: string; text: string }[] }) {
  if (lines.length === 0) return null;
  return (
    <section className="block activity no-drag">
      <p className="label">Activity</p>
      <div className="feed">
        {lines.slice(-40).map((line, index) => (
          <div key={index} className={`line ${line.role}`}>
            {line.role === "tool" ? (
              <span className="tool">⚙ {line.text}</span>
            ) : line.role === "system" ? (
              <span className="system">{line.text}</span>
            ) : (
              <>
                <span className="who">{line.role === "you" ? "You" : "Jobvis"}</span>
                {line.text}
              </>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
