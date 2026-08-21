/**
 * Typed client for the Python API. Same-origin in the built console; in dev it
 * points at :8000 via NEXT_PUBLIC_API_BASE.
 */

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export type Job = {
  rank: number;
  job_id: string;
  title: string;
  company: string;
  location: string;
  url: string;
  listing_url?: string;
  application_url?: string;
  fit_score: number;
  why: string;
  final_priority_score?: number;
  role_fit_score?: number;
  evidence_fit_score?: number;
  eligibility_status?: string;
  eligibility_reasons?: string[];
  hard_blockers?: string[];
  primary_or_adjacent?: string;
  start_timing_fit?: string;
  employment_type?: string;
  work_mode?: string;
  experience_level?: string;
  posted_at?: string | null;
  start_date_text?: string;
  clearance_required?: boolean;
  authorization_requirement?: string;
  sponsorship_signal?: string;
  salary_text?: string;
  source_url?: string;
  source?: string;
};

export type Pack = {
  job_id?: string;
  headline: string;
  summary: string;
  cover_letter: string;
  honesty_note: string;
  flags: number;
  verdict: string;
  links?: { label: string; url: string; page: number }[];
  cover_letter_quality?: { passed: boolean; word_count: number; reasons: string[] } | null;
};

export type ApplicationState = {
  application_id?: string;
  job_id: string;
  url: string;
  listing_url?: string;
  application_url?: string;
  status: string;
  message: string;
  ats: string | null;
  fields: { field_id: string; label: string; confidence: number; sensitive: boolean; reason: string; has_value: boolean }[];
};

export type Candidate = {
  name: string;
  role: string;
  seniority: string;
  locations: string[];
  remote_ok: boolean;
  phone?: string | null;
  professional_experience_months?: number | null;
  education_history?: { institution: string; degree: string; field: string; start_date?: string | null; end_date?: string | null; in_progress: boolean }[];
  expected_graduation_date?: string | null;
  current_program?: string | null;
  degree_fields?: string[];
};

export type RunStatus = {
  running: boolean;
  kind?: string;
  latest_status?: string;
  done?: boolean;
  failed?: boolean;
  error?: string;
  note?: string;
  run_id?: string;
  phase?: string;
  cancelled?: boolean;
};

export type State = {
  step: string;
  thread_id: string;
  candidate: Candidate | null;
  candidate_preferences?: Record<string, unknown> | null;
  jobs: Job[];
  primary_jobs?: Job[];
  adjacent_jobs?: Job[];
  blocked_or_review_jobs?: Job[];
  source_coverage?: {
    fetched: number;
    ranked: number;
    primary: number;
    adjacent: number;
    blocked_or_review: number;
    sources: string[];
    diagnostics?: {
      source: string;
      requested: boolean;
      completed: boolean;
      timed_out: boolean;
      latency_ms: number;
      returned: number;
      contributed: boolean;
      error?: string | null;
    }[];
  };
  pack: Pack | null;
  run: RunStatus;
  application: ApplicationState;
};

export type Config = {
  voice_ok: boolean;
  voice_hint: string;
  wizard_url: string;
  has_candidate: boolean;
};

export type ContactCandidate = {
  name: string;
  title: string;
  email: string;
  source_url: string;
  evidence: string;
  confidence: number;
  requires_manual_verification: boolean;
};

export type OutreachDraft = {
  draft_id: string;
  job_id: string;
  company: string;
  role: string;
  contact?: ContactCandidate | null;
  subject: string;
  email_body: string;
  video_script: string;
  why_me: string[];
  requirement_targets: string[];
  evidence_refs: string[];
  review_notes: string[];
  created_at: string;
};

export type PackAudit = {
  passed: boolean;
  original_cv: string;
  tailored_cv: string;
  cover_letter: string;
  cv_pages: number;
  cv_words: number;
  cover_letter_words: number;
  source_links: number;
  generated_links: number;
  missing_links: string[];
  issues: { code: string; severity: string; message: string }[];
  hashes: Record<string, string>;
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) throw new Error(`${path} failed: ${response.status}`);
  return (await response.json()) as T;
}

export const getConfig = () => getJson<Config>("/api/config");
export const getState = () => getJson<State>("/api/state");

export async function openApplication(jobId: string): Promise<ApplicationState> {
  const response = await fetch(`${BASE}/api/application/open`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ job_id: jobId }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail ?? `application open failed: ${response.status}`);
  return body as ApplicationState;
}

export async function fillSafeFields(approvedFieldIds: string[]): Promise<ApplicationState> {
  const response = await fetch(`${BASE}/api/application/fill-safe`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ approved_field_ids: approvedFieldIds }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail ?? `safe fill failed: ${response.status}`);
  return body as ApplicationState;
}

export async function cancelRun(runId?: string): Promise<Record<string, unknown>> {
  const response = await fetch(`${BASE}/api/run/cancel`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(runId ? { run_id: runId } : {}),
  });
  return (await response.json()) as Record<string, unknown>;
}

export type SessionStart = {
  token: string;
  /** Fills persona.FIRST_MESSAGE — unset, the agent greets you with literal braces. */
  dynamicVariables: Record<string, string>;
};

/** Mint a conversation token. The ElevenLabs key never reaches the browser. */
export async function getSessionStart(): Promise<SessionStart> {
  const response = await fetch(`${BASE}/api/voice/token`, { method: "POST" });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail ?? `token request failed: ${response.status}`);
  return { token: body.token as string, dynamicVariables: body.dynamic_variables ?? {} };
}

/**
 * Run one client tool in Python.
 *
 * This is the whole grounding story in one function: the agent asks for a fact,
 * and the answer comes from the LangGraph checkpoint rather than from the
 * model. The browser is a courier, not a source.
 */
export async function callTool(name: string, parameters: Record<string, unknown>): Promise<unknown> {
  const response = await fetch(`${BASE}/api/tools/${name}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(parameters ?? {}),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    return { error: body.detail ?? `tool ${name} failed with ${response.status}` };
  }
  return await response.json();
}

/** Why the last conversation died. The SDK only ever says "Unknown error". */
export async function getLastVoiceError(): Promise<{ reason: string; quota: boolean }> {
  try {
    return await getJson<{ reason: string; quota: boolean }>("/api/voice/last-error");
  } catch {
    return { reason: "", quota: false };
  }
}

export const packUrl = (kind: "pdf" | "tex") => `${BASE}/api/pack/${kind}`;
export const coverLetterUrl = (kind: "pdf" | "tex") => `${BASE}/api/pack/cover-letter/${kind}`;
export const eventsUrl = () => `${BASE}/api/events`;

export async function auditPack(): Promise<PackAudit> {
  const response = await fetch(`${BASE}/api/pack/audit`);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail?.message ?? body.detail ?? `pack audit failed: ${response.status}`);
  return body as PackAudit;
}

export async function generateOutreach(jobId: string, contact?: ContactCandidate): Promise<OutreachDraft> {
  const response = await fetch(`${BASE}/api/outreach/generate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ job_id: jobId, ...(contact ? { contact } : {}) }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail ?? `outreach generation failed: ${response.status}`);
  return body as OutreachDraft;
}
