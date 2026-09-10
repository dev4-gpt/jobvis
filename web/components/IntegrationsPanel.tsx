"use client";

import { useCallback, useEffect, useState } from "react";
import { integrationRequest as request, exportQueueUrl, type Expansion, type Integrations, type MemoryEntry,
  type PackAudit, type ReviewedSnapshot, type State, type TrackedApplication } from "@/lib/api";

export default function IntegrationsPanel({ state }: { state: State }) {
  const [config, setConfig] = useState<Integrations | null>(null);
  const [expansion, setExpansion] = useState<Expansion>({ status: "idle", run_id: null });
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
  const [memoryAvailable, setMemoryAvailable] = useState(false);
  const [memoryError, setMemoryError] = useState("");
  const [snapshots, setSnapshots] = useState<ReviewedSnapshot[]>([]);
  const [applications, setApplications] = useState<TrackedApplication[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [preview, setPreview] = useState<{ pack_key: string; audit: PackAudit } | null>(null);
  const [acknowledge, setAcknowledge] = useState(false);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [content, setContent] = useState("");
  const [question, setQuestion] = useState("");
  const [sensitive, setSensitive] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<MemoryEntry[]>([]);
  const [lastExport, setLastExport] = useState<string | null>(null);

  const reload = useCallback(async () => {
    const [configuration, packs, tracker] = await Promise.all([
      request<Integrations>("/integrations"),
      request<{ snapshots: ReviewedSnapshot[] }>("/packs/reviewed"),
      request<{ applications: TrackedApplication[] }>("/applications"),
    ]);
    setConfig(configuration); setSnapshots(packs.snapshots); setApplications(tracker.applications);
  }, []);

  async function loadMemory() {
    const result = await request<{ available: boolean; entries: MemoryEntry[]; error?: string }>("/memory");
    setMemoryAvailable(result.available); setMemories(result.entries); setMemoryError(result.error || "");
  }

  useEffect(() => {
    if (state.candidate?.name) void Promise.resolve().then(reload).catch((error) => setNotice(String(error)));
  }, [reload, state.thread_id, state.candidate?.name]);

  useEffect(() => {
    const timer = setInterval(() => {
      void request<Expansion>("/apify/expansion").then(setExpansion).catch(() => undefined);
    }, 2000);
    return () => clearInterval(timer);
  }, []);

  async function action(work: () => Promise<void>) {
    setBusy(true); setNotice("");
    try { await work(); await reload(); }
    catch (error) { setNotice(String(error)); }
    finally { setBusy(false); }
  }

  if (!state.candidate) return null;
  const running = busy || state.run.running || ["starting", "running", "ranking"].includes(expansion.status);
  return <section className="block integrations-panel no-drag">
    <h2>Search and application tools</h2>
    <p>Curated boards: {config?.direct_sources_enabled ? "enabled" : "disabled"} · Listing checks: {config?.liveness_enabled ? "enabled" : "disabled"}</p>
    {config && <p>{Object.entries(config.boards).map(([source, boards]) => `${source}: ${boards.length}`).join(" · ")}</p>}
    <button disabled={running || !config?.apify_ready} onClick={() => void action(async () => {
      await request("/apify/expansion", "POST"); setExpansion({ status: "starting", run_id: null });
    })}>Expand search with Apify</button>
    <p>{config?.apify_hint || "Runs one configured task, up to 25 results. Provider charges may apply."}</p>
    <p aria-live="polite">Expansion: {expansion.status}{expansion.run_id ? ` · ${expansion.run_id}` : ""} {expansion.error}</p>

    <details onToggle={(event) => { if (event.currentTarget.open) void loadMemory().catch((error) => setMemoryError(String(error))); }}>
      <summary>Encrypted local memory</summary>
      <p>Save and confirm reusable answers here. Sensitive answers need review each time. Resume claims still come from your uploaded resume.</p>
      {memoryError && <p role="alert">{memoryError}</p>}
      <label>Question or label<input value={question} onChange={(event) => setQuestion(event.target.value)} /></label>
      <label>Answer or fact<textarea value={content} onChange={(event) => setContent(event.target.value)} /></label>
      <label><input type="checkbox" checked={sensitive} onChange={(event) => setSensitive(event.target.checked)} /> Sensitive answer</label>
      <button disabled={busy || !memoryAvailable || !content.trim()} onClick={() => void action(async () => {
        await request(editing ? `/memory/${editing}` : "/memory", editing ? "PUT" : "POST",
          { content, question_key: question, sensitive, category: "qa_answer" });
        setEditing(null); setContent(""); await loadMemory();
      })}>{editing ? "Save edit for review" : "Save for review"}</button>
      <button disabled={busy || !memoryAvailable} onClick={() => void action(async () => {
        const result = await request<{ note: string }>("/memory/import-legacy", "POST");
        setNotice(result.note); await loadMemory();
      })}>Import legacy memory for review</button>
      <button disabled={busy || !memoryAvailable} onClick={() => void action(async () => {
        const result = await request<{ note: string }>("/memory/import-resume", "POST");
        setNotice(result.note); await loadMemory();
      })}>Import current resume facts for review</button>
      <button disabled={busy || !memoryAvailable || !question.trim()} onClick={() => void action(async () => {
        const result = await request<{ suggestions: MemoryEntry[] }>(`/memory/suggestions?question=${encodeURIComponent(question)}`);
        setSuggestions(result.suggestions);
      })}>Find confirmed answer suggestions</button>
      {suggestions.map((entry) => <p key={entry.id}>{entry.content} {entry.sensitive && "— sensitive: review before using"}</p>)}
      {memories.map((entry) => <article key={entry.id}>
        <p>{entry.question_key}: {entry.content}</p>
        <p>{entry.confirmed ? "Confirmed" : "Needs confirmation"} · {entry.provenance}</p>
        <button disabled={busy} onClick={() => { setEditing(entry.id); setContent(entry.content); setQuestion(entry.question_key); setSensitive(entry.sensitive); }}>Edit</button>
        {!entry.confirmed && <button disabled={busy} onClick={() => void action(async () => {
          await request(`/memory/${entry.id}/confirm`, "POST"); await loadMemory();
        })}>Confirm this memory</button>}
        <button disabled={busy} onClick={() => void action(async () => {
          await request(`/memory/${entry.id}`, "DELETE"); setSuggestions([]); await loadMemory();
        })}>Delete</button>
      </article>)}
    </details>

    <details>
      <summary>Review and export application packs</summary>
      <button disabled={running || !state.pack} onClick={() => void action(async () => {
        setPreview(await request("/pack/review"));
      })}>Check current pack for review</button>
      {preview && <div>
        <p>{preview.audit.passed ? "Artifact checks passed. Inspect the resume and letter before approving." : "Artifact checks failed."}</p>
        {preview.audit.issues.map((issue, index) => <p key={index}>{issue.message}</p>)}
        <label><input type="checkbox" checked={acknowledge} onChange={(event) => setAcknowledge(event.target.checked)} />
          I acknowledge any uncertainty in whether these selected postings are still open.</label>
        <button disabled={running || !preview.audit.passed} onClick={() => void action(async () => {
          await request("/pack/review", "POST", { pack_key: preview.pack_key, approved: true, acknowledge_uncertain: acknowledge });
          setNotice("Reviewed snapshot saved."); setPreview(null);
        })}>I reviewed this pack — approve snapshot</button>
      </div>}
      {snapshots.map((snapshot) => <label key={snapshot.id} style={{ display: "block" }}>
        <input type="checkbox" disabled={!snapshot.approved_at} checked={selected.includes(snapshot.id)}
          onChange={(event) => setSelected((ids) => event.target.checked ? [...ids, snapshot.id] : ids.filter((id) => id !== snapshot.id))} />
        {snapshot.company} · {snapshot.role} · {snapshot.score}/100 · {snapshot.approved_at ? "reviewed" : "needs new review"}
      </label>)}
      <label><input type="checkbox" checked={acknowledge} onChange={(event) => setAcknowledge(event.target.checked)} /> Acknowledge uncertain listing checks for export</label>
      <button disabled={running || !selected.length} onClick={() => void action(async () => {
        const result = await request<{ id: string; queue_path: string }>("/packs/export", "POST", { snapshot_ids: selected, acknowledge_uncertain: acknowledge });
        setLastExport(result.id);
        setNotice(`Export saved: ${result.queue_path}. Launch the external worker separately.`);
      })}>Export selected reviewed packs</button>
      {lastExport && <p><a href={exportQueueUrl(lastExport)}>Download AIHawk queue</a></p>}
    </details>

    <details><summary>Application tracker and follow-ups</summary>
      {applications.map((record) => <article key={record.application_id}>
        <p>{record.company} · {record.title} · {record.status}</p>
        {record.followup_due?.map((date) => <p key={date}>Follow up: {new Date(date).toLocaleString()}</p>)}
        {record.thank_you_due && <p>Thank you due: {new Date(record.thank_you_due).toLocaleString()}</p>}
        <button disabled={busy} onClick={() => void action(async () => { await request(`/applications/${record.application_id}/status`, "POST", { status: "submitted_by_user" }); })}>Record submission now</button>
        <button disabled={busy} onClick={() => void action(async () => { await request(`/applications/${record.application_id}/status`, "POST", { status: "interview" }); })}>Record interview now</button>
      </article>)}
    </details>
    {notice && <p role="status">{notice}</p>}
  </section>;
}
