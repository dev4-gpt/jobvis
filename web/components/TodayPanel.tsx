"use client";

import { useEffect, useState } from "react";
import { integrationRequest as request, type State } from "@/lib/api";

type Today = {
  date: string;
  timezone: string;
  digest: string;
  followups: { application_id: string; title: string; company: string; due: string; kind: string }[];
  run: null | {
    ranked: { job: { job_id: string; title: string; company: string; location: string }; fit_score: number;
      fit_explanation: string; eligibility_status: string }[];
    delivery: { status: string; error?: string };
    warnings: string[];
    diagnostics: { source: string; error?: string; requested: boolean; timed_out: boolean }[];
  };
};
type Application = {
  application_id: string; title: string; company: string; status: string;
  interview_notes: string; interview_story_ids: string[];
};
type Preview = {
  preview_id: string; created: number; matched: number; skipped: number; warnings: string[];
  rows: { company: string; title: string; status: string; matched: boolean }[];
};
const storyFields = ["title", "situation", "task", "action", "result", "evidence"] as const;
type Story = Record<(typeof storyFields)[number], string>;
type Preparation = { stories: Record<string, Story>; supporting_url: string };
type SavedSearch = { watchlist_id: string; query: string; location: string; remote: boolean };
const emptyStory: Story = { title: "", situation: "", task: "", action: "", result: "", evidence: "" };

export default function TodayPanel({ state }: { state: State }) {
  const [today, setToday] = useState<Today | null>(null);
  const [applications, setApplications] = useState<Application[]>([]);
  const [prep, setPrep] = useState<Preparation>({ stories: {}, supporting_url: "" });
  const [source, setSource] = useState("base44");
  const [content, setContent] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [searches, setSearches] = useState<SavedSearch[]>([]);
  const [query, setQuery] = useState("");
  const [location, setLocation] = useState("");
  const [remote, setRemote] = useState(false);
  const [storyId, setStoryId] = useState("");
  const [story, setStory] = useState<Story>(emptyStory);
  const [resource, setResource] = useState<string | null>(null);
  const [selected, setSelected] = useState("");
  const [notes, setNotes] = useState("");
  const [storyIds, setStoryIds] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const applicationId = state.application?.application_id;
  const applicationStatus = state.application?.status;
  const sourceIssues = new Set(today?.run?.diagnostics
    .filter((item) => item.requested && (item.error || item.timed_out))
    .map((item) => `${item.source} · ${item.error || "Timed out"}`));

  useEffect(() => {
    let cancelled = false;
    void Promise.all([
      request<Today>("/today"),
      request<{ applications: Application[] }>("/applications"),
      request<Preparation>("/interview-prep"),
      request<{ watchlists: SavedSearch[] }>("/watchlists"),
    ]).then(([daily, tracker, preparation, watchlists]) => {
      if (cancelled) return;
      setToday(daily); setApplications(tracker.applications); setPrep(preparation); setSearches(watchlists.watchlists);
    }).catch((caught) => { if (!cancelled) setError(String(caught)); });
    return () => { cancelled = true; };
  }, [applicationId, applicationStatus]);

  async function action(work: () => Promise<void>) {
    setBusy(true); setError(""); setNotice("");
    try { await work(); }
    catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
    finally { setBusy(false); }
  }

  async function refreshTracker() {
    const [tracker, daily] = await Promise.all([
      request<{ applications: Application[] }>("/applications"), request<Today>("/today"),
    ]);
    setApplications(tracker.applications); setToday(daily);
  }

  return <section className="block integrations-panel no-drag">
    <h2>Today</h2>
    {today && <p>{today.date} · {today.timezone}</p>}
    {error && <p role="alert">{error}</p>}
    {notice && <p role="status">{notice}</p>}
    <button disabled={busy || state.run.running || !state.candidate || Boolean(today?.run)} onClick={() => void action(async () => {
      setToday(await request<Today>("/today/run", "POST"));
    })}>Run today’s shortlist</button>
    {!state.candidate && <p>Upload your résumé to rank openings.</p>}
    <details>
      <summary>Saved searches</summary>
      {searches.length === 0 && <p>Today uses the default ML Engineer and Applied AI Engineer searches.</p>}
      {today?.run && <p>Changes apply on the next daily run.</p>}
      {searches.map((item) => <p key={item.watchlist_id}>
        {item.query} · {item.location || "United States"}{item.remote ? " · remote" : ""}
        {" "}<button disabled={busy} onClick={() => void action(async () => {
          await request(`/watchlists/${item.watchlist_id}`, "DELETE");
          setSearches((current) => current.filter((search) => search.watchlist_id !== item.watchlist_id));
        })}>Remove search</button>
      </p>)}
      <label>Role <input value={query} onChange={(event) => setQuery(event.target.value)} /></label>
      <label>Location <input value={location} onChange={(event) => setLocation(event.target.value)} /></label>
      <label><input type="checkbox" checked={remote} onChange={(event) => setRemote(event.target.checked)} /> Remote</label>
      <button disabled={busy || !query.trim()} onClick={() => void action(async () => {
        const item = await request<SavedSearch>("/watchlists", "POST", { query, location, remote });
        setSearches((current) => [...current, item]); setQuery(""); setLocation(""); setRemote(false);
      })}>Save search</button>
    </details>
    {today?.run?.ranked.map((row) => <article key={row.job.job_id}>
      <h3>{row.job.title} · {row.job.company}</h3>
      <p>{row.job.location} · {row.fit_score}/100 · {row.eligibility_status}</p>
      <p>{row.fit_explanation}</p>
      <button disabled={busy || state.run.running || !state.candidate || row.eligibility_status === "blocked"}
        onClick={() => void action(async () => {
          await request(`/today/jobs/${encodeURIComponent(row.job.job_id)}/tailor`, "POST");
          setNotice("Tailoring started. Look for the application pack below.");
          await refreshTracker();
        })}>Tailor application</button>
    </article>)}
    {today?.run?.warnings.map((warning) => <p key={warning}>{warning}</p>)}
    {[...sourceIssues].map((issue) => <p key={issue}>{issue}</p>)}
    <h3>Follow-ups due</h3>
    {today && today.followups.length === 0 && <p>None due.</p>}
    {today?.followups.map((item) => <p key={`${item.application_id}:${item.due}`}>
      {item.title} · {item.company} · {item.kind === "thank_you" ? "Thank-you" : "Follow-up"}
      {" "}<button disabled={busy} onClick={() => void action(async () => {
        await request(`/applications/${item.application_id}/followup`, "POST", { due: item.due });
        setToday(await request<Today>("/today"));
      })}>Mark done</button>
    </p>)}
    {today?.run && <details>
      <summary>Daily digest · {today.run.delivery.status}</summary>
      <pre style={{ whiteSpace: "pre-wrap" }}>{today.digest}</pre>
      {today.run.delivery.error && <p>{today.run.delivery.error}</p>}
    </details>}
    <details>
      <summary>Import application tracker</summary>
      <label>Export source <select value={source} onChange={(event) => { setSource(event.target.value); setPreview(null); }}>
        <option value="base44">JobReach (Base44 JSON or CSV)</option>
        <option value="career-ops">Career-Ops (Markdown or CSV)</option>
        <option value="sheets">Google Sheets (CSV)</option>
      </select></label>
      <label>Paste export <textarea value={content} onChange={(event) => { setContent(event.target.value); setPreview(null); }} /></label>
      <button disabled={busy || !content.trim()} onClick={() => void action(async () => {
        setPreview(await request<Preview>("/applications/import/preview", "POST", { source, content }));
      })}>Preview import</button>
      {preview && <>
        <p>{preview.created} new · {preview.matched} matched · {preview.skipped} skipped</p>
        {preview.warnings.slice(0, 10).map((warning) => <p key={warning}>{warning}</p>)}
        {preview.rows.slice(0, 10).map((row, index) => <p key={index}>{row.company} · {row.title} · {row.status}</p>)}
        <button disabled={busy || preview.rows.length === 0} onClick={() => void action(async () => {
          const result = await request<Preview>("/applications/import", "POST", { source, content, preview_id: preview.preview_id });
          setPreview(null); setContent("");
          setNotice(`Imported ${result.created} new records and matched ${result.matched} existing records.`);
          await refreshTracker();
        })}>Apply reviewed import</button>
      </>}
    </details>
    <details>
      <summary>Interview prep and story bank</summary>
      <label>Application <select value={selected} onChange={(event) => {
        const item = applications.find((record) => record.application_id === event.target.value);
        setSelected(event.target.value); setNotes(item?.interview_notes ?? ""); setStoryIds(item?.interview_story_ids ?? []);
      }}>
        <option value="">Choose an application</option>
        {applications.map((item) => <option key={item.application_id} value={item.application_id}>
          {item.title} · {item.company} · {item.status}
        </option>)}
      </select></label>
      {selected && <>
        <label>Interview notes <textarea value={notes} onChange={(event) => setNotes(event.target.value)} /></label>
        {Object.entries(prep.stories).map(([id, item]) => <label key={id}>
          <input type="checkbox" checked={storyIds.includes(id)} onChange={(event) =>
            setStoryIds((current) => event.target.checked ? [...current, id] : current.filter((value) => value !== id))} />
          {item.title}
        </label>)}
        <button disabled={busy} onClick={() => void action(async () => {
          const updated = await request<Application>(`/applications/${selected}/interview-prep`, "PUT", { notes, story_ids: storyIds });
          setApplications((current) => current.map((item) => item.application_id === selected ? updated : item));
          setNotice("Interview prep saved.");
        })}>Save interview prep</button>
      </>}
      <label>STAR story <select value={storyId} onChange={(event) => {
        setStoryId(event.target.value); setStory(prep.stories[event.target.value] ?? emptyStory);
      }}>
        <option value="">Write a new story</option>
        {Object.entries(prep.stories).map(([id, item]) => <option key={id} value={id}>{item.title}</option>)}
      </select></label>
      {storyFields.map((field) => <label key={field} style={{ textTransform: "capitalize" }}>
        {field}<textarea value={story[field]} onChange={(event) => setStory((current) => ({ ...current, [field]: event.target.value }))} />
      </label>)}
      <button disabled={busy || !story.title.trim()} onClick={() => void action(async () => {
        const path = storyId ? `/interview-prep/stories/${storyId}` : "/interview-prep/stories";
        setPrep(await request<Preparation>(path, storyId ? "PUT" : "POST", story));
        setStoryId(""); setStory(emptyStory); setNotice("Story saved.");
      })}>Save story</button>
      <label>Supporting materials URL <input type="url" value={resource ?? prep.supporting_url}
        onChange={(event) => setResource(event.target.value)} /></label>
      {prep.supporting_url && <p><a href={prep.supporting_url} target="_blank" rel="noreferrer">Open supporting materials</a></p>}
      <button disabled={busy || !resource?.trim()} onClick={() => void action(async () => {
        setPrep(await request<Preparation>("/interview-prep/resource", "PUT", { url: resource }));
        setResource(null); setNotice("Supporting link saved.");
      })}>Save supporting link</button>
      {prep.supporting_url && <button disabled={busy} onClick={() => void action(async () => {
        setPrep(await request<Preparation>("/interview-prep/resource", "PUT", { url: null }));
        setResource(null); setNotice("Supporting link removed.");
      })}>Remove supporting link</button>}
    </details>
  </section>;
}
