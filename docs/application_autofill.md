# Review-gated application preparation

Jobvis prepares an application; a person submits it. The local browser layer
exists to remove repetitive typing while keeping the consequential action in
human hands.

Each selected job has separate listing and direct-application links. The local
tracker stores the job, asset manifest, review events, and current status under
macOS Application Support (or `JOBVIS_DATA_DIR`). It never stores passwords,
MFA answers, or a submit endpoint.

The allowed status path is:

`discovered → saved → tailored → reviewed → opened → safe_fields_filled → final_review → submitted_by_user`

The final transition is a manual record only. Jobvis opens the employer page,
fills only explicitly approved safe fields, and pauses before Submit.

## Flow

1. Select a ranked job and choose **Open application**.
2. Jobvis opens the URL in a visible Chromium-compatible browser. Greenhouse,
   Lever, and Ashby are detected from the URL/page.
3. Log in, complete MFA, solve CAPTCHA, or create an account yourself.
4. Review the proposed field mapping. It contains confidence, provenance, and
   a sensitivity flag.
5. Approve safe fields. Each file upload is its own approval: approving a name,
   email, or other text field never implicitly uploads the tailored CV or cover
   letter. Approved upload fields receive the matching PDFs when available.
6. Sensitive, ambiguous, and unknown questions pause for an answer. The
   resume is never used to guess an ambiguous response.
7. Jobvis stops at the final review page. There is no submit method or submit
   endpoint; click the employer's Submit button yourself.

The parser and adapter contract is deliberately pure enough to test against
local HTML fixtures. Page text is untrusted data: no instructions in a job
listing or form can change the workflow.

## Local setup

The browser dependency is optional and is excluded from normal keyless CI:

```bash
uv sync --extra application
uv run playwright install chromium
```

Jobvis prefers an installed Brave binary when available. Set
`JOBVIS_BROWSER_EXECUTABLE` to another Chromium-compatible executable if
needed. The browser is visible by design.

Passwords never enter Jobvis. Browser session state is encrypted before being
written to `data/private/application/browser_state.enc`; the encryption key is
stored through the operating system keychain by `keyring`. Confirmed answer
memory uses the same protected store. Sensitive answers require consent to
remember and are presented for confirmation on every application.

`data/private/` is ignored by Git. Never commit a Playwright storage state,
browser profile, cookies, answer memory, CV, or generated application pack.
Playwright authentication state is impersonation-capable, so treat it like a
password even though Jobvis does not print or upload it.

## Voice actions

ElevenLabs is only a voice control surface over the same process-wide
checkpoint. It can open a selected application and request filling of field
IDs already approved on screen. It cannot receive credentials, answer a
sensitive question, or submit an application.

Browserbase is intentionally not the default. A future adapter may use it only
with an explicit remote-session decision and a separate privacy review.
