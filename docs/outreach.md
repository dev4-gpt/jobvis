# Manual hiring-manager outreach

Jobvis can extract an explicitly visible email address from a public employer
page that you select and approve, then generate a job-specific email and short
video script from the current checkpointed job and tailored pack.

It deliberately does not:

- guess `firstname@company.com` patterns;
- scrape LinkedIn or Indeed;
- buy or enrich personal contact data;
- send email, upload a video, or automate repeated outreach.

The safe flow is:

1. Open the employer’s public team, careers, or listing page.
2. Copy the relevant visible text and provide its URL to
   `POST /api/contacts/discover`.
3. Manually verify the person, role, address, and employer contact policy.
4. Tailor a selected Jobvis application.
5. Click `Why me · email/video` in the voice console, or call
   `POST /api/outreach/generate` with the verified contact.
6. Personalize the opening, record the video yourself, and send it manually.

Example contact extraction:

```bash
curl -s http://localhost:8003/api/contacts/discover \
  -H 'content-type: application/json' \
  -d '{"sources":[{"url":"https://example.com/team","source_type":"company_page","text":"Jordan Lee, Head of Engineering — jordan@example.com"}]}'
```

The generated draft includes the job requirements it used, evidence references,
a `why_me` section, an email body, a concise video script, and manual-review
warnings. The draft is stored only in the local Jobvis data directory.

## Artifact quality loop

After downloading a CV and cover letter, run the deterministic artifact audit:

```bash
make pack-audit \
  ORIGINAL_CV=/Users/aryamandev/Downloads/Aryaman_resume.pdf \
  TAILORED_CV="/Users/aryamandev/Downloads/tailored_cv (2).pdf" \
  COVER_LETTER_TEXT=/path/to/cover-letter.txt \
  JOB=/path/to/job-description.txt
```

The audit checks PDF readability, page density, the 12 original clickable
links, PDF annotations, cover-letter length, job-requirement matches, and known
corruption/internal-marker patterns. A nonzero exit means the pack remains a
draft and should not be sent.

The running console performs the same audit automatically when a pack appears.
Use **Audit pack** to rerun it. PDF downloads are gated by the result; when a
blocker is found, the API returns the failing checks and the `.tex` downloads
remain available for correction. No model call is made by this audit.
