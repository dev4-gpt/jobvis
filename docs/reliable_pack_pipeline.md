# Reliable application-pack pipeline

Jobvis treats a tailored application as a build artifact, not as text that is
ready merely because a model returned JSON.

The pipeline is:

1. Generate a corpus-grounded pack.
2. Run the deterministic backtest for grounding, policy safety, letter quality,
   source links, and CV density.
3. Render the CV and cover letter with Tectonic when available.
4. Re-open the PDFs and check extracted text, page counts, link annotations,
   known corruption markers, and required companion `.tex` files.
5. Apply only bounded repairs, then repeat the complete audit.
6. Expose PDF downloads only when the final manifest is `ready`.

If the final manifest is `withheld`, the UI shows the issue code and keeps the
`.tex` files available for inspection. A failed PDF is never labelled as a
ready application asset.

## Local checks

Run the deterministic checks before sharing a pack:

```bash
make doctor
make ci
make gates
make checkmate

# Or run the same release sequence with one command:
make release-check
```

Run one provider-backed smoke test against a local resume only when you want a
real end-to-end search and tailoring run:

```bash
make pack-e2e CV=/Users/aryamandev/Downloads/Aryaman_resume.pdf YES=1
```

The smoke test never submits an application. Personal resumes and generated
packs remain local; CI uses synthetic fixtures.

## Similar-role search

The target-search panel accepts one additional title per line. These titles
are merged with the selected role families, capped by `SCOUT_MAX_ROLE_QUERIES`,
deduplicated, and passed through the same full-time, timing, location,
authorization, sponsorship, and clearance policy.
