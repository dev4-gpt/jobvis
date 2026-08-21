# Application-pack backtesting

Jobvis runs a deterministic backtest before it presents a tailored pack. The
backtest is a safety gate, not another LLM judge: it does not browse, spend
tokens, or silently rewrite unsupported claims.

It checks five dimensions:

- grounding: CV and cover-letter claims remain traceable to the uploaded CV;
- policy safety: authorization, sponsorship, visa, clearance, and other
  unresolved claims are not presented as facts;
- cover-letter quality: 250–350 words, at least two evidence matches, and at
  least two genuine job-requirement matches;
- source links: clickable links extracted from the original resume survive in
  the tailored CV model;
- CV density: the tailored CV remains substantive instead of collapsing into a
  short summary.

If a pack fails, Jobvis may make at most `SCOUT_TAILOR_MAX_REPAIRS` bounded
repair calls (default `2`). Each candidate is measured again and is accepted
only when its deterministic score is strictly higher. The loop stops when the
pack passes, a repair is unavailable, a repair does not improve the score, or
the finite attempt limit is reached. It never loops until a model says it is
happy and never submits an application.

Set `SCOUT_TAILOR_MAX_REPAIRS=0` to disable repair calls while retaining the
backtest. Any final failure remains visible in the application-pack warning so
you can review it before sending.

## Offline command

The graph uses the same evaluator automatically. For a saved typed pack, the
standalone command is:

```bash
make backtest CV=/path/to/Aryaman_resume.pdf PACK=/path/to/pack.json JOB=/path/to/job-description.txt
```

It exits `0` only when every contract passes and writes the complete report to
stdout. Add `OUTPUT=/path/to/report.json` to save it as well. This command is
offline and is suitable for fixture-based regression tests.

The improvement loop is deliberately bounded and non-regressing. A better
letter cannot compensate for a lost resume link or an unsupported claim, since
grounding and policy safety carry the highest weights and all dimensions must
pass before the pack is considered ready.

For a real local search→tailor→render→audit smoke test, use the explicit
provider-backed command:

```bash
make pack-e2e CV=/Users/aryamandev/Downloads/Aryaman_resume.pdf YES=1
```

This is intentionally opt-in. It uses the configured providers and sources,
does not add personal material to CI or evaluation datasets, and never submits
an application.
