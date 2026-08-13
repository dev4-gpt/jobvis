# System design lessons

Job Scout is deliberately small enough to read end to end. The important
design lesson is not the number of framework packages; it is where each kind of
decision belongs.

## The execution model

The system has one typed LangGraph state, plain Python nodes, and explicit
edges. CV extraction happens once before the graph. Search then follows:

```text
Profile -> fetch_jobs -> rank_jobs -> conditional reformulation -> end
                              |
                              +-> selected job -> tailor -> validate -> render
```

Most edges are fixed. The conditional edge after ranking is agentic because the
current result state determines whether the system broadens the query. The
reformulation count and LLM call budget keep that decision bounded.

## Deterministic versus probabilistic work

LLMs are used for profile extraction, search argument selection, ranking,
reformulation, and tailoring. Pydantic schemas validate their shape, but shape
validation does not prove truth.

The source cascade, deduplication, location preference application, call
budget, checkpoint routing, fabrication validator, and LaTeX escaping are
deterministic. Keeping those responsibilities in code gives us tests that can
disagree with a model and makes failures diagnosable.

## Ports and adapters

`JobSource` is the port. JSearch, Adzuna, Remotive, and the committed cache are
adapters. Each adapter normalizes into `JobPosting`; the graph never needs to
know a provider's response format.

The same principle applies to model providers. `get_chat_model` owns provider
selection, while nodes own the task contract and structured-output schema.

## State and checkpoints

The thread checkpoint is the handoff between search and tailoring. Search writes
the profile, CV text, preferences, jobs, and ranked jobs. Tailoring receives only
the selected job id and reads the rest from the checkpoint.

Because state persists, every search invocation must explicitly set
`selected_job_id=None`. Omitting it can route a new search into an old tailoring
run. This is why the runner owns invocation envelopes instead of letting UI
handlers construct graph state ad hoc.

## Observability as a contract

Opik traces are useful for investigating model behavior, but traces are not a
substitute for application state. `SourceDiagnostic` records source latency,
timeouts, result counts, errors, and contribution in the typed result itself.
That makes the same facts available to the UI, tests, batch reports, and traces.

## Safety boundaries

Job descriptions and company research are untrusted input. They are evidence for
the model, not instructions to the model. Generated CV text is escaped before
LaTeX rendering. Generated claims must retain corpus references and remain
visible to a human reviewer. The agent never submits an application.
