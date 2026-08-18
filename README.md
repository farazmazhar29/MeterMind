# MeterMind

Natural-language query and anomaly-explanation agent over energy time-series
data. A Claude-powered agent with native tool use answers questions about
electricity consumption across a fleet of industrial sites, combining
time-series queries (InfluxDB) with retrieved operational context (pgvector),
and is scored by a deterministic evaluation suite.

> **Status: Week 1 of 4 — data foundation.** The stack runs and the real data
> loads. The agent, the RAG layer and the eval harness are not built yet. The
> section headings below that describe them are placeholders, marked as such.
> Nothing in this README claims a capability that is not in the repository.

---

## Overview

*Placeholder — written once the agent works end to end.*

The intended demo: ask *"Frostlager Bremen drew far more power than usual for a
few days in March — what caused it?"* and get an answer that locates the
excursion in the time series, retrieves the maintenance note that explains it,
and connects the two. Neither store answers that alone.

---

## Architecture

*Placeholder — expand as components land. What exists today is marked ✅.*

```
                    ┌─────────────────┐
   question ───────►│  FastAPI  (wk3) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Agent loop     │   Claude API, native tool use.
                    │  (wk2)          │   Plain while-loop, no framework.
                    └────────┬────────┘
                             │ tool calls
              ┌──────────────┼──────────────┐
              │              │              │
      ┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼───────┐
      │ get_timeseries│ │detect_     │ │ search_docs │
      │ aggregate     │ │anomalies   │ │             │
      └───────┬──────┘ └─────┬──────┘ └─────┬───────┘
              │              │              │
       ┌──────▼──────────────▼───┐   ┌──────▼────────┐
       │      InfluxDB 2.7    ✅ │   │ Postgres +    │
       │  grid_load  (real)      │   │ pgvector   ✅ │
       │  site_consumption (syn) │   │ sites ✅ docs │
       └─────────────────────────┘   └───────────────┘
```

**The agent does not write Flux.** It calls typed Python tools that build
parameterised queries internally. Three reasons, and they are the honest ones:
generated query syntax fails in ways that are hard to evaluate; a public repo
that executes model-generated queries is a liability; and a fixed tool surface
makes *tool-selection accuracy* measurable separately from answer accuracy,
which is most of what makes the eval suite worth having.

**No LangChain or LangGraph in v1.** The agent loop is a `while` loop over
`client.messages.create` with a `tools=[...]` parameter, appending `tool_result`
blocks. It is short enough to read in one sitting and explain without hedging.

### Why InfluxDB 2.7 rather than 3.x

InfluxDB 3 Core — the free OSS tier — limits queries to roughly the last 72
hours of data. This project queries months of history, so 3 Core is unusable
here. 2.7 works, at the cost of Flux being on a deprecation path upstream.

---

## Setup

Requires Docker Desktop and Python 3.11+.

```bash
git clone <repo> && cd MeterMind
cp .env.example .env          # defaults work for local dev

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

docker compose up -d          # InfluxDB + Postgres/pgvector
docker compose ps             # wait for both to report (healthy)

python scripts/build_dataset.py
pytest                        # 27 tests, no containers needed
```

The build downloads ~124 MB of real data on first run and caches it. Useful
flags: `--dry-run` (generate and print statistics, write nothing), `--only-real`
(skip the synthetic sites), `--force-download`.

Then check the data landed:

```
InfluxDB UI    http://localhost:8086     (user/password from .env)
Postgres       psql postgresql://metermind:...@localhost:5432/metermind
```

---

## Data: real vs synthetic

This distinction matters and the repository is built to keep it unambiguous —
the two kinds of data live in **separate InfluxDB measurements**, so the
boundary is enforced by the schema rather than by a comment.

### Real ✅

| | |
|---|---|
| Source | [Open Power System Data](https://open-power-system-data.org/), `time_series` package, version `2020-10-06` |
| Series | `DE_load_actual_entsoe_transparency` — German national electricity load, hourly, MW |
| Origin | ENTSO-E Transparency Platform, via the German TSOs |
| Coverage | 2015-01-01 to 2020-09-30, 50,400 points |
| Measurement | `grid_load`, tag `source=opsd` |

Loaded at its **true timestamps**, unmodified.

### Synthetic ⚠️

Eight fictional industrial sites across four categories (manufacturing, office,
cold storage, retail). Companies, addresses, meter IDs and nameplate ratings are
all invented — see [data/generated/sites.yaml](data/generated/sites.yaml).

| | |
|---|---|
| Measurement | `site_consumption`, tags `site_id` + `category` |
| Resolution | 15 minutes, mirroring real German industrial submetering |
| History | 548 days ending at build time |
| Reproducibility | fixed seed in `.env`; parameters recorded in `data/generated/build_manifest.json` |

**What is derived from the real data, precisely:** the *seasonal envelope* only
— the winter/summer swing and the genuine weather-driven day-to-day variation,
taken as a daily-resolution multiplier from the OPSD series.

**What is not:** the intra-day and weekly shape. The national curve peaks in the
evening because it is dominated by aggregate residential and commercial demand.
A factory on two shifts, or a cold store that never switches off, looks nothing
like that. Passing the national daily cycle through to all eight sites would
make them near-perfectly correlated and obviously fake. Per-category load
profiles supply that shape instead.

### The time shift, and why it exists

The OPSD package was last published in October 2020 and is no longer updated.
Questions like *"what happened last Tuesday"* have nothing to land on.

The synthetic sites are therefore built on a **time-shifted** copy of the real
seasonal signal. The offset is always a whole multiple of **364 days (52 weeks)**
— currently 2184 days — which is the only step satisfying both constraints:

- 364 is divisible by 7, so **every timestamp keeps its weekday**. Mondays stay
  Mondays, which matters when the entire dataset is built on weekday/weekend
  behaviour.
- 364 is within 1.25 days of a calendar year, so **seasonality barely moves** —
  about 7 days of drift over six years. Shifting by whole *weeks* would preserve
  weekdays equally well but slide summer into autumn.

The real series is **not** shifted. Only the synthetic sites are, and the exact
offset is recorded in the build manifest.

### Injected anomalies

A small number of anomalies (spikes, drop-offs, drift) are deliberately injected
so there is something genuine to detect and explain. Ground truth — site, type,
window, magnitude, and a written **cause** — is committed to
[data/generated/anomalies.json](data/generated/anomalies.json) and is what the
eval suite scores against. In Week 2 each `cause` is expanded into a maintenance
note in the RAG corpus, which is what makes the "why" questions answerable.

German public holidays are modelled as expected behaviour, not anomalies. A
holiday looks like a drop-off, and a detector that flags all of them is not
detecting anything.

---

## Evaluation

*Results placeholder — filled in once the harness runs.*

The question set was written **before** any agent code, which is the only
ordering under which it means anything. 24–30 questions across three categories:

| Category | Tests |
|---|---|
| `timeseries_lookup` | direct queries — peaks, totals, comparisons |
| `anomaly` | detection and timing against injected ground truth |
| `contextual_rag` | questions needing site metadata or operational notes alongside the series |

**No expected answer is hand-written.** Each question carries a `verify` block
naming a deterministic function; `resolve_gold` runs those against the loaded
dataset to materialise the gold answers. Since the dataset is regenerated from a
seed and its window slides forward, a hardcoded number would be wrong within a
day — and a silently wrong eval suite is worse than none.

See [eval/questions.yaml](eval/questions.yaml).

| Metric | Result |
|---|---|
| Answer accuracy | *not yet measured* |
| Tool-selection accuracy | *not yet measured* |
| Anomaly detection F1 | *not yet measured* |
| Refusal rate on unanswerable questions | *not yet measured* |

---

## Repository layout

```
docker-compose.yml         InfluxDB 2.7 + Postgres/pgvector
docker/postgres/           init.sql (bootstrap) + schema.sql (idempotent)
data/generated/            committed ground truth: sites, anomalies, manifest
data/raw/                  gitignored — the 124 MB OPSD download
src/metermind/
  config.py                one source of truth, shared with docker-compose
  data/opsd.py             real data: download, parse, time-shift, envelope
  data/profiles.py         per-category load shapes            (week 1, in progress)
  data/synth.py            site generation                     (week 1, in progress)
  data/anomalies.py        injection + ground truth            (week 1, in progress)
  data/load_influx.py      InfluxDB writes
  data/load_postgres.py    schema + site register
  rag/                     (week 2)
  agent/                   (week 2)
  api/                     (week 3)
scripts/build_dataset.py   one-command pipeline
eval/questions.yaml        the evaluation set
docs/CONTRACTS.md          module interfaces
tests/
```

---

## Licence and attribution

Underlying load data is published by [Open Power System
Data](https://open-power-system-data.org/) under the MIT licence, sourced from
the ENTSO-E Transparency Platform and the German TSOs. All site identities,
consumption series and operational documents in this repository are synthetic.
