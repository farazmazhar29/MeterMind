# Module contracts

This file defines the seam between the plumbing modules and the modelling
modules. The plumbing is written; the modelling is not. Anything that matches
the signatures below will drop straight into `scripts/build_dataset.py`.

**Written already (plumbing):**
`config.py`, `data/opsd.py`, `data/load_influx.py`, `data/load_postgres.py`,
`scripts/build_dataset.py`, `data/generated/sites.yaml`

**To be written (modelling):**
`data/profiles.py`, `data/synth.py`, `data/anomalies.py`

`build_dataset.py` runs today. It downloads and loads the real OPSD series, then
prints exactly which of the three modules is missing and skips the synthetic
half. As each module lands, more of the pipeline lights up.

---

## The core idea: envelope x shape

The real German national load curve supplies the **seasonal envelope** — how
demand rises in January and falls in August, plus real weather-driven
day-to-day wiggle. That is genuine measured data, and it is the honest anchor.

It must NOT supply the intra-day shape. National demand peaks in the evening;
a manufacturing plant peaks during shifts and a retail store is shut on Sunday.
If every site inherits the national daily curve, all eight sites correlate at
~1.0 and the dataset is transparently fake to anyone in the energy sector.

So `opsd.seasonal_envelope()` deliberately strips the intra-day cycle and
returns a daily-resolution multiplier only. Supplying the intra-day and weekly
shape is what `profiles.py` is for.

```
power_kw(t) = base_load_kw
            + (peak_load_kw - base_load_kw)
              * category_shape(t)          <- profiles.py   (yours)
              * seasonal_envelope(t)       <- opsd.py       (written)
              * weather_sensitivity        <- profiles.py   (yours)
              + noise(t)                   <- synth.py      (yours)
```

That formula is a suggestion, not a requirement. It is the seam that matters.

---

## `data/profiles.py`

```python
def category_shape(
    category: str,
    index: pd.DatetimeIndex,   # tz-aware UTC
    site: dict,                # one entry from sites.yaml
) -> pd.Series:
    """Dimensionless 0..1 load shape for one site category.

    Returns a Series indexed by `index`. 0.0 means "at base load",
    1.0 means "at nameplate peak".
    """
```

Requirements this needs to satisfy:

| Category | Expected behaviour |
|---|---|
| `manufacturing` | two weekday shifts ~06:00-22:00 local, hard drop to base load at weekends, low weather sensitivity |
| `office` | ~07:00-19:00 weekdays, strong weekend collapse, meaningful HVAC weather term |
| `cold_storage` | near-flat 24/7, weak weekly pattern, strong temperature sensitivity |
| `retail` | ~09:00-20:00, Saturday peak, **closed Sunday** (German retail law) |

**Local time, not UTC.** `index` is UTC. A shift starting at 06:00 means 06:00
in `Europe/Berlin`, which is UTC+1 or UTC+2 depending on the date. Convert with
`index.tz_convert("Europe/Berlin")` before taking `.hour`, or every shift
boundary will be an hour off for half the year.

**German public holidays** must behave like weekends, or every holiday reads as
a false anomaly. `holidays` is already a declared dependency:

```python
import holidays
de = holidays.Germany(years=range(2024, 2027))   # optionally subdiv= per city
```

Holidays are *expected* behaviour, so they must not appear in the anomaly
manifest.

---

## `data/synth.py`

```python
def generate_site_series(
    site: dict,                 # one entry from sites.yaml
    envelope: pd.Series,        # from opsd.seasonal_envelope(), reindexed onto `index`
    index: pd.DatetimeIndex,    # exact target timestamps, tz-aware UTC
    rng: np.random.Generator,   # seeded; do not create your own
) -> pd.Series:
    """Return power in kW, indexed by `index`, float64, no NaNs."""
```

Hard requirements:

- Index must equal `index` exactly. No NaNs, no negatives.
- Use the passed `rng` only. A fresh `np.random.default_rng()` or anything
  touching the global `np.random` breaks reproducibility, and the eval gold
  answers are computed from this data — if it changes between runs, every
  numeric expected answer silently becomes wrong.
- Stay roughly within the site's `base_load_kw`..`peak_load_kw` envelope. These
  are nameplate figures from `sites.yaml`; brief excursions above peak are
  realistic, sustained ones are not.

---

## `data/anomalies.py`

```python
def inject(
    series_by_site: dict[str, pd.Series],   # site_id -> kW series from synth
    sites: list[dict],                      # sites.yaml entries
    rng: np.random.Generator,
) -> tuple[dict[str, pd.Series], list[dict]]:
    """Return (modified series, anomaly records).

    Records are written verbatim to data/generated/anomalies.json and are the
    ground truth the eval suite scores category (b) questions against.
    """
```

Record schema — every field required:

```json
{
  "anomaly_id": "anom-001",
  "site_id": "frostlager-bremen",
  "type": "spike",
  "start_utc": "2026-03-14T02:00:00+00:00",
  "end_utc": "2026-03-16T22:00:00+00:00",
  "peak_deviation_pct": 180.0,
  "detectability": "obvious",
  "cause": "Chiller #2 replaced; the replacement compressor ran continuously for three days during commissioning before the controller was retuned.",
  "cause_doc_type": "maintenance_log"
}
```

### `cause` is load-bearing — do not leave it as a stub

It looks like documentation. It is not. In Week 2 the RAG corpus generator
expands each `cause` into a maintenance log or incident report, and that
document is what makes eval category (c) work: the agent finds the anomaly in
InfluxDB, retrieves the note from pgvector, and explains *why*. That join is
the demo the whole project exists to show.

Anomalies injected without a written cause mean regenerating the dataset in
Week 2 — and regenerating invalidates every numeric eval answer already
computed. Write the cause at injection time.

### Type and timing guidance

| Type | Shape | Duration |
|---|---|---|
| `spike` | sharp multiplicative excursion above normal | hours to ~3 days |
| `dropoff` | sustained fall toward or below base load | hours to ~2 weeks |
| `drift` | slow monotonic baseline creep, e.g. fouling heat exchanger | 2-8 weeks |

Aim for 12-20 anomalies across all sites and the full history. Mix
`detectability`: some should be obvious at a glance, some should need real
analysis — an eval suite where everything is trivially detectable measures
nothing.

`drift` has a genuinely fuzzy onset, which is why `eval/questions.yaml` scores
timing per type: 60 minutes for spike/dropoff, 5 days for drift. Set `start_utc`
to the point where deviation first becomes *sustained*, not the first sample
that happens to tick upward.

Do not place anomalies on German public holidays. A holiday already looks like
a drop-off, and the resulting ambiguity is untestable.

---

## Reproducibility rule

`METERMIND_SEED` in `.env` fixes the entire dataset. Same seed plus same OPSD
version must give the same numbers, because `eval/gold.json` is computed from
the loaded data and cached. `build_dataset.py` records the seed, the resolved
time offset and the OPSD version in `data/generated/build_manifest.json`.

If you change the modelling code in a way that shifts the numbers, regenerate
the gold answers too. That is a deliberate, visible step — not something to
discover when eval scores drop for no reason.
