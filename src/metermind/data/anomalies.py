"""Inject anomalies into the synthetic series and record their ground truth.

The records this returns ARE the eval suite's answer key for every "which site
behaved unusually" question, so they have to be exact.

Two design points worth knowing if anyone asks:

1. Anomalies are tapered, never rectangular. A hard-edged anomaly is detectable
   purely by its vertical edge, which would test edge detection rather than
   anomaly detection.

2. Roughly half are deliberately SUBTLE -- a 15% drop-off, an 8% drift -- which
   sits inside the range of ordinary business variation that synth.py already
   generates. If every anomaly were a 200% spike, detection F1 would be a
   meaningless 1.0 and the eval suite would measure nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metermind.data import profiles

ANOMALIES_PER_SITE = (1, 3)

# Hours. A spike is over in days; a drift takes weeks to become visible.
DURATION_HOURS = {
    "spike": (6, 72),
    "dropoff": (12, 336),
    "drift": (336, 1344),
}

# Fractional deviation, by type and by how hard it should be to spot.
DEVIATION = {
    ("spike", "obvious"): (0.80, 2.00),
    ("spike", "subtle"): (0.25, 0.50),
    ("dropoff", "obvious"): (0.50, 0.80),
    ("dropoff", "subtle"): (0.15, 0.30),
    ("drift", "obvious"): (0.20, 0.35),
    ("drift", "subtle"): (0.08, 0.15),
}

# Keep clear of the window edges so a detector always has clean data either side.
EDGE_MARGIN_DAYS = 21

# ---------------------------------------------------------------------------
# CAUSE LIBRARY
# ---------------------------------------------------------------------------
# Each of these strings is expanded into a document in the Week 2 RAG corpus,
# and that document is what the agent has to retrieve to answer "what caused
# it?". This is the content the whole contextual_rag eval category rests on.
#
# Three rules they all follow:
#   * Name specific equipment ("Chiller #2", "the stamping line"). Generic text
#     gives the retriever nothing to match on and makes RAG look worse than it is.
#   * Vary doc_type -- Week 2 filters on it.
#   * NO DURATIONS OR SEASONS IN THE TEXT. A cause is chosen independently of the
#     window that gets picked for it, so "closed for a two week refit" will sooner
#     or later be attached to a three day drop-off and contradict its own data.
#     The Week 2 corpus generator has the real dates from this manifest and can
#     render them; the cause string is a seed, not the finished document.
#
# Edit the wording freely; it is meant to sound like your operations team wrote it.

CAUSES = {
    ("manufacturing", "spike"): [
        ("Powder coating oven 3 was recommissioned after a burner replacement and held "
         "at full output overnight while the new control loop was tuned.",
         "commissioning"),
        ("A compressed air main developed a leak in the ceiling void; the compressors ran "
         "near-continuously until the leak was traced and sealed.",
         "incident"),
    ],
    ("manufacturing", "dropoff"): [
        ("The stamping line was down for a die change that overran waiting "
         "on a replacement part.",
         "maintenance_log"),
        ("Annual works shutdown. Only security lighting and compressed air stayed on.",
         "shift_notice"),
    ],
    ("manufacturing", "drift"): [
        ("Press hydraulics were running against a partially blocked return filter, so pump "
         "load climbed steadily until the filter was changed at the next service.",
         "maintenance_log"),
        ("A second night shift was phased in over several weeks as an export order ramped up.",
         "shift_notice"),
    ],
    ("office", "spike"): [
        ("A building management system fault left the chillers running overnight and through "
         "the weekend before anyone noticed.",
         "incident"),
        ("The heat recovery damper actuator failed closed during a warm spell, forcing the "
         "cooling plant to run at full capacity.",
         "incident"),
    ],
    ("office", "dropoff"): [
        ("Two of the four tenant floors sat vacant between leases; lighting and terminal "
         "units were isolated on those floors.",
         "contract"),
        ("The main air handling unit was offline for a coil replacement, so ventilation ran "
         "on the reduced-capacity backup unit.",
         "maintenance_log"),
    ],
    ("office", "drift"): [
        ("Supply air filters loaded progressively between quarterly service visits, raising "
         "fan power until they were replaced.",
         "maintenance_log"),
        ("Server room load grew steadily as test hardware was added without a "
         "corresponding cooling upgrade.",
         "incident"),
    ],
    ("cold_storage", "spike"): [
        ("Chiller #2 was replaced; the new compressor ran continuously during "
         "commissioning before the controller was retuned.",
         "commissioning"),
        ("A dock door seal failed, so the loading bay ran warm and the compressors could not "
         "cycle down.",
         "incident"),
    ],
    ("cold_storage", "dropoff"): [
        ("Half the freezer hall was emptied for floor repairs and the associated evaporators "
         "were isolated.",
         "maintenance_log"),
        ("Chiller #1 was taken offline for a compressor rebuild; the remaining plant held "
         "temperature at reduced capacity.",
         "maintenance_log"),
    ],
    ("cold_storage", "drift"): [
        ("Condenser coils fouled progressively; efficiency fell away until "
         "the next scheduled clean.",
         "maintenance_log"),
        ("A refrigerant undercharge developed slowly from a weeping joint on the liquid line, "
         "increasing compressor run time week by week.",
         "incident"),
    ],
    ("retail", "spike"): [
        ("Night blinds on the chilled aisle were left retracted after a merchandising change, "
         "so the cases ran hard overnight.",
         "incident"),
        ("A defrost timer failed on the frozen food pack, leaving the heaters energised far "
         "longer than scheduled.",
         "maintenance_log"),
    ],
    ("retail", "dropoff"): [
        ("The refrigeration pack tripped overnight; stock was moved to a neighbouring store "
         "while it was repaired.",
         "incident"),
        ("The store closed for a refit. Only the cold room and security systems "
         "stayed live.",
         "shift_notice"),
    ],
    ("retail", "drift"): [
        ("Door seals on the multideck cases degraded, so the pack ran "
         "progressively longer to hold temperature.",
         "maintenance_log"),
        ("Additional chilled display was added in stages during a category reset, raising the "
         "refrigeration base load.",
         "commissioning"),
    ],
}


# ---------------------------------------------------------------------------
# SITE-SPECIFIC CAUSES  (checked first; CAUSES above is the fallback)
# ---------------------------------------------------------------------------
# Category-level causes are not good enough on their own. Two sites can share a
# category and be completely different businesses -- nordwerk-kassel stamps
# sheet metal, hansawerk-hamburg makes chilled ready meals -- and a category
# lookup happily gave the food plant a powder coating oven.
#
# It also matters for retrieval quality. "Blast chiller on line 2" is a far
# better embedding target than "equipment fault", so site-specific text makes
# the Week 2 RAG numbers honest rather than artificially poor.
#
# Every string here is checked against that site's `notes` in sites.yaml.

SITE_CAUSES = {
    ("nordwerk-kassel", "spike"): [
        ("Powder coating oven 3 was recommissioned after a burner replacement and held "
         "at full output overnight while the new control loop was tuned.", "commissioning"),
    ],
    ("nordwerk-kassel", "dropoff"): [
        ("The stamping line was down for a die change that overran waiting on a "
         "replacement part.", "maintenance_log"),
    ],
    ("nordwerk-kassel", "drift"): [
        ("Press hydraulics were running against a partially blocked return filter, so "
         "pump load climbed steadily until the filter was changed at the next service.",
         "maintenance_log"),
    ],

    ("hansawerk-hamburg", "spike"): [
        ("The blast chiller on line 2 was left in continuous run after a controller "
         "reset, so process refrigeration never cycled down between batches.", "incident"),
    ],
    ("hansawerk-hamburg", "dropoff"): [
        ("The ready-meal lines were stopped for the annual deep clean and hygiene audit; "
         "only process refrigeration stayed live.", "shift_notice"),
    ],
    ("hansawerk-hamburg", "drift"): [
        ("A third night shift was phased in as the pre-Christmas order book built up.",
         "shift_notice"),
    ],

    ("sudpark-muenchen", "spike"): [
        ("A building management system fault left the chillers running overnight and "
         "through the weekend before anyone noticed.", "incident"),
    ],
    ("sudpark-muenchen", "dropoff"): [
        ("Two of the four tenant floors sat vacant between leases; lighting and terminal "
         "units were isolated on those floors.", "contract"),
    ],
    ("sudpark-muenchen", "drift"): [
        ("Supply air filters loaded progressively between quarterly service visits, "
         "raising fan power until they were replaced.", "maintenance_log"),
    ],

    ("techcampus-dresden", "spike"): [
        ("A cooling failure in the server room forced the backup units to run at full "
         "output alongside the primary plant.", "incident"),
    ],
    ("techcampus-dresden", "dropoff"): [
        ("The R&D wing was closed for a lab refit and its ventilation and process power "
         "were isolated.", "maintenance_log"),
    ],
    ("techcampus-dresden", "drift"): [
        ("Server room load grew steadily as test hardware was added without a "
         "corresponding cooling upgrade.", "incident"),
    ],

    ("frostlager-bremen", "spike"): [
        ("Chiller #2 was replaced; the new compressor ran continuously during "
         "commissioning before the controller was retuned.", "commissioning"),
    ],
    ("frostlager-bremen", "dropoff"): [
        ("Half the freezer hall was emptied for floor repairs and the associated "
         "evaporators were isolated.", "maintenance_log"),
    ],
    ("frostlager-bremen", "drift"): [
        ("Condenser coils on the oldest chiller fouled progressively; efficiency fell "
         "away until the next scheduled clean.", "maintenance_log"),
    ],

    ("kuehlhaus-duisburg", "spike"): [
        ("A dock door seal failed on the inbound bay, so the chilled hall ran warm and "
         "the compressors could not cycle down.", "incident"),
    ],
    ("kuehlhaus-duisburg", "dropoff"): [
        ("Inbound container volumes collapsed during a river closure, leaving much of "
         "the chilled hall empty and lightly loaded.", "incident"),
    ],
    ("kuehlhaus-duisburg", "drift"): [
        ("A refrigerant undercharge developed slowly from a weeping joint on the liquid "
         "line, increasing compressor run time week by week.", "incident"),
    ],

    ("marktplatz-koeln", "spike"): [
        ("Night blinds on the chilled aisle were left retracted after a merchandising "
         "change, so the cases ran hard overnight.", "incident"),
    ],
    ("marktplatz-koeln", "dropoff"): [
        ("The refrigeration pack tripped overnight; stock was moved to a neighbouring "
         "store while it was repaired.", "incident"),
    ],
    ("marktplatz-koeln", "drift"): [
        ("Door seals on the multideck cases degraded, so the pack ran progressively "
         "longer to hold temperature.", "maintenance_log"),
    ],

    ("stadtmarkt-leipzig", "spike"): [
        ("A defrost timer failed on the frozen food pack, leaving the heaters energised "
         "far longer than scheduled.", "maintenance_log"),
    ],
    ("stadtmarkt-leipzig", "dropoff"): [
        ("The hall closed for a refit of the bakery unit. Only the cold room and "
         "security systems stayed live.", "shift_notice"),
    ],
    ("stadtmarkt-leipzig", "drift"): [
        ("Additional chilled display was added in stages as two new tenants fitted out "
         "their units.", "commissioning"),
    ],
}


def _taper(n: int, edge: float = 0.15) -> np.ndarray:
    """Flat 1.0 with smooth cosine ramps at both ends.

    Same reasoning as _window in profiles.py: a compressor failure is not
    instantaneous, and a rectangular anomaly would be trivially detectable by
    its vertical edge alone.
    """
    ramp_len = max(1, int(n * edge))
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ramp_len)))
    profile = np.ones(n)
    profile[:ramp_len] = ramp
    profile[-ramp_len:] = ramp[::-1]
    return profile


def _pick_window(index, duration, holiday_set, taken, rng, attempts: int = 60):
    """Find a start time clear of holidays, window edges, and existing anomalies."""
    earliest = index[0] + pd.Timedelta(days=EDGE_MARGIN_DAYS)
    latest = index[-1] - pd.Timedelta(days=EDGE_MARGIN_DAYS) - duration
    if latest <= earliest:
        return None

    span_seconds = int((latest - earliest).total_seconds())

    # Only short anomalies dodge public holidays. A one-day drop-off on Tag der
    # Deutschen Einheit is genuinely ambiguous -- nobody could separate the
    # anomaly from the holiday, so it is untestable. A six-week drift will span
    # several holidays wherever it is placed, and is not ambiguous at all
    # because the trend either side of them is still visible.
    check_holidays = duration <= pd.Timedelta(days=7)

    for _ in range(attempts):
        start = earliest + pd.Timedelta(seconds=int(rng.integers(0, span_seconds)))
        end = start + duration

        if any(start <= taken_end and end >= taken_start for taken_start, taken_end in taken):
            continue

        if check_holidays:
            days = pd.date_range(start.floor("D"), end.ceil("D"), freq="D")
            if any(day.date() in holiday_set for day in days):
                continue

        return start, end
    return None


def _pick_cause(site: dict, kind: str, rng, used: set[str]) -> tuple[str, str]:
    """Choose a cause, preferring one not already used anywhere in the fleet.

    Site-specific text wins over the category fallback, because a category
    lookup cannot know that one manufacturing site stamps metal and the other
    makes ready meals.

    Reusing a cause string means Week 2 generates two identical maintenance
    notes, and retrieval then has no way to tell which anomaly a note explains.
    Falls back to the full pool once every option is spent.
    """
    options = SITE_CAUSES.get((site["site_id"], kind)) or CAUSES.get((site["category"], kind))
    if not options:
        # An empty cause makes build_dataset.py print a warning naming the
        # anomaly, which is more useful than a silent placeholder.
        return ("", "maintenance_log")

    fresh = [option for option in options if option[0] not in used]
    pool = fresh or options
    choice = pool[int(rng.integers(0, len(pool)))]
    used.add(choice[0])
    return choice


def inject(
    series_by_site: dict[str, pd.Series],
    sites: list[dict],
    rng: np.random.Generator,
) -> tuple[dict[str, pd.Series], list[dict]]:
    """Return (modified series, anomaly records)."""
    modified: dict[str, pd.Series] = {}
    records: list[dict] = []
    used_causes: set[str] = set()
    counter = 0

    for site in sites:
        site_id = site["site_id"]
        series = series_by_site[site_id].copy()
        index = series.index
        holiday_set = profiles.holiday_dates(site, index)
        taken: list[tuple] = []

        count = int(rng.integers(ANOMALIES_PER_SITE[0], ANOMALIES_PER_SITE[1] + 1))

        # Draw types WITHOUT replacement, so one site never gets two anomalies
        # of the same kind and the fleet-wide mix stays roughly even. Uniform
        # per-anomaly draws do not self-balance at this sample size: an earlier
        # run produced 8 spikes against 2 drop-offs, too few drop-offs to write
        # a meaningful eval question against.
        kinds = [str(k) for k in rng.permutation(["spike", "dropoff", "drift"])[:count]]

        for kind in kinds:
            detectability = str(rng.choice(["obvious", "subtle"]))

            low_hours, high_hours = DURATION_HOURS[kind]
            duration = pd.Timedelta(hours=int(rng.integers(low_hours, high_hours + 1)))

            window = _pick_window(index, duration, holiday_set, taken, rng)
            if window is None:
                continue
            start, end = window
            taken.append((start, end))

            low_dev, high_dev = DEVIATION[(kind, detectability)]
            magnitude = float(rng.uniform(low_dev, high_dev))

            mask = (index >= start) & (index <= end)
            sample_count = int(mask.sum())
            if sample_count < 4:
                continue

            if kind == "spike":
                factor = 1.0 + magnitude * _taper(sample_count)
                record_start = start
                peak_pct = magnitude * 100
            elif kind == "dropoff":
                factor = 1.0 - magnitude * _taper(sample_count)
                record_start = start
                peak_pct = -magnitude * 100
            else:
                factor = 1.0 + magnitude * np.linspace(0.0, 1.0, sample_count)
                # A drift has no single true onset. Ground truth marks where the
                # creep becomes SUSTAINED, not the first sample that ticks up.
                # This is why eval/questions.yaml gives drift a 5-day timing
                # tolerance while spikes get 60 minutes.
                record_start = start + duration * 0.2
                peak_pct = magnitude * 100

            series.loc[mask] = series.loc[mask].to_numpy() * factor

            counter += 1
            cause, doc_type = _pick_cause(site, kind, rng, used_causes)
            records.append(
                {
                    "anomaly_id": f"anom-{counter:03d}",
                    "site_id": site_id,
                    "type": kind,
                    "start_utc": record_start.isoformat(),
                    "end_utc": end.isoformat(),
                    "peak_deviation_pct": round(peak_pct, 1),
                    "detectability": detectability,
                    "cause": cause,
                    "cause_doc_type": doc_type,
                }
            )

        modified[site_id] = series

    return modified, records
