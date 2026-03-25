"""Scenario configuration and scripted vessel specs."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from custody.simulation.profiles import (
    BehaviorProfile, NORMAL_TRANSIT, SLOW_TRANSIT,
    LOITERING, ZONE_APPROACH, EVASIVE, PATROL,
    DEMO_TRANSIT, DEMO_APPROACH, DEMO_LOITER, DEMO_EGRESS,
)


@dataclass(frozen=True)
class PhaseTrigger:
    """Optional condition that can activate a phase independent of time.

    When attached to a ProfilePhase, the phase activates when BOTH its
    ``start_hour`` has been reached AND the trigger condition is satisfied.
    Use ``start_hour=0`` for a trigger-only phase (eligible from the start).

    Supported condition types:
        "anomaly_above"      — entity anomaly_score > threshold
        "anomaly_below"      — entity anomaly_score < threshold
        "custody_below"      — custody_confidence < threshold
        "custody_above"      — custody_confidence > threshold
        "missed_collections" — consecutive timesteps with no TASK action >= threshold
        "zone_entry"         — sensitive_zone detector score > 0 (threshold ignored)
        "zone_exit"          — sensitive_zone detector score == 0 (threshold ignored)

    Once a trigger fires for a vessel, it latches: the phase stays active
    until a later phase (by start_hour or its own trigger) supersedes it.
    """
    condition: str
    threshold: float = 0.0


@dataclass(frozen=True)
class ProfilePhase:
    """One behavior phase in a vessel's scripted timeline.

    Phase selection priority:
      1. Time-gated: ``start_hour <= current_hour``.
      2. Trigger-gated: if ``trigger`` is set, the phase only activates when
         the trigger condition is also satisfied (checked after start_hour).
      3. Later phases override earlier ones (last eligible phase wins).
    """
    start_hour: int
    profile: BehaviorProfile
    # If both set, vessel steers toward this point each step
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    # If set, override heading to this fixed value
    heading_override: Optional[float] = None
    # Optional condition for activation (evaluated after start_hour gate)
    trigger: Optional[PhaseTrigger] = None


@dataclass
class VesselSpec:
    vessel_id: str
    start_lat: float
    start_lon: float
    start_heading_deg: float
    phases: list[ProfilePhase]   # must be sorted by start_hour, first phase start_hour=0
    is_anomalous: bool = False
    tags: list = field(default_factory=list)
    # Operator tracking directive: NONE | MAINTAIN_CUSTODY
    # MAINTAIN_CUSTODY forces the vessel into ACTIVE_CUSTODY tier regardless of behavior.
    tracking_directive: str = "NONE"
    # Hour index at which the vessel's AIS transponder goes silent.
    # None means AIS is active for the full scenario duration.
    ais_dropout_hour: Optional[int] = None


@dataclass
class ScenarioConfig:
    name: str
    seed: int
    n_background: int
    start_time: datetime
    duration_hours: int
    dt_hours: int
    # (lat_min, lat_max, lon_min, lon_max) — bounding box for background placement
    region: tuple[float, float, float, float]
    scripted_vessels: list[VesselSpec]
    # Optional: weighted dict of archetype names → weight (None = legacy 70/30 split)
    background_archetype_weights: dict | None = None
    # Optional: center point for approach-archetype vessels (None = use region center)
    zone_center: tuple | None = None


# ── Two-vessel smoke / regression scenario ────────────────────────────────────
#
# Compact internal fixture: V001 + V002 over 9 hours, zero-variance profiles.
# NOT a product-facing scenario. Used only by run_simulation() for smoke tests,
# regression suites, and fast unit-test baselines.
# For the real product demo, use PORTFOLIO_SCENARIO.

TWO_VESSEL_SMOKE = ScenarioConfig(
    name="two_vessel_smoke",
    seed=42,
    n_background=0,
    start_time=datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc),
    duration_hours=9,
    dt_hours=1,
    region=(-3.0, 5.0, -3.0, 5.0),
    scripted_vessels=[
        VesselSpec(
            vessel_id="V001",
            start_lat=0.0, start_lon=0.0,
            start_heading_deg=45.0,
            phases=[
                ProfilePhase(0, DEMO_TRANSIT,  heading_override=45.0),
                ProfilePhase(4, DEMO_APPROACH, heading_override=20.0),
                ProfilePhase(6, DEMO_LOITER),
                ProfilePhase(9, DEMO_EGRESS,   heading_override=120.0),
            ],
        ),
        VesselSpec(
            vessel_id="V002",
            start_lat=0.2, start_lon=0.1,
            start_heading_deg=45.0,
            phases=[
                ProfilePhase(0, DEMO_TRANSIT,  heading_override=45.0),
                ProfilePhase(5, DEMO_APPROACH, heading_override=20.0),
                ProfilePhase(7, DEMO_EGRESS,   heading_override=120.0),
            ],
        ),
    ],
)


# ── Default 24-hour, 30-vessel scenario ───────────────────────────────────────
#
# 3 scripted anomalous actors + 27 background vessels.
# Scenario date: 2026-03-23 (matches calibrated TLE passes in sensors.py).
# Key action window: 10:00-18:00z (orbital passes available).
#
# SIGMA-1: SW approach → loiters in ZONE_ALPHA 14:00-20:00z
# SIGMA-2: N transit + route deviation → approaches zone 18:00-24:00z
# SIGMA-3: E transit + evasive → reaches zone boundary 20:00-24:00z

DEFAULT_SCENARIO = ScenarioConfig(
    name="multi_target_24h",
    seed=2026,
    n_background=27,
    start_time=datetime(2026, 3, 23, 0, 0, tzinfo=timezone.utc),
    duration_hours=24,
    dt_hours=1,
    region=(-3.0, 5.0, -3.0, 5.0),
    scripted_vessels=[
        # SIGMA-1: Approaches zone from SW, loiters inside
        VesselSpec(
            vessel_id="SIGMA-1",
            start_lat=-0.5, start_lon=-0.5,
            start_heading_deg=45.0,
            phases=[
                ProfilePhase(0,  NORMAL_TRANSIT),
                ProfilePhase(8,  ZONE_APPROACH, target_lat=1.2, target_lon=0.75),
                ProfilePhase(14, LOITERING),
                ProfilePhase(20, EVASIVE, heading_override=120.0),
            ],
            is_anomalous=True,
        ),
        # SIGMA-2: Route deviation from N, reaches zone late
        VesselSpec(
            vessel_id="SIGMA-2",
            start_lat=3.0, start_lon=2.0,
            start_heading_deg=200.0,
            phases=[
                ProfilePhase(0,  NORMAL_TRANSIT),
                ProfilePhase(8,  NORMAL_TRANSIT, heading_override=230.0),
                ProfilePhase(14, ZONE_APPROACH, target_lat=1.3, target_lon=0.9),
                ProfilePhase(19, LOITERING),
            ],
            is_anomalous=True,
        ),
        # SIGMA-3: Evasive from E, approaches zone boundary
        VesselSpec(
            vessel_id="SIGMA-3",
            start_lat=1.5, start_lon=3.0,
            start_heading_deg=270.0,
            phases=[
                ProfilePhase(0,  SLOW_TRANSIT),
                ProfilePhase(6,  EVASIVE),
                ProfilePhase(14, ZONE_APPROACH, target_lat=1.1, target_lon=0.6),
                ProfilePhase(20, LOITERING),
            ],
            is_anomalous=True,
        ),
    ],
)


# ── Portfolio 36-hour, 24-entity scenario ─────────────────────────────────────
#
# 4 scripted actors + 20 background vessels.
# Background uses 4-archetype weighted distribution (transit/slow_transit/patrol/approach).
# Scenario date: 2026-03-23 (36h window).
#
# BRAVO-1 : Zone Loiterer  — SW approach reaches zone ~h15, loiters h16-28, evasive egress
# ECHO-1/2: Rendezvous pair — converge from E/W on shared waypoint; rendezvous fires ~h12
# PORT-1  : Manual custody — nominal slow transit under MAINTAIN_CUSTODY, AIS goes dark h12
#
# Narrative timeline:
#   h0-11  : normal portfolio — all actors in transit, PORT-1 under MAINTAIN_CUSTODY
#   h12    : PORT-1 AIS dropout + ECHO rendezvous fires simultaneously → portfolio competition
#   h12-16 : dark-vessel + rendezvous concerns overlap; BRAVO-1 still approaching zone
#   h16-22 : BRAVO-1 loiters inside ZONE_ALPHA; all three concerns compete for sensor tasking
#   h22    : ECHO pair separates on diverging headings
#   h22-28 : PORT-1 track degrading; BRAVO-1 loitering continues; ECHO pair clear
#   h28-36 : BRAVO-1 evasive egress; PORT-1 uncertainty large; ECHO pair gone

PORTFOLIO_SCENARIO = ScenarioConfig(
    name="portfolio_36h",
    seed=3141,
    n_background=20,
    start_time=datetime(2026, 3, 23, 0, 0, tzinfo=timezone.utc),
    duration_hours=36,
    dt_hours=1,
    region=(-2.0, 4.0, -2.0, 4.0),
    zone_center=(1.3, 0.75),
    background_archetype_weights={
        "transit":      0.40,
        "slow_transit": 0.25,
        "patrol":       0.25,
        "approach":     0.10,
    },
    scripted_vessels=[
        # BRAVO-1: Zone Loiterer — approaches ZONE_ALPHA from SW; start is ~147 km from
        # the zone centre so it arrives around h15 and loiters inside for 12h.
        VesselSpec(
            vessel_id="BRAVO-1",
            start_lat=0.5, start_lon=-0.3,
            start_heading_deg=50.0,
            phases=[
                ProfilePhase(0,  ZONE_APPROACH, target_lat=1.3, target_lon=0.75),
                ProfilePhase(16, LOITERING),
                ProfilePhase(28, EVASIVE, heading_override=240.0),
            ],
            is_anomalous=True,
            tags=["zone_loiterer", "scripted"],
        ),
        # PORT-1: Port-departure vessel — nominal slow transit under a MAINTAIN_CUSTODY
        # directive.  AIS signal lost at h12; position freezes at last-known, uncertainty
        # grows, and dark-vessel floor raises it to ACTIVE_CUSTODY.
        VesselSpec(
            vessel_id="PORT-1",
            start_lat=3.5, start_lon=-1.5,
            start_heading_deg=90.0,
            phases=[
                ProfilePhase(0, SLOW_TRANSIT, heading_override=90.0),
            ],
            is_anomalous=False,
            tracking_directive="MAINTAIN_CUSTODY",
            ais_dropout_hour=12,
            tags=["port_departure", "manual_custody", "dark_vessel"],
        ),
        # ECHO-1 / ECHO-2: Rendezvous pair — converge on a shared waypoint north of
        # ZONE_ALPHA from opposite sides, dwell for ~10h, then separate on diverging
        # headings.  Rendezvous detection fires around h12, coinciding with PORT-1 dropout.
        #
        # Physics: ZONE_APPROACH (10 km/h) for 4h narrows the 110 km gap to ~31 km;
        # LOITERING (2 km/h) toward the shared anchor closes the remaining distance at
        # ~4 km/h combined, bringing both inside the 5 km threshold around h11-12.
        VesselSpec(
            vessel_id="ECHO-1",
            start_lat=2.5, start_lon=0.5,
            start_heading_deg=90.0,
            phases=[
                ProfilePhase(0,  ZONE_APPROACH, target_lat=2.5, target_lon=1.0),
                ProfilePhase(4,  LOITERING, target_lat=2.5, target_lon=1.0),
                ProfilePhase(22, EVASIVE, heading_override=270.0),
            ],
            is_anomalous=True,
            tags=["rendezvous_pair", "scripted"],
        ),
        VesselSpec(
            vessel_id="ECHO-2",
            start_lat=2.5, start_lon=1.5,
            start_heading_deg=270.0,
            phases=[
                ProfilePhase(0,  ZONE_APPROACH, target_lat=2.5, target_lon=1.0),
                ProfilePhase(4,  LOITERING, target_lat=2.5, target_lon=1.0),
                ProfilePhase(22, EVASIVE, heading_override=90.0),
            ],
            is_anomalous=True,
            tags=["rendezvous_pair", "scripted"],
        ),
    ],
)


# ── Multi-day 72-hour, 30-entity scenario ─────────────────────────────────────
#
# 6 scripted actors + 24 background vessels.
# Scenario date: 2026-03-23 00:00z → 2026-03-26 00:00z (72h window).
#
# Three-act narrative arc:
#
# ═══════════════════════════════════════════════════════════════════════════════
#  ACT 1 — BASELINE (h0–h24)
# ═══════════════════════════════════════════════════════════════════════════════
#  Normal portfolio operations.  Background traffic transits the region.
#  Scripted actors establish their presence and begin initial approaches.
#
#  FOXTROT-1 : Slow approach from SW toward ZONE_ALPHA; arrives ~h20.
#  GOLF-1    : Normal transit from N; begins route deviation at h16.
#  HOTEL-1/2 : Rendezvous pair — converge from E/W; dwell begins ~h14.
#  INDIA-1   : Port departure under MAINTAIN_CUSTODY; AIS goes dark at h20.
#  JULIET-1  : Fast transit from SE; no anomaly in Act 1 — surveillance baseline.
#
# ═══════════════════════════════════════════════════════════════════════════════
#  ACT 2 — SUSPICIOUS BUILDUP / COMPETING PRIORITIES (h24–h48)
# ═══════════════════════════════════════════════════════════════════════════════
#  Multiple concerns compete for limited sensor capacity.
#
#  FOXTROT-1 : Loitering inside ZONE_ALPHA (h20–h44).  Primary collection target.
#  GOLF-1    : Zone approach begins h24; enters zone ~h36.  Second priority.
#  HOTEL-1/2 : Rendezvous confirmed; pair separates at h36 on evasive headings.
#  INDIA-1   : Dark since h20; uncertainty growing.  Dark-vessel floor holds
#              ACTIVE_CUSTODY.  Competes for sensors it cannot confirm.
#  JULIET-1  : Begins zone approach at h28 from distance; third priority.
#
# ═══════════════════════════════════════════════════════════════════════════════
#  ACT 3 — ESCALATION, MISSED COLLECTIONS, RETASKING (h48–h72)
# ═══════════════════════════════════════════════════════════════════════════════
#  Consequences of constrained resources accumulate.  Some tracks degrade;
#  resolved targets clear the board; new concern surfaces.
#
#  FOXTROT-1 : Evasive egress from zone starting h44.  Anomaly drops slowly.
#  GOLF-1    : Loitering in zone h36–h60.  Becomes top priority after FOXTROT
#              clears.  Evasive egress h60.
#  HOTEL-1/2 : Gone — evasive since h36, leaving the region.
#  INDIA-1   : Still dark.  Uncertainty very large.  Custody LOST by ~h60.
#  JULIET-1  : Approaches zone h28–h48; loiters inside h48–h64.  Late-arriving
#              concern overlaps GOLF-1 zone loitering — the peak of portfolio
#              contention.  Patrol exit h64.
#

MULTI_DAY_SCENARIO = ScenarioConfig(
    name="multi_day_72h",
    seed=7272,
    n_background=24,
    start_time=datetime(2026, 3, 23, 0, 0, tzinfo=timezone.utc),
    duration_hours=72,
    dt_hours=1,
    region=(-2.0, 4.0, -2.0, 4.0),
    zone_center=(1.3, 0.75),
    background_archetype_weights={
        "transit":      0.35,
        "slow_transit": 0.25,
        "patrol":       0.25,
        "approach":     0.15,
    },
    scripted_vessels=[
        # ── FOXTROT-1: Zone loiterer ─────────────────────────────────────
        # Act 1: slow approach from SW.  Act 2: loiters in ZONE_ALPHA for
        # 24 hours.  Act 3: evasive egress.
        VesselSpec(
            vessel_id="FOXTROT-1",
            start_lat=0.3, start_lon=-0.5,
            start_heading_deg=50.0,
            phases=[
                ProfilePhase(0,  ZONE_APPROACH, target_lat=1.3, target_lon=0.75),
                ProfilePhase(20, LOITERING),
                ProfilePhase(44, EVASIVE, heading_override=220.0),
            ],
            is_anomalous=True,
            tags=["zone_loiterer", "scripted", "act1_approach"],
        ),

        # ── GOLF-1: Delayed zone entrant ─────────────────────────────────
        # Act 1: normal transit from N, route deviation at h16.  Act 2:
        # zone approach → enters zone, loiters.  The loitering phase uses a
        # zone_entry trigger: it activates when the vessel's sensitive_zone
        # score > 0, rather than at a fixed hour.  Act 3: evasive egress h60.
        VesselSpec(
            vessel_id="GOLF-1",
            start_lat=3.5, start_lon=1.5,
            start_heading_deg=200.0,
            phases=[
                ProfilePhase(0,  NORMAL_TRANSIT),
                ProfilePhase(16, NORMAL_TRANSIT, heading_override=240.0),
                ProfilePhase(24, ZONE_APPROACH, target_lat=1.2, target_lon=0.8),
                ProfilePhase(24, LOITERING,
                             trigger=PhaseTrigger("zone_entry")),
                ProfilePhase(60, EVASIVE, heading_override=30.0),
            ],
            is_anomalous=True,
            tags=["delayed_entrant", "scripted", "act2_zone", "trigger_zone_entry"],
        ),

        # ── HOTEL-1 / HOTEL-2: Rendezvous pair ──────────────────────────
        # Act 1: converge from E/W to shared waypoint (2.3, 0.8); dwell
        # begins ~h14.  Act 2: sustained rendezvous through h36, then
        # separate on evasive headings.  Act 3: left the region.
        VesselSpec(
            vessel_id="HOTEL-1",
            start_lat=2.3, start_lon=0.0,
            start_heading_deg=90.0,
            phases=[
                ProfilePhase(0,  ZONE_APPROACH, target_lat=2.3, target_lon=0.8),
                ProfilePhase(6,  LOITERING, target_lat=2.3, target_lon=0.8),
                ProfilePhase(36, EVASIVE, heading_override=270.0),
            ],
            is_anomalous=True,
            tags=["rendezvous_pair", "scripted"],
        ),
        VesselSpec(
            vessel_id="HOTEL-2",
            start_lat=2.3, start_lon=1.6,
            start_heading_deg=270.0,
            phases=[
                ProfilePhase(0,  ZONE_APPROACH, target_lat=2.3, target_lon=0.8),
                ProfilePhase(6,  LOITERING, target_lat=2.3, target_lon=0.8),
                ProfilePhase(36, EVASIVE, heading_override=90.0),
            ],
            is_anomalous=True,
            tags=["rendezvous_pair", "scripted"],
        ),

        # ── INDIA-1: Dark vessel ─────────────────────────────────────────
        # Slow transit under MAINTAIN_CUSTODY.  AIS goes dark at h20 (Act 1
        # boundary).  Uncertainty grows through Acts 2 and 3.  Custody
        # degrades to LOST by ~h60.
        VesselSpec(
            vessel_id="INDIA-1",
            start_lat=3.0, start_lon=-1.0,
            start_heading_deg=100.0,
            phases=[
                ProfilePhase(0, SLOW_TRANSIT, heading_override=100.0),
            ],
            is_anomalous=False,
            tracking_directive="MAINTAIN_CUSTODY",
            ais_dropout_hour=20,
            tags=["dark_vessel", "manual_custody", "scripted"],
        ),

        # ── JULIET-1: Late-arriving zone concern ────────────────────────
        # Act 1: fast transit from SE, unremarkable.  Act 2: begins zone
        # approach h28, arrives ~h48.  Act 3: loiters inside zone h48–h64,
        # overlapping GOLF-1 for peak contention.  Patrol exit uses a
        # zone_exit trigger (starts when vessel leaves zone) rather than
        # a fixed hour.  Time floor h56 prevents premature activation.
        VesselSpec(
            vessel_id="JULIET-1",
            start_lat=-0.5, start_lon=3.0,
            start_heading_deg=315.0,
            phases=[
                ProfilePhase(0,  NORMAL_TRANSIT),
                ProfilePhase(28, ZONE_APPROACH, target_lat=1.1, target_lon=0.7),
                ProfilePhase(48, LOITERING),
                ProfilePhase(56, PATROL, heading_override=135.0,
                             trigger=PhaseTrigger("zone_exit")),
            ],
            is_anomalous=True,
            tags=["late_entrant", "scripted", "act3_contention", "trigger_zone_exit"],
        ),
    ],
)


# ── Rendezvous smoke / test scenario ──────────────────────────────────────────
#
# Minimal 2-vessel scenario for rendezvous detection tests.
# ECHO-1 and ECHO-2 start within the proximity threshold and loiter at the same
# anchor point — detection fires from the second timestep onward.
# Not a product-facing scenario; used only by tests and as a regression fixture.

RENDEZVOUS_SMOKE = ScenarioConfig(
    name="rendezvous_smoke",
    seed=999,
    n_background=0,
    start_time=datetime(2026, 3, 23, 0, 0, tzinfo=timezone.utc),
    duration_hours=8,
    dt_hours=1,
    region=(-2.0, 4.0, -2.0, 4.0),
    scripted_vessels=[
        VesselSpec(
            vessel_id="ECHO-1",
            start_lat=2.5, start_lon=0.99,
            start_heading_deg=90.0,
            phases=[ProfilePhase(0, LOITERING, target_lat=2.5, target_lon=1.0)],
            is_anomalous=True,
            tags=["rendezvous_pair"],
        ),
        VesselSpec(
            vessel_id="ECHO-2",
            start_lat=2.5, start_lon=1.01,
            start_heading_deg=270.0,
            phases=[ProfilePhase(0, LOITERING, target_lat=2.5, target_lon=1.0)],
            is_anomalous=True,
            tags=["rendezvous_pair"],
        ),
    ],
)


# ── Dark vessel smoke / test scenario ─────────────────────────────────────────
#
# Minimal scenario for dark-vessel detection tests.
# DARK-1: under MAINTAIN_CUSTODY directive; AIS drops at hour 3.
# Two background vessels provide a non-dark population to compare against.
#
# Expected behaviour:
#   Hours 0-2: DARK-1 emits current position, dark_vessel_flag=False
#   Hours 3-8: DARK-1 dark_vessel_flag=True, lat/lon frozen at last-known,
#              anomaly_score frozen, attention_state=ACTIVE_CUSTODY via dark floor.

DARK_VESSEL_SMOKE = ScenarioConfig(
    name="dark_vessel_smoke",
    seed=777,
    n_background=2,
    start_time=datetime(2026, 3, 23, 0, 0, tzinfo=timezone.utc),
    duration_hours=8,
    dt_hours=1,
    region=(-2.0, 4.0, -2.0, 4.0),
    scripted_vessels=[
        VesselSpec(
            vessel_id="DARK-1",
            start_lat=1.0, start_lon=0.0,
            start_heading_deg=90.0,
            phases=[ProfilePhase(0, SLOW_TRANSIT, heading_override=90.0)],
            is_anomalous=False,
            tracking_directive="MAINTAIN_CUSTODY",
            ais_dropout_hour=3,
            tags=["dark_vessel", "manual_custody"],
        ),
    ],
)
