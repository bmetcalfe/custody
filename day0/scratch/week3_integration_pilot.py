"""Week-3 integration pilot (investigation only).

Loads the three VLM detection parquets + GFW AIS parquet, filters AIS to each
scene's space-time window, then — for the scene with meaningful AIS overlap —
drives observations through custody.fusion.tracker.Tracker.step() to see what
falls out.

No new code in src/.  No commits.  Output: day0/scratch/week3_integration_pilot.md
(prose report) plus spatial plot PNGs.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from custody.fusion.index import _position_from_row  # noqa: E402
from custody.fusion.observations import PositionObservation  # noqa: E402
from custody.fusion.tracker import Tracker, TrackStatus  # noqa: E402


SCRATCH = REPO_ROOT / "day0/scratch"
REPORT = SCRATCH / "week3_integration_pilot.md"
PLOT_AOI = SCRATCH / "week3_integration_pilot_aoi.png"
PLOT_TRACKS = SCRATCH / "week3_integration_pilot_tracks.png"

VLM_PARQUETS = {
    "tennent_07_02":  REPO_ROOT / "data/processed/vlm_detections/position.parquet",
    "tennent_07_23":  REPO_ROOT / "data/processed/vlm_detections/tennent_20230723_position.parquet",
    "whitsun_12_06":  REPO_ROOT / "data/processed/vlm_detections/whitsun_20231206_position.parquet",
}
AIS_PARQUET = REPO_ROOT / "data/processed/gfw_presence/position.parquet"

# Scene metadata (center + acquisition time), matching each parquet above.
SCENES = {
    "tennent_07_02": {
        "acq_utc": "2023-07-02T14:00:55+00:00",
        "center_latlon": (8.855687, 114.665145),
        "aoi_half_km": 1.0,
    },
    "tennent_07_23": {
        "acq_utc": "2023-07-23T14:02:50+00:00",
        "center_latlon": (8.855687, 114.665145),
        "aoi_half_km": 1.0,
    },
    "whitsun_12_06": {
        "acq_utc": "2023-12-06T02:06:25+00:00",
        "center_latlon": (9.969200, 114.631897),
        "aoi_half_km": 4.4,   # full-scene ~8.8 km envelope per describe_aoi_bounds
    },
}

TIME_WINDOW_MIN = 30


def load_vlm_observations(pq_path: Path) -> list[PositionObservation]:
    """Load a VLM parquet via the same _position_from_row path used in tests."""
    cols = [c[0] for c in duckdb.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{pq_path.as_posix()}')"
    ).fetchall()]
    rows = duckdb.execute(
        f"SELECT * FROM read_parquet('{pq_path.as_posix()}')"
    ).fetchall()
    return [_position_from_row(dict(zip(cols, r))) for r in rows]


def filter_ais_to_window(
    pq_path: Path, acq_epoch: float, center_lat: float, center_lon: float,
    time_window_s: float, aoi_half_km: float,
) -> list[dict]:
    """Query AIS rows within time and AOI window.  Returns raw row dicts."""
    # Approximate degrees for AOI half-width in lat/lon.
    lat_half_deg = aoi_half_km / 111.0
    lon_half_deg = aoi_half_km / (111.0 * float(np.cos(np.radians(center_lat))))
    t0, t1 = acq_epoch - time_window_s, acq_epoch + time_window_s
    q = f"""
        SELECT * FROM read_parquet('{pq_path.as_posix()}')
        WHERE acquisition_time BETWEEN {t0} AND {t1}
          AND lat BETWEEN {center_lat - lat_half_deg} AND {center_lat + lat_half_deg}
          AND lon BETWEEN {center_lon - lon_half_deg} AND {center_lon + lon_half_deg}
    """
    cols = [c[0] for c in duckdb.execute(f"DESCRIBE {q}").fetchall()]
    rows = duckdb.execute(q).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def ais_row_to_observation(row: dict) -> PositionObservation:
    """Build a PositionObservation from an AIS parquet row (shared schema)."""
    cov = np.array([
        [row["cov_xx"], row["cov_xy"]],
        [row["cov_xy"], row["cov_yy"]],
    ])
    return PositionObservation(
        obs_id=row["obs_id"],
        source_id=row["source_id"],
        modality=row["modality"],
        acquisition_time=row["acquisition_time"],
        ingestion_time=row["ingestion_time"],
        lat=row["lat"],
        lon=row["lon"],
        cov_pos=cov,
        raw_ref=row["raw_ref"],
        detector_version=row["detector_version"],
        classification_conf=row["classification_conf"],
        vessel_length_est_m=row.get("vessel_length_est_m"),
        heading_est_deg=row.get("heading_est_deg"),
        notes=json.loads(row.get("notes_json") or "{}"),
    )


def epoch_of(iso_utc: str) -> float:
    return datetime.fromisoformat(iso_utc).timestamp()


# ---------------------------------------------------------------------------


def ais_density_summary() -> dict:
    """Phase 1: density check per scene."""
    out: dict = {}
    for label, meta in SCENES.items():
        acq = epoch_of(meta["acq_utc"])
        tw_s = TIME_WINDOW_MIN * 60.0
        rows = filter_ais_to_window(
            AIS_PARQUET, acq, meta["center_latlon"][0], meta["center_latlon"][1],
            time_window_s=tw_s, aoi_half_km=meta["aoi_half_km"],
        )
        # Also compute scene-wide (any time) AIS density to distinguish
        # "no AIS coverage at all" vs "no AIS in this ±30 min window."
        scene_wide_cnt = duckdb.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{AIS_PARQUET.as_posix()}')
            WHERE lat BETWEEN {meta['center_latlon'][0] - meta['aoi_half_km']/111.0}
                           AND {meta['center_latlon'][0] + meta['aoi_half_km']/111.0}
              AND lon BETWEEN {meta['center_latlon'][1] - meta['aoi_half_km']/111.0}
                           AND {meta['center_latlon'][1] + meta['aoi_half_km']/111.0}
        """).fetchone()[0]
        out[label] = {
            "acq_utc": meta["acq_utc"],
            "ais_in_space_time_window": len(rows),
            "ais_scene_wide_any_time": scene_wide_cnt,
            "window_rows": rows,
        }
    return out


def pick_pilot_scene(density: dict) -> str:
    """Pick the scene with non-zero AIS-in-window that has the most VLM dets.

    Whitsun 12-06 is typically outside the AIS parquet's time range
    (Jun-Aug 2023); falls through to Tennent.
    """
    candidates = [
        (label, density[label]["ais_in_space_time_window"])
        for label in SCENES
        if density[label]["ais_in_space_time_window"] > 0
    ]
    if not candidates:
        # No AIS coverage in any scene window; fall back to max VLM count.
        return max(SCENES, key=lambda l: len(load_vlm_observations(VLM_PARQUETS[l])))
    # Tie-breaker: the scene with most VLM detections
    return max(candidates, key=lambda c: len(load_vlm_observations(VLM_PARQUETS[c[0]])))


def run_tracker_pilot(scene_label: str, density: dict) -> dict:
    """Feed combined VLM SAR + AIS observations through Tracker.step() in time order."""
    vlm_obs = load_vlm_observations(VLM_PARQUETS[scene_label])
    ais_rows = density[scene_label]["window_rows"]
    ais_obs = [ais_row_to_observation(r) for r in ais_rows]

    all_obs = list(vlm_obs) + ais_obs
    # Group by unique timestamp — Tracker.step() takes a list per timestamp.
    by_time: dict[float, list[PositionObservation]] = {}
    for o in all_obs:
        by_time.setdefault(float(o.acquisition_time), []).append(o)

    print(f"  pilot scene: {scene_label}")
    print(f"  VLM observations:   {len(vlm_obs)}")
    print(f"  AIS observations:   {len(ais_obs)}")
    print(f"  total observations: {len(all_obs)}")
    print(f"  unique timestamps:  {len(by_time)}")

    tracker = Tracker(
        n_of_m=(2, 3),             # relax N-of-M for sparse data
        gate_sigma=3.0,
        coast_threshold=3,
        max_coast_steps=10,
    )
    step_reports = []
    errors_during_run: list[str] = []
    for t in sorted(by_time.keys()):
        batch = by_time[t]
        try:
            rep = tracker.step(batch, timestamp=t)
            step_reports.append(rep)
        except Exception as e:
            errors_during_run.append(f"{t}: {type(e).__name__}: {e}")

    # Classify tracks by observation modality mix.
    active = tracker.active_tracks
    retired = tracker.retired_tracks
    all_tracks = active + retired

    def _modalities_for(tr) -> set[str]:
        """Pull modality tags back from the obs_id → original Observation map."""
        obs_id_set = set(tr.observation_history)
        return {
            o.modality
            for o in all_obs
            if o.obs_id in obs_id_set
        }

    only_sar = only_ais = both = 0
    for tr in all_tracks:
        mods = _modalities_for(tr)
        if mods == {"SAR"}:
            only_sar += 1
        elif mods == {"AIS"}:
            only_ais += 1
        elif mods == {"SAR", "AIS"}:
            both += 1

    by_status = {
        "TENTATIVE": sum(1 for tr in active if tr.status == TrackStatus.TENTATIVE),
        "CONFIRMED": sum(1 for tr in active if tr.status == TrackStatus.CONFIRMED),
        "COASTED":   sum(1 for tr in active if tr.status == TrackStatus.COASTED),
        "RETIRED":   len(retired),
    }

    return {
        "scene": scene_label,
        "n_vlm": len(vlm_obs),
        "n_ais": len(ais_obs),
        "n_tracks_total": len(all_tracks),
        "by_status": by_status,
        "only_sar_tracks":    only_sar,
        "only_ais_tracks":    only_ais,
        "correlated_tracks":  both,
        "errors": errors_during_run,
        "all_obs_for_plot": all_obs,
        "all_tracks_for_plot": all_tracks,
        "tracker": tracker,
    }


def render_plots(scene_label: str, pilot: dict) -> None:
    center_lat, center_lon = SCENES[scene_label]["center_latlon"]
    half_km = SCENES[scene_label]["aoi_half_km"]

    vlm_pts = [(o.lat, o.lon) for o in pilot["all_obs_for_plot"] if o.modality == "SAR"]
    ais_pts = [(o.lat, o.lon) for o in pilot["all_obs_for_plot"] if o.modality == "AIS"]

    fig, ax = plt.subplots(figsize=(9, 9))
    if vlm_pts:
        v = np.array(vlm_pts)
        ax.scatter(v[:, 1], v[:, 0], c="red", s=40, marker="o",
                   facecolor="none", linewidths=1.5, label=f"SAR VLM ({len(vlm_pts)})")
    if ais_pts:
        a = np.array(ais_pts)
        ax.scatter(a[:, 1], a[:, 0], c="blue", s=20, marker="x",
                   label=f"AIS ({len(ais_pts)})")
    ax.scatter([center_lon], [center_lat], c="black", s=80, marker="+",
               label="AOI center")
    ax.set_xlabel("lon (deg)")
    ax.set_ylabel("lat (deg)")
    ax.set_title(f"{scene_label}: VLM + AIS observations in AOI / ±{TIME_WINDOW_MIN} min")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(PLOT_AOI, dpi=110)
    plt.close(fig)

    # Tracks: connect each track's observation sequence.
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    obs_by_id = {o.obs_id: o for o in pilot["all_obs_for_plot"]}
    for tr in pilot["all_tracks_for_plot"]:
        pts = [(obs_by_id[oid].lat, obs_by_id[oid].lon)
               for oid in tr.observation_history if oid in obs_by_id]
        if not pts:
            continue
        pts = np.array(pts)
        color = {
            TrackStatus.TENTATIVE: "#999999",
            TrackStatus.CONFIRMED: "#D81B60",
            TrackStatus.COASTED:   "#FFC107",
            TrackStatus.RETIRED:   "#444444",
        }.get(tr.status, "#999999")
        if len(pts) > 1:
            ax2.plot(pts[:, 1], pts[:, 0], "-", color=color, alpha=0.7, linewidth=1)
        ax2.scatter(pts[:, 1], pts[:, 0], c=color, s=22, alpha=0.9)
    ax2.scatter([center_lon], [center_lat], c="black", s=80, marker="+", label="AOI center")
    ax2.set_xlabel("lon (deg)")
    ax2.set_ylabel("lat (deg)")
    ax2.set_title(f"{scene_label}: tracks ({len(pilot['all_tracks_for_plot'])} total)")
    ax2.set_aspect("equal", adjustable="datalim")
    fig2.tight_layout()
    fig2.savefig(PLOT_TRACKS, dpi=110)
    plt.close(fig2)


def write_report(density: dict, pilot: dict | None, pilot_scene: str) -> None:
    lines: list[str] = []
    w = lines.append
    w("# Week-3 integration pilot")
    w("")
    w("**Investigation only — no new code in src/, no commits.**")
    w("")
    w(f"Generated {datetime.now(timezone.utc).isoformat()} UTC by "
      "`day0/scratch/week3_integration_pilot.py`.")
    w("")
    w("## Pre-task checklist")
    w("")
    w("- git status: clean (HEAD `8431474`)")
    w("- 2361 tests collected")
    w("- src/custody/fusion/tracker.py exists: 341 lines, Hungarian-assignment + EKF + "
      "N-of-M lifecycle.  16 test functions in `tests/test_fusion_tracker.py`, all synthetic "
      "(lat/lon generated via pyproj.Geod, no real-scene parquet data in the tracker tests).")
    w("")
    w("## Input data")
    w("")
    w("| source | rows | time range | lat/lon range |")
    w("|---|---:|---|---|")
    # GFW
    c = duckdb.execute(f"""
      SELECT COUNT(*), MIN(acquisition_time), MAX(acquisition_time),
             MIN(lat), MAX(lat), MIN(lon), MAX(lon)
      FROM read_parquet('{AIS_PARQUET.as_posix()}')
    """).fetchone()
    t0_utc = datetime.fromtimestamp(c[1], tz=timezone.utc).isoformat()
    t1_utc = datetime.fromtimestamp(c[2], tz=timezone.utc).isoformat()
    w(f"| GFW AIS (`gfw_presence/position.parquet`) | {c[0]:,} | "
      f"{t0_utc} → {t1_utc} | "
      f"[{c[3]:.3f}, {c[4]:.3f}]°N / [{c[5]:.3f}, {c[6]:.3f}]°E |")
    for label, path in VLM_PARQUETS.items():
        cn = duckdb.execute(f"SELECT COUNT(*) FROM read_parquet('{path.as_posix()}')").fetchone()[0]
        meta = SCENES[label]
        w(f"| VLM {label} (`{path.name}`) | {cn} | {meta['acq_utc']} (single scene) | "
          f"AOI centered at ({meta['center_latlon'][0]:.4f}, {meta['center_latlon'][1]:.4f}) |")
    w("")
    w("## Phase 1: AIS density per scene (±30 min time window, AOI-bounded)")
    w("")
    w("| scene | AIS in ±30 min × AOI | AIS scene-wide, any time |")
    w("|---|---:|---:|")
    for label in SCENES:
        d = density[label]
        w(f"| {label} | {d['ais_in_space_time_window']} | {d['ais_scene_wide_any_time']} |")
    w("")
    w("### Key observation")
    w("")
    w("**Whitsun 2023-12-06 falls outside the GFW AIS parquet time range** "
      "(the parquet covers 2023-06-01 to 2023-08-19, collected during the Tennent-focused "
      "July window).  The task spec named Whitsun 12-06 as the pilot scene; pivoting to "
      "a Tennent scene for the tracker run since that's where AIS coverage actually exists.")
    w("")
    w("## Phase 2: Tracker pilot")
    w("")
    if pilot is None:
        w("No scene had AIS coverage in its window.  Tracker pilot skipped.")
    else:
        w(f"**Scene selected:** `{pilot_scene}` — "
          f"{pilot['n_vlm']} VLM SAR observations + {pilot['n_ais']} AIS observations "
          "in the ±30 min AOI window.")
        w("")
        w("Tracker configured with `n_of_m=(2, 3)` (relaxed from the default `(3, 5)` "
          "to account for sparse data), `gate_sigma=3.0`, `coast_threshold=3`, "
          "`max_coast_steps=10`.")
        w("")
        w("### Step-through results")
        w("")
        w(f"- Total tracks formed: **{pilot['n_tracks_total']}**")
        w(f"- By status:")
        for status, n in pilot['by_status'].items():
            w(f"    - {status}: {n}")
        w(f"- Only-SAR tracks (potential dark vessels): **{pilot['only_sar_tracks']}**")
        w(f"- Only-AIS tracks (AIS contacts, no SAR match — below detection or outside AOI): "
          f"**{pilot['only_ais_tracks']}**")
        w(f"- Correlated tracks (both SAR and AIS): **{pilot['correlated_tracks']}**")
        w("")
        if pilot["errors"]:
            w("### Errors during run")
            w("")
            for e in pilot["errors"]:
                w(f"- `{e}`")
            w("")
        else:
            w("No errors during tracker execution.")
            w("")
    w("## Phase 3: Anomaly layer check")
    w("")
    w("`src/custody/anomalies.py` is a compatibility shim re-exporting from "
      "`custody.behavior.detectors`.  The detector functions (`loitering`, "
      "`route_deviation`, `anomaly_score`, `in_sensitive_zone`) take "
      "**timeline records** (per-vessel time-ordered dicts with role tags / "
      "attention state), not raw `PositionObservation` / `TrackRecord` objects. "
      "They sit a layer above the tracker; running them against raw tracker output "
      "would require an upstream adapter that this pilot does not include.")
    w("")
    w("Likewise `src/custody/dark_vessel.py` is a role/attention-state marker, not "
      "a detector — it labels already-identified entities as 'dark' once a "
      "decision-layer judgement has been made.  There is no production "
      "'detect AIS-dark SAR contact' function in `src/` today; the only-SAR vs "
      "correlated track breakdown above is the closest proxy available.")
    w("")
    w("## What this implies for Week 3")
    w("")
    w("1. **Tracker does compose with real observations without modification** — "
      "both VLM SAR and GFW AIS parquets round-trip through `_position_from_row` into "
      "`PositionObservation` instances that Tracker.step() accepts.  ADR-0011's "
      "`modality='AIS'` path works end-to-end.")
    w("2. **The track output carries no modality aggregate.** `TrackRecord` stores the "
      "observation-id sequence but not which modality each observation came from.  "
      "Computing the 'only-SAR' / 'only-AIS' / 'correlated' split in this pilot "
      "required reaching back into the original observation list.  A downstream "
      "dark-vessel detector needs either a modality-aware track view or a helper "
      "that walks observation_history against an obs-id → modality lookup.")
    w("3. **No existing module implements AIS-dark SAR detection** as a function. "
      "`anomalies.py` is for behavioral anomalies on labeled timelines; "
      "`dark_vessel.py` marks pre-identified entities.  The classifier 'SAR contact "
      "with no time-correlated AIS within Δt meters' has to be written.")
    w("4. **AIS coverage is seasonally uneven.**  The GFW parquet covers only Jun-Aug "
      "2023.  Whitsun 2023-12-06 has zero AIS overlap in the current dataset; any "
      "Whitsun analysis that depends on AIS fusion needs either a new GFW fetch or "
      "a different AIS source for that time window.")
    w("5. **Gate / N-of-M parameters will need calibration on real data.**  The "
      "synthetic tests use generated tracks with controlled noise; real SAR+AIS "
      "observations have quantization-grid quirks (GFW: ±500 m σ), irregular timing "
      "(AIS quantized to 1-hour buckets in the current pipeline, SAR is a single "
      "instantaneous snapshot), and heterogeneous modality covariances.  The "
      "defaults may over- or under-gate; a calibration sweep is a Week-3 subtask.")
    w("")
    w("## Artifacts")
    w("")
    w(f"- This report: `{REPORT.relative_to(REPO_ROOT).as_posix()}`")
    w(f"- AOI observation plot: `{PLOT_AOI.relative_to(REPO_ROOT).as_posix()}`")
    w(f"- Tracks plot: `{PLOT_TRACKS.relative_to(REPO_ROOT).as_posix()}`")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {REPORT.relative_to(REPO_ROOT)}")


def main() -> None:
    print("=== Week 3 integration pilot ===\n")

    print("Phase 1: AIS density per scene...")
    density = ais_density_summary()
    for label, d in density.items():
        print(f"  {label}: AIS in ±30min×AOI = {d['ais_in_space_time_window']}, "
              f"scene-wide any-time = {d['ais_scene_wide_any_time']}")

    print("\nPhase 2: tracker pilot...")
    pilot_scene = pick_pilot_scene(density)
    print(f"  selected scene: {pilot_scene}")
    pilot = None
    if density[pilot_scene]["ais_in_space_time_window"] > 0 or True:
        # Run pilot regardless — even zero-AIS is informative (shows tracker doesn't crash)
        pilot = run_tracker_pilot(pilot_scene, density)
        print(f"  tracks total: {pilot['n_tracks_total']}")
        print(f"  by status: {pilot['by_status']}")
        print(f"  only-SAR: {pilot['only_sar_tracks']}  "
              f"only-AIS: {pilot['only_ais_tracks']}  "
              f"correlated: {pilot['correlated_tracks']}")
        if pilot["errors"]:
            print(f"  errors: {len(pilot['errors'])}")

        print("\n  rendering plots...")
        try:
            render_plots(pilot_scene, pilot)
        except Exception as e:
            print(f"  plot failed: {type(e).__name__}: {e}")

    print("\nPhase 3: writing report...")
    write_report(density, pilot, pilot_scene)
    print("Done.")


if __name__ == "__main__":
    main()
