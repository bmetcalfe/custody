"""Navigation callbacks: scenario load, slider sync, entity selection.

Entity selection uses ``selected-entity`` store as the single canonical
source of truth.  Three simple callbacks write to or read from it:

  1. Table active_cell → selected-entity (table row click)
  2. Entity dropdown   → selected-entity (dropdown pick)
  3. selected-entity   → entity dropdown value (sync display)
"""
from __future__ import annotations

from dash import Dash, Input, Output, State, no_update

import state as app_state
from adapter import entity_ids, timestep_count, records_at_timestep
from layout.sidebar import (
    SCENARIO_DROPDOWN, TIMELINE_SLIDER, ENTITY_DROPDOWN, TIMESTEP_DISPLAY,
    BTN_PREV_STEP, BTN_NEXT_STEP,
)
from layout.overview import PORTFOLIO_TABLE


def register(app: Dash) -> None:
    """Register navigation callbacks on *app*."""

    # ── Callback 1: scenario change ──────────────────────────────────────
    # Loads the scenario into server cache, resets all controls.

    @app.callback(
        Output(app_state.SCENARIO_KEY, "data"),
        Output(TIMELINE_SLIDER, "max"),
        Output(TIMELINE_SLIDER, "value"),
        Output(TIMELINE_SLIDER, "marks"),
        Output(ENTITY_DROPDOWN, "options"),
        Output(ENTITY_DROPDOWN, "value"),
        Output(app_state.TIMESTEP_INDEX, "data"),
        Output(app_state.SELECTED_ENTITY, "data"),
        Output(PORTFOLIO_TABLE, "active_cell"),
        Output(PORTFOLIO_TABLE, "selected_rows"),
        Input(SCENARIO_DROPDOWN, "value"),
        prevent_initial_call=False,
    )
    def scenario_changed(scenario_name):
        if not scenario_name:
            return (no_update,) * 10

        records = app_state.get_records(scenario_name)
        n_steps = timestep_count(records)
        max_idx = n_steps - 1

        # Slider marks at regular intervals for readability
        marks = {i: str(i) for i in range(0, n_steps, max(1, n_steps // 6))}
        if max_idx not in marks:
            marks[max_idx] = str(max_idx)

        ids = entity_ids(records)
        options = [{"label": eid, "value": eid} for eid in ids]

        return (
            scenario_name,   # scenario-key store
            max_idx,         # slider max
            0,               # slider value (reset)
            marks,           # slider marks
            options,         # entity dropdown options
            None,            # entity dropdown value (reset)
            0,               # timestep-index store (reset)
            None,            # selected-entity store (reset)
            None,            # table active_cell (clear)
            [],              # table selected_rows (clear)
        )

    # ── Callback 2: prev/next buttons → slider value ──────────────────────

    @app.callback(
        Output(TIMELINE_SLIDER, "value", allow_duplicate=True),
        Output(BTN_PREV_STEP, "disabled"),
        Output(BTN_NEXT_STEP, "disabled"),
        Input(BTN_PREV_STEP, "n_clicks"),
        Input(BTN_NEXT_STEP, "n_clicks"),
        Input(TIMELINE_SLIDER, "value"),
        State(TIMELINE_SLIDER, "max"),
        prevent_initial_call=True,
    )
    def step_buttons(prev_clicks, next_clicks, current, max_idx):
        from dash import ctx
        triggered = ctx.triggered_id
        if triggered == BTN_PREV_STEP:
            new_val = max(0, (current or 0) - 1)
        elif triggered == BTN_NEXT_STEP:
            new_val = min(max_idx or 0, (current or 0) + 1)
        else:
            # Slider was moved directly — just update button states
            new_val = no_update
        prev_disabled = (new_val if new_val is not no_update else current) == 0
        next_disabled = (new_val if new_val is not no_update else current) == (max_idx or 0)
        return new_val, prev_disabled, next_disabled

    # ── Callback 3: slider → timestep index store ────────────────────────

    @app.callback(
        Output(app_state.TIMESTEP_INDEX, "data", allow_duplicate=True),
        Input(TIMELINE_SLIDER, "value"),
        prevent_initial_call=True,
    )
    def slider_changed(value):
        if value is None:
            return no_update
        return value

    # ── Callback 4: table row click → selected-entity store ──────────────

    @app.callback(
        Output(app_state.SELECTED_ENTITY, "data", allow_duplicate=True),
        Input(PORTFOLIO_TABLE, "active_cell"),
        State(PORTFOLIO_TABLE, "data"),
        prevent_initial_call=True,
    )
    def table_select_entity(active_cell, table_data):
        if not active_cell or not table_data:
            return no_update
        row_idx = active_cell["row"]
        if row_idx < 0 or row_idx >= len(table_data):
            return no_update
        return table_data[row_idx].get("Entity")

    # ── Callback 5: entity dropdown → selected-entity store ──────────────

    @app.callback(
        Output(app_state.SELECTED_ENTITY, "data", allow_duplicate=True),
        Input(ENTITY_DROPDOWN, "value"),
        prevent_initial_call=True,
    )
    def dropdown_select_entity(dropdown_value):
        return dropdown_value

    # ── Callback 6: selected-entity store → sync dropdown display ────────

    @app.callback(
        Output(ENTITY_DROPDOWN, "value", allow_duplicate=True),
        Input(app_state.SELECTED_ENTITY, "data"),
        prevent_initial_call=True,
    )
    def sync_dropdown_to_store(entity_id):
        return entity_id

    # ── Callback 7: timestep display text ────────────────────────────────

    @app.callback(
        Output(TIMESTEP_DISPLAY, "children"),
        Input(app_state.SCENARIO_KEY, "data"),
        Input(app_state.TIMESTEP_INDEX, "data"),
    )
    def update_timestep_display(scenario_key, timestep_idx):
        if not scenario_key or timestep_idx is None:
            return "Step 0"
        records = app_state.get_records(scenario_key)
        ts_records = records_at_timestep(records, timestep_idx)
        if not ts_records:
            return f"Step {timestep_idx}"
        t = ts_records[0]["time"]
        time_str = t.strftime("%b %d, %H:%M UTC")
        return f"Step {timestep_idx} — {time_str}"
