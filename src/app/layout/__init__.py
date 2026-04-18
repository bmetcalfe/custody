"""Shared layout utilities."""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc


def label_with_tooltip(label: str, tooltip: str, id_suffix: str) -> html.Span:
    """Return *label* with a small '?' that shows *tooltip* on hover."""
    return html.Span([
        label,
        html.Span(
            " ?",
            id=f"tooltip-target-{id_suffix}",
            style={
                "color": "#888",
                "fontSize": "10px",
                "marginLeft": "4px",
                "cursor": "pointer",
            },
        ),
        dbc.Tooltip(
            tooltip,
            target=f"tooltip-target-{id_suffix}",
            placement="top",
        ),
    ])
