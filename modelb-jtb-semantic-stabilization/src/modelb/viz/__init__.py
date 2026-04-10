"""Visualization helpers (Primitive-replication color scheme).

The plotting module contains the canonical palette used in Primitive-replication.
Use :func:`set_style` to apply consistent matplotlib defaults.
"""
from __future__ import annotations

def set_style():
    """Apply consistent matplotlib rcParams for figures."""
    try:
        import matplotlib as mpl
    except Exception:
        return
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
    })

__all__ = ["set_style"]
