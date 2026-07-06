"""Shared plotting style and figure-saving helpers for the SolarOscFitter utils.

This module is the single source of truth for the publication look used across
the plotting scripts (CMS/mplhep base + LaTeX rendering + larger tick/axis
sizes). Importing it applies the style as a side effect, matching the previous
import-time behaviour of ``plotting.py``; call :func:`apply_style` explicitly to
re-apply after any local ``rcParams`` override.

Usage::

    import style            # applies the style on import
    style.save_figure(fig, "images/foo.pdf", dpi=300)
"""
import os

import matplotlib.pyplot as plt
import mplhep as hep


# --- shared plotting conventions (single source of truth for the scripts) ---
SIGMA_LEVELS     = [0.6827, 0.9545, 0.9973]   # 1, 2, 3-sigma enclosed probabilities
SIGMA_LINESTYLES = [':', '-.', '-']           # 3-sigma -> 1-sigma contour line styles
WHITE_MIX        = [0.15, 0.6]                 # 3-sigma, 2-sigma colour->white mix (1-sigma = full colour)

# posterior-comparison palette, consumed by compareOutput.py
COMPARE_COLORS    = ("#003CFF", "#FA2727FF")    # posterior A (prussian), B (vermillion)
COMPARE_SOLAR     = "#FFD900"                   # global-solar contour colour (contrasts with A/B)

def apply_style():
    """Apply the shared CMS/LaTeX rcParams used by all plotting scripts.

    Note: ``text.usetex`` is always on — every plotting script depends on a
    working LaTeX/dvipng toolchain being installed.
    """
    plt.rcParams['text.usetex'] = True
    hep.style.use("CMS")
    plt.rcParams.update({
        'font.size': 14,           # Default text size (affects all text)
        'axes.titlesize': 16,      # Title size
        'axes.labelsize': 32,
        'axes.linewidth': 2.5,

        'xtick.labelsize': 30,
        'ytick.labelsize': 30,     # Y-axis tick label size

        "xtick.major.width": 2.5,
        "ytick.major.width": 2.5,
        "xtick.minor.width": 1.5,
        "ytick.minor.width": 1.5,

        "hatch.linewidth": 2.4,    # used by plotBinDiagnostics hatching
    })


def ensure_parent_dir(path):
    """Create the parent directory of ``path`` if it does not exist.

    Use before opening a writer that does not create directories itself
    (e.g. ``matplotlib.backends.backend_pdf.PdfPages``), so writes to
    ``images/`` never crash on a fresh checkout.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def save_figure(fig, path, **kwargs):
    """Save ``fig`` to ``path``, creating the parent directory if needed.

    A thin wrapper over ``fig.savefig`` that guarantees the destination
    directory exists, so writes to ``images/`` never crash on a fresh checkout.
    Extra keyword arguments are forwarded to ``savefig`` (e.g. ``dpi=300``).
    """
    ensure_parent_dir(path)
    fig.savefig(path, **kwargs)


# Apply on import so a bare ``import style`` is enough to set the look.
apply_style()
