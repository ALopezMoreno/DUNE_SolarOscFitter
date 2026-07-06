import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

FILES = {
    "longW":  "inputs/inputs_neutronOnly_may2026/Nbkg/3D/data_file_Nbkg_1mFidu_discri_100Emin_longW.csv",
    "Bottom": "inputs/inputs_neutronOnly_may2026/Nbkg/3D/data_file_Nbkg_1mFidu_discri_100Emin_Bottom.csv",
    "shortW": "inputs/inputs_neutronOnly_may2026/Nbkg/3D/data_file_Nbkg_1mFidu_discri_100Emin_shortW.csv",
}
COLORS = {"longW": "#0894bc", "Bottom": "#E69F00", "shortW": "#009E73"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Ereco histograms of joined Nbkg background CSVs, with and without mask."
    )
    parser.add_argument("-o", "--output", default=None, help="Output file (PDF/PNG). Omit to show interactively.")
    parser.add_argument("--bins", type=int, default=80, help="Number of histogram bins (default: 80)")
    parser.add_argument("--range", nargs=2, type=float, default=[2, 20], metavar=("EMIN", "EMAX"),
                        help="Energy range in MeV (default: 2 20)")
    return parser.parse_args()


def load_data():
    dfs = {}
    for name, rel_path in FILES.items():
        path = os.path.join(REPO_ROOT, rel_path)
        df = pd.read_csv(path)
        df["Ereco_MeV"] = df["Ereco"] * 1000.0
        dfs[name] = df
    return dfs


def draw_panel(ax, dfs, use_mask, bins, erange):
    stacked_data = []
    stacked_weights = []
    labels = []
    colors = []
    total = 0

    for name, df in dfs.items():
        sel = df["mask"] if use_mask else pd.Series(True, index=df.index)
        data = df.loc[sel, "Ereco_MeV"].values
        w = df.loc[sel, "weights"].values
        stacked_data.append(data)
        stacked_weights.append(w)
        labels.append(f"{name} (N={len(data):,})")
        colors.append(COLORS[name])
        total += len(data)

    ax.hist(
        stacked_data,
        bins=bins,
        range=erange,
        weights=stacked_weights,
        stacked=True,
        color=colors,
        label=labels,
        edgecolor="none",
        alpha=0.85,
    )

    title = f"{'Mask applied (mask=True)' if use_mask else 'No mask (all events)'}  N={total:,}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("$E_{\\mathrm{reco}}$ [MeV]", fontsize=11)
    ax.set_ylabel("Events / bin", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.tick_params(direction="in", top=True, right=True)


def main():
    args = parse_args()
    dfs = load_data()
    erange = tuple(args.range)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Nbkg ES backgrounds — joined longW + Bottom + shortW", fontsize=13)

    draw_panel(axes[0], dfs, use_mask=False, bins=args.bins, erange=erange)
    draw_panel(axes[1], dfs, use_mask=True,  bins=args.bins, erange=erange)

    axes[1].set_ylabel("")
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
