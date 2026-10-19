"""Render the paper's five figures from existing, read-only experiment outputs.

Usage: python scripts/plot_paper_figures.py [--project-root PATH]

No model, simulation, bootstrap, statistical test, or confidence interval is run.
The displayed confidence limits are imported from the original summaries.
The only transformation of bootstrap data is histogram binning for display.
Writes PDF/SVG/400-dpi PNG and provenance under this revision's figures/ only.
Requires numpy, matplotlib and the manuscript's TeX Gyre Termes fonts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
from paper_plot_style import set_style, WIDTH, INK, M0, M4, GRAY, LIGHT_GRAY, FILL


REVISION = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, folder: Path, stem: str) -> list[str]:
    paths = []
    fig.canvas.draw()
    for extension in ("pdf", "svg", "png"):
        path = folder / f"{stem}.{extension}"
        options = {"dpi": 400} if extension == "png" else {}
        if extension == "pdf":
            options["backend"] = "pgf"
        fig.savefig(path, **options)
        paths.append(path.name)
    plt.close(fig)
    return paths


def clean_axis(ax: plt.Axes, percent: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.tick_params(direction="out", length=3)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def fig1(folder: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(WIDTH, 4.8))
    fig.subplots_adjust(left=.015, right=.985, bottom=.025, top=.985)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    xs = [.165, .5, .835]
    widths = [.29, .29, .29]
    headings = ["Observed statistic", "M0: fixed path", "M4: full pipeline"]
    colors = [INK, M0, M4]
    for x, title, color in zip(xs, headings, colors):
        ax.text(x, .975, title, ha="center", va="top", fontsize=10,
                color=color, weight="bold")
    ax.text(xs[0], .925, "Unchanged by calibration", ha="center", fontsize=9)
    ax.text(xs[1], .925, "Conditional multipliers", ha="center", fontsize=9)
    ax.text(xs[2], .925, "Fitted stationary null", ha="center", fontsize=9)

    def box(x: float, y: float, text: str, color: str, width: float, h: float=.079):
        patch = FancyBboxPatch((x-width/2, y-h/2), width, h,
                               boxstyle="square,pad=0.005",
                               linewidth=.5, edgecolor=color, facecolor="white")
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5,
                linespacing=1.22)
        return (x, y-h/2-.007), (x, y+h/2+.007)

    def chain(x, ys, texts, color, width):
        last = None
        for y, text in zip(ys, texts):
            bottom, top = box(x, y, text, color, width)
            if last:
                ax.add_patch(FancyArrowPatch(last, top, arrowstyle="-|>",
                                            mutation_scale=7, linewidth=.6,
                                            color=color, shrinkA=0, shrinkB=0))
            last = bottom

    chain(xs[0], np.linspace(.845, .185, 7), [
        "Longitudinal trajectories",
        "Candidate splits\nand risk sets",
        "Left / right\nfitted Q iteration",
        "Q-function contrasts",
        "Cluster studentization\nand boundary weighting",
        "Maximize over candidate\nand state-action grid",
        r"Observed $D_4$ and $\hat\tau$",
    ], colors[0], widths[0])
    chain(xs[1], [.845, .68, .515, .35, .185], [
        "Hold states, actions,\nsupport and Q fits fixed",
        "Draw one multiplier\nper participant",
        "Perturb the fitted\ninfluence process",
        "Use observed variance\nand the same maximization",
        r"Multiplier reference $D_4^*$",
    ], colors[1], widths[1])
    chain(xs[2], [.845, .713, .581, .449, .317, .185], [
        "Fit trajectory generator\nunder the stationary null",
        "Regenerate full\nstate-action-reward paths",
        "Rebuild the panel; check\nthe same candidate support",
        "Refit left / right FQI\nfor every candidate",
        "Recompute variance,\nweighting and maximization",
        r"Full-pipeline reference $D_4^*$",
    ], colors[2], widths[2])
    ax.plot([.02, .98], [.105, .105], color=LIGHT_GRAY, linewidth=.75)
    ax.text(.5, .073,
            r"Compare the same observed $D_4$ with each calibration distribution",
            ha="center", va="center", fontsize=9)
    ax.text(.5, .023,
            "The evaluation grid is fixed. A draw losing candidate support is invalid, not repaired.",
            ha="center", va="center", fontsize=9, color=GRAY)
    return save(fig, folder, "fig1_method_pipeline")


def fig2(folder: Path, rows: list[dict]) -> tuple[list[str], list[dict]]:
    rows = [r for r in rows if r["layer"] == "D4"]
    assert len(rows) == 9
    labels = ["H0\nReference", "H1\nPatient\neffects",
              "H2\nDesign\nimbalance", "H3\nReward\ntails",
              "H4\nTransition\ntails", "H5\nHetero-\nskedasticity",
              "H6\nReward\nAR(1)", "H7\nWeaker\noverlap",
              "H8\nCombined\nstress"]
    values = []
    for row in rows:
        assert int(row["outer_replicates"]) == int(row["testable_replicates"]) == 200
        values.append({"scenario_id": row["scenario_id"], "n": 200,
                       "rejections": int(row["rejections_005"]),
                       "rate": float(row["size_005"]),
                       "saved_wilson95": [float(row["wilson_low_005"]),
                                          float(row["wilson_high_005"])]})
    rate = np.array([v["rate"] for v in values])
    ci = np.array([v["saved_wilson95"] for v in values])
    fig, ax = plt.subplots(figsize=(WIDTH, 3.25))
    fig.subplots_adjust(left=.10, right=.985, top=.90, bottom=.25)
    x = np.arange(9)
    ax.axhline(.05, color=GRAY, linestyle=(0, (4, 3)), linewidth=1,
               label="Nominal 5%", zorder=1)
    ax.errorbar(x, rate, yerr=np.vstack((rate-ci[:, 0], ci[:, 1]-rate)),
                fmt="o", color=M4, capsize=2.5, markersize=4.2,
                linewidth=.85, label="M4 (95% Wilson interval)", zorder=3)
    ax.set(xlim=(-.5, 8.55), ylim=(0, .13), ylabel=r"$D_4$ null rejection rate",
           xticks=x, xticklabels=labels)
    ax.yaxis.set_major_locator(MultipleLocator(.02))
    ax.tick_params(axis="x", labelsize=9, length=0, pad=7)
    clean_axis(ax, True)
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(.5, 1.01),
              handlelength=2.1, columnspacing=1.7)
    return save(fig, folder, "fig2_null_stress"), values


def fig3(folder: Path, data: dict) -> tuple[list[str], dict]:
    values = {name: data[key] for name, key in [("M0", "Original"), ("M4-HS", "M4_HS")]}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    fig.subplots_adjust(left=.10, right=.985, bottom=.23, top=.97)
    ax.axhline(.05, color=GRAY, linestyle=(0, (4, 3)), linewidth=1)
    for x, (label, value), color, marker in zip([0, 1], values.items(), [M0, M4], ["s", "o"]):
        rate = value["estimate"]; lo, hi = value["wilson95"]
        assert value["analyzable"] == 200
        ax.errorbar(x, rate, yerr=[[rate-lo], [hi-rate]], fmt=marker,
                    color=color, markerfacecolor="white" if label == "M0" else color,
                    capsize=3, markersize=4.5, linewidth=.9)
        ax.text(x+.085, rate, f"{rate:.1%}\n({value['reject']}/200)",
                ha="left", va="center", fontsize=9, color=INK)
    ax.text(1.45, .0525, "Nominal 5%", color=GRAY, fontsize=9, ha="right", va="bottom")
    ax.set(xlim=(-.55, 1.5), ylim=(0, .16), ylabel=r"$D_4$ null rejection rate",
           xticks=[0, 1], xticklabels=["M0\nFixed-path calibration", "M4-HS\nFull-pipeline calibration"])
    ax.yaxis.set_major_locator(MultipleLocator(.05)); clean_axis(ax, True)
    ax.tick_params(axis="x", length=0, pad=8, labelsize=9)
    return save(fig, folder, "fig3_hs_type1_comparison"), values


def fig4(folder: Path, rows: list[dict]) -> tuple[list[str], list[dict]]:
    scenarios = ["reward_moderate", "reward_strong", "transition_moderate", "transition_strong"]
    labels = ["Reward\nmoderate", "Reward\nstrong", "Transition\nmoderate", "Transition\nstrong"]
    values = []
    fig, ax = plt.subplots(figsize=(WIDTH, 2.95))
    fig.subplots_adjust(left=.10, right=.985, top=.86, bottom=.22)
    for shift, method, label, color, marker in [(-.16, "Original RS-CUSUM", "M0", M0, "s"),
                                               (.16, "Locked M4", "M4", M4, "o")]:
        selected = [next(r for r in rows if r["Scenario"] == scenario and r["Method"] == method)
                    for scenario in scenarios]
        y = [float(r["Detection Rate"]) for r in selected]
        ax.scatter(np.arange(4)+shift, y, facecolors="white" if label == "M0" else color,
                   edgecolors=color, linewidths=.85, marker=marker, s=22, label=label, zorder=4)
        for i, rate in enumerate(y):
            ax.annotate(f"{rate:.0%}", (i+shift, rate), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=INK, fontsize=9)
        for row in selected:
            assert int(row["N"]) == 100
            values.append({"scenario": row["Scenario"], "source_method": method, "plot_method": label,
                           "n": int(row["N"]), "detections": int(row["Detections"]),
                           "rate": float(row["Detection Rate"]), "MCSE": float(row["MCSE"])})
    ax.set(xticks=range(4), xticklabels=labels, xlim=(-.5, 3.5), ylim=(0, 1.1),
           ylabel="Detection probability")
    ax.yaxis.set_major_locator(MultipleLocator(.2)); clean_axis(ax, True)
    ax.tick_params(axis="x", length=0, pad=8, labelsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.09), ncol=2, columnspacing=2)
    return save(fig, folder, "fig4_detection_power"), values


def fig5(folder: Path, candidate: list[dict], distribution: list[dict], calibration: list[dict]) -> tuple[list[str], dict]:
    days = np.array([float(r["candidate_elapsed_day"]) for r in candidate])
    process = np.array([float(r["D4_candidate_max_joint"]) for r in candidate])
    reference = np.array([float(r["D4"]) for r in distribution])
    result = next(r for r in calibration if r["layer"] == "D4")
    observed = float(result["observed"]); q95 = float(result["q95"])
    assert len(candidate) == 7 and len(reference) == int(result["valid_draws"]) == 199
    assert len(set(r["draw_id"] for r in distribution)) == 199
    assert np.isfinite(reference).all()
    assert np.isclose(process.max(), observed) and np.isclose(reference.max(), float(result["maximum"]))
    assert int(np.sum(reference >= observed)) == int(result["count_ge_observed"]) == 0
    maximum = int(np.argmax(process))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.25))
    fig.subplots_adjust(left=.09, right=.985, bottom=.20, top=.84, wspace=.38)
    a, b = axes
    for ax, panel in zip(axes, ['(a)', '(b)']):
        ax.text(0, 1.08, panel, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    a.plot(days, process, color=INK, marker="o", markersize=3.5, linewidth=1.0)
    a.scatter(days[maximum], observed, facecolors="white", edgecolors=INK, s=26, zorder=5)
    a.axvline(days[maximum], color=GRAY, linestyle=(0, (3, 3)), linewidth=.7)
    a.annotate(f"Day {days[maximum]:.1f}\n{observed:.3f}", (days[maximum], observed),
               xytext=(13, 10), textcoords="offset points", color=INK, fontsize=9,
               ha="left", va="bottom")
    a.set(xticks=days, xticklabels=[f"{d:g}" for d in days], xlabel="Elapsed study day",
          ylabel=r"Candidate-wise $D_4$", ylim=(0, 620), xlim=(7, 35))
    a.tick_params(axis="x", labelsize=8.5)
    a.yaxis.set_major_locator(MultipleLocator(100)); clean_axis(a)
    # Fixed display bins; no statistical estimation or new draws.
    bins = np.arange(100, 551, 25)
    counts, _, _ = b.hist(reference, bins=bins, color=FILL, edgecolor=M4, linewidth=.45)
    b.axvline(q95, color=GRAY, linestyle=(0, (4, 3)), linewidth=1.2,
              label=f"95th percentile: {q95:.3f}")
    b.axvline(observed, color=INK, linewidth=1.0, label=f"Observed: {observed:.3f}")
    b.set(xlim=(110, 530), ylim=(0, max(counts)*1.4), xlabel=r"Bootstrap $D_4^*$", ylabel="Number of draws")
    b.set_xticks([150, 250, 350, 450]); clean_axis(b)
    b.legend(loc="upper left", fontsize=8.5, handlelength=1.5, labelspacing=.6,
             borderaxespad=.2)
    return save(fig, folder, "fig5_heartsteps_application"), {
        "candidate_days": days.tolist(), "candidate_D4": process.tolist(),
        "observed_D4": observed, "maximizing_day": float(days[maximum]),
        "reference_draw_ids": [int(r["draw_id"]) for r in distribution],
        "reference_D4": reference.tolist(), "saved_q95": q95,
        "saved_reference_maximum": float(result["maximum"]),
        "saved_p_value": float(result["p_value"]), "saved_count_ge_observed": int(result["count_ge_observed"]),
        "histogram_edges": bins.tolist(), "histogram_counts": counts.astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REVISION.parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    folder = REVISION / "figures"
    folder.mkdir(exist_ok=True)
    sources = {
        "stress": "results_rs_cusum/phase2rh/scenario_type1_summary.csv",
        "paired": "type1_comparison/results/final_result.json",
        "alternatives": "results_rs_cusum/representative_altloc/detection_summary.csv",
        "heartsteps_candidates": "heartsteps_v1/results/candidate_process.csv",
        "heartsteps_reference": "heartsteps_v1/m4_hs/results/formal_b199/D0_D4_reference_distribution.csv",
        "heartsteps_calibration": "heartsteps_v1/m4_hs/results/formal_b199/formal_calibration_results.csv",
        "method_M0": "src/rs_cusum_phase2rb/engine.py",
        "method_M4": "src/rs_cusum_phase2rd/engine.py",
    }
    before = {key: {"path": relative, "sha256": sha256(root/relative)} for key, relative in sources.items()}
    set_style()
    outputs = {"fig1": {"files": fig1(folder), "sources": ["method_M0", "method_M4"]}}
    files, values = fig2(folder, csv_rows(root/sources["stress"]))
    outputs["fig2"] = {"files": files, "sources": ["stress"], "selector": "layer == D4; alpha = 0.05",
                       "fields": ["size_005", "rejections_005", "outer_replicates", "wilson_low_005", "wilson_high_005"], "values": values}
    files, values = fig3(folder, json.loads((root/sources["paired"]).read_text(encoding="utf-8")))
    outputs["fig3"] = {"files": files, "sources": ["paired"], "fields": ["Original", "M4_HS"], "values": values}
    files, values = fig4(folder, csv_rows(root/sources["alternatives"]))
    outputs["fig4"] = {"files": files, "sources": ["alternatives"], "fields": ["Scenario", "Method", "N", "Detections", "Detection Rate", "MCSE"], "values": values}
    files, values = fig5(folder, csv_rows(root/sources["heartsteps_candidates"]),
                        csv_rows(root/sources["heartsteps_reference"]), csv_rows(root/sources["heartsteps_calibration"]))
    outputs["fig5"] = {"files": files, "sources": ["heartsteps_candidates", "heartsteps_reference", "heartsteps_calibration"],
                       "fields": ["candidate_elapsed_day", "D4_candidate_max_joint", "draw_id", "D4", "q95", "observed", "maximum", "p_value"], "values": values}
    unchanged = all(sha256(root/entry["path"]) == entry["sha256"] for entry in before.values())
    assert unchanged, "A source changed while figures were being generated."
    provenance = {"scope": "Plotting of pre-existing results only; no experiments or interval estimation.",
                  "source_files_unchanged": unchanged, "font": "TeX Gyre Termes", "raster_dpi": 400,
                  "sources": before, "figures": outputs,
                  "figure_sha256": {name: sha256(folder/name) for item in outputs.values() for name in item["files"]}}
    (folder/"figure_data_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"figure_sets": len(outputs), "files": 15, "source_files_unchanged": unchanged,
                      "destination": str(folder)}, indent=2))


if __name__ == "__main__":
    main()
