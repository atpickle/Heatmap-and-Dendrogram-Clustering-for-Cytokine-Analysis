"""Compare LPS-only cytokines with w3 cytokines on one shared time/value axis.

Run this file for cytokine-only plots, or call
Plot_LPS_w3_Cytokine_Plots(cytokines=["TNFα"]) for a smaller selection.
The earlier dual-axis morphology function remains available explicitly.
Requires pandas, openpyxl, numpy, matplotlib and scipy.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


MORPHOLOGY_FILE = Path(
    "/Users/allisonpickle/Desktop/261016_Compiled NP BioReps/"
    "Compiled Analysis w3/All/Isolated microglia/w3_microglia_results.xlsx"
)
CYTOKINE_FILE = Path(
    "/Users/allisonpickle/Desktop/261007_Cytokine Reanalysis/"
    "Compiled_Cytokines_3BioReps.xlsx"
)
TREATMENT_LABELS = {
    "Tri_Ctrl": "Control",
    "Tri_w3": "w3 only (no LPS)",
    "24hrw3_5ng": "w3 24 h before 5 ng/mL LPS",
    "24hrw3_100ng": "w3 24 h before 100 ng/mL LPS",
    "1hrw3_5ng": "w3 1 h before 5 ng/mL LPS",
    "1hrw3_100ng": "w3 1 h before 100 ng/mL LPS",
    "Cow3_5ng": "w3 concurrent with 5 ng/mL LPS",
    "Cow3_100ng": "w3 concurrent with 100 ng/mL LPS",
    "5ng_LPS": "5 ng/mL LPS only",
    "100ng_LPS": "100 ng/mL LPS only",
}
NP_SAMPLE_MAP = {
    "10w3": ("Tri_w3", 24),
    "10 uM w3, 100ng LPS 24hr": ("Cow3_100ng", 24),
    "1hrw3_5ng, LPS24hr": ("1hrw3_5ng", 24),
    "1hrw3_100ng, LPS 24hr": ("1hrw3_100ng", 24),
    "24hr 10w3, 5ng LPS 24hr": ("24hrw3_5ng", 24),
    "24hr 10w3, 100ng LPS 24hr": ("24hrw3_100ng", 24),
}
METRICS = [
    "Area (µm²)", "Perimeter (µm)", "Convex Hull Area (µm²)",
    "Convex Hull Perimeter (µm)", "Circularity", "Aspect Ratio", "Roughness",
]


def _selection(requested, available, label):
    if requested is None:
        return list(available)
    if (
        not isinstance(requested, list) or not requested
        or any(not isinstance(x, str) for x in requested)
        or len(set(requested)) != len(requested)
    ):
        raise ValueError(f"{label} must be a nonempty list of unique names or None.")
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Unknown {label}: {sorted(unknown)}. Available: {list(available)}")
    return requested


def _numeric(values, source):
    result = pd.to_numeric(values, errors="coerce")
    if (values.notna() & result.isna()).any() or np.isinf(result).any():
        raise ValueError(f"Invalid/nonfinite numeric values in {source}.")
    return result


def _read_cytokines(path):
    """Read inclusive Excel rows; preserve stored units and negative values."""
    records = []
    selections = [("NP Analysis", 2, 121, 129), ("Timed LPS", 16, 305, 317)]
    with pd.ExcelFile(path) as book:
        for sheet, header, first, last in selections:
            raw = pd.read_excel(book, sheet_name=sheet, header=None)
            if len(raw) < last:
                raise ValueError(f"{sheet}: expected data through Excel row {last}.")
            headers = [
                (str(raw.iat[header - 1, col]).strip(), col + 1)
                for col in range(5, raw.shape[1] - 1, 3)
                if pd.notna(raw.iat[header - 1, col])
            ]
            if not headers or len({name for name, _ in headers}) != len(headers):
                raise ValueError(f"{sheet}: expected unique cytokine headers.")
            for row in range(first - 1, last):
                sample, rep = raw.iat[row, 1], raw.iat[row, 0]
                if pd.isna(sample) or pd.isna(rep):
                    raise ValueError(f"{sheet}, Excel row {row + 1}: missing sample/rep.")
                sample = str(sample).strip()
                if sheet == "NP Analysis":
                    if sample not in NP_SAMPLE_MAP:
                        raise ValueError(f"Unmapped NP cytokine sample: {sample!r}")
                    treatment, time = NP_SAMPLE_MAP[sample]
                else:
                    match = re.fullmatch(r"(\d+)hr\s+(5|100)ng", sample)
                    if not match:
                        if re.fullmatch(r"\d+hr\s+Ctrl", sample):
                            print(f"Plain-LPS reference: excluded control {sample!r}.")
                            continue
                        raise ValueError(f"Unmapped plain-LPS sample: {sample!r}")
                    time, dose = map(int, match.groups())
                    treatment = f"{dose}ng_LPS"
                for cytokine, col in headers:
                    records.append({
                        "Sample": sample, "Rep": str(rep).strip(),
                        "Treatment": treatment, "Time_hr": time,
                        "Cytokine": cytokine, "Value": raw.iat[row, col],
                        "Sheet": sheet, "Excel_Row": row + 1,
                    })
    observations = pd.DataFrame(records)
    observations["Value"] = _numeric(observations["Value"], str(path))
    print(f"Cytokine missing values omitted: {observations['Value'].isna().sum()}.")
    points = observations.dropna(subset=["Value"]).groupby(
        ["Sheet", "Cytokine", "Treatment", "Time_hr", "Rep"],
        sort=False, as_index=False,
    ).agg(Value=("Value", "mean"), TechnicalRows=("Value", "size"))
    if points.empty:
        raise ValueError("No numeric cytokine observations in the selected rows.")
    return observations, points


def _read_morphology(path, metrics):
    raw = pd.read_excel(path)
    required = {"File", "Bio Rep", *metrics}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing morphology columns: {sorted(required - set(raw.columns))}")
    if raw.empty or raw[["File", "Bio Rep"]].isna().any().any():
        raise ValueError("Morphology requires nonempty File and Bio Rep values.")
    # Pretreatment duration is part of Treatment, never the observation time.
    parsed = raw["File"].str.extract(
        r"^(?P<File_Rep>\d+)_(?P<Time_hr>\d+)hrs?_"
        r"(?P<Treatment>.+)_(?P<Well>\d+)_(?P<Image>\d+)_C_\d+\.tiff?$"
    )
    if parsed.isna().any().any():
        raise ValueError(f"Unrecognized morphology filenames: {raw.loc[parsed.isna().any(axis=1), 'File'].tolist()}")
    rep = _numeric(raw["Bio Rep"], "morphology Bio Rep")
    mismatch = ~rep.eq(parsed["File_Rep"].astype(int))
    if mismatch.any():
        print(
            "Using confirmed filename replicate instead of workbook Bio Rep: "
            f"{raw.loc[mismatch, ['File', 'Bio Rep']].to_dict('records')}"
        )
    if not parsed["Treatment"].isin(TREATMENT_LABELS).all():
        raise ValueError("Unknown morphology treatments in filenames.")
    raw = pd.concat([raw, parsed], axis=1)
    raw["Bio Rep"] = raw["File_Rep"].astype(int)
    raw["Time_hr"] = raw["Time_hr"].astype(int)
    frames = []
    for metric in metrics:
        values = raw.copy()
        values["Results"] = _numeric(values[metric], metric)
        print(f"{metric}: missing morphology rows omitted: {values['Results'].isna().sum()}.")
        values = values.dropna(subset=["Results"])
        if values.empty:
            raise ValueError(f"No numeric morphology values for {metric}.")
        wells = values.groupby(
            ["Treatment", "Time_hr", "Bio Rep", "Well"], sort=False, as_index=False
        ).agg(Results=("Results", "mean"), Number_Images=("Results", "size"))
        if wells["Results"].le(0).any():
            raise ValueError(f"{metric}: ln morphology requires positive well means.")
        for scale in ("raw", "ln"):
            transformed = wells.copy()
            if scale == "ln":
                transformed["Results"] = np.log(transformed["Results"])
            bio = transformed.groupby(
                ["Treatment", "Time_hr", "Bio Rep"], sort=False, as_index=False
            ).agg(
                Results=("Results", "mean"), SEM=("Results", "sem"),
                Number_Wells=("Well", "size"), Number_Images=("Number_Images", "sum"),
            ).rename(columns={"Bio Rep": "Biological_Rep"})
            single = bio.loc[bio["Number_Wells"].eq(1)]
            if not single.empty:
                print(f"{metric}, {scale}: SEM unavailable for {len(single)} single-well groups.")
            bio["Metric"] = metric
            bio["Morphology_Scale"] = scale
            bio["Source_Workbook"] = str(Path(path).resolve())
            frames.append(bio)
    return pd.concat(frames, ignore_index=True)


def Plot_w3_Cytokine_Morphology_Plots(
    morphology_file=MORPHOLOGY_FILE,
    cytokine_file=CYTOKINE_FILE,
    *,
    cytokines=None,
    metrics=None,
    output_dir=None,
    save_results=True,
    display_results=False,
):
    """Plot matched NP and plain-LPS-reference views separately.

    Morphology: images averaged within wells, then equal-weight well means per
    biological replicate, with SEM across wells. For ln plots, transform well
    means before calculating replicate means/SEM (no offset).
    Replicate IDs come from filenames, as experimentally confirmed; disagreements
    with the workbook Bio Rep column are reported without modifying the workbook.
    Cytokines: exact Obs Conc. columns (G, J, ...), as stored, without another
    transformation or censoring substitution. Technical repeats are averaged
    within condition/time/replicate. Replicates are not paired between assays.

    NP sample mapping explicitly assigns w3-only to 24 h and the unnamed
    w3+100ng pretreatment to concurrent addition, per experimental confirmation.
    Two views use the SAME w3 morphology: matched NP cytokines versus plain-LPS
    reference cytokines. Plain-LPS morphology is not imported. Missing groups
    remain absent; cytokine values are not extended to other times.
    Produces 5ng, 100ng and combined-dose plots, raw/ln morphology, and selected
    metric horizontal cytokine compilations, like the original plot function.
    This is descriptive only: no tests, cross-assay correlations or pairing.

    Returns cytokine replicate points, morphology replicate points and PNG paths.
    Saves source cytokine rows and replicate tables alongside plots. Sources and
    the original analysis script are never modified.
    """
    metrics = _selection(metrics, METRICS, "metrics")
    observations, cyto_points = _read_cytokines(cytokine_file)
    cytokines = _selection(
        cytokines, list(observations["Cytokine"].unique()), "cytokines"
    )
    observations = observations.loc[observations["Cytokine"].isin(cytokines)].copy()
    cyto_points = cyto_points.loc[cyto_points["Cytokine"].isin(cytokines)].copy()
    morphology = _read_morphology(morphology_file, metrics)
    output = Path(output_dir) if output_dir is not None else Path(morphology_file).parent / "Cytokine_w3_Morphology_Plots"
    if save_results:
        output.mkdir(parents=True, exist_ok=True)
        observations.to_csv(output / "Cytokine_Source_Observations.csv", index=False)
        cyto_points.to_csv(output / "Cytokine_Replicate_Points.csv", index=False)
        morphology.to_csv(output / "Morphology_Replicate_Points.csv", index=False)
    colors = dict(zip(TREATMENT_LABELS, plt.get_cmap("tab10").colors))
    paths = []
    compilation_panels = {}

    def trend(axis, points, column, treatment, linestyle):
        means = points.groupby("Time_hr")[column].mean().sort_index()
        if len(means) >= 3:
            x = np.linspace(means.index.min(), means.index.max(), 200)
            y = PchipInterpolator(means.index, means.values, extrapolate=False)(x)
            axis.plot(x, y, color=colors[treatment], linestyle=linestyle)
        elif len(means) == 2:
            axis.plot(means.index, means.values, color=colors[treatment], linestyle=linestyle)

    for metric in metrics:
        for scale in ("raw", "ln"):
            bio = morphology.loc[
                morphology["Metric"].eq(metric) & morphology["Morphology_Scale"].eq(scale)
            ]
            metric_label = metric if scale == "raw" else f"ln({metric})"
            for view, sheet in (("Matched_w3", "NP Analysis"), ("Plain_LPS_reference", "Timed LPS")):
                for dose in (5, 100, None):
                    treatments = [
                        t for t in TREATMENT_LABELS
                        if t in set(bio["Treatment"])
                        and (dose is None or t in ("Tri_Ctrl", "Tri_w3") or t.endswith(f"_{dose}ng"))
                    ]
                    morph = bio.loc[bio["Treatment"].isin(treatments)]
                    for cytokine in cytokines:
                        cyto = cyto_points.loc[
                            cyto_points["Sheet"].eq(sheet) & cyto_points["Cytokine"].eq(cytokine)
                        ]
                        wanted = treatments if view == "Matched_w3" else [
                            f"{d}ng_LPS" for d in (5, 100) if dose is None or d == dose
                        ]
                        cyto = cyto.loc[cyto["Treatment"].isin(wanted)]
                        available_doses = {
                            d for d in (5, 100)
                            if any(
                                t.endswith(f"_{d}ng") or t == f"{d}ng_LPS"
                                for t in cyto["Treatment"]
                            )
                        }
                        required_doses = {5, 100} if dose is None else {dose}
                        if not required_doses.issubset(available_doses):
                            print(
                                f"Skipping {metric}, {cytokine}, {view}, {dose}: "
                                f"no numeric cytokines for doses {sorted(required_doses - available_doses)}."
                            )
                            continue
                        missing = sorted(set(wanted) - set(cyto["Treatment"]))
                        print(
                            f"{cytokine}, {view}, dose={dose}: absent cytokine conditions {missing}; "
                            f"available hours {sorted(cyto['Time_hr'].unique())}."
                        )
                        fig, left = plt.subplots(figsize=(14, 7))
                        try:
                            right = left.twinx()
                            rng = np.random.default_rng(42)
                            for axis, frame, column, marker, mean_marker, style in (
                                (left, morph, "Results", "^", "s", "--"),
                                (right, cyto, "Value", "o", "D", "-"),
                            ):
                                for treatment, group in frame.groupby("Treatment", sort=False):
                                    x = group["Time_hr"].to_numpy() + rng.uniform(-0.5, 0.5, len(group))
                                    axis.scatter(x, group[column], marker=marker, color=colors[treatment], s=45)
                                    if column == "Results":
                                        valid = group["SEM"].notna().to_numpy()
                                        axis.errorbar(
                                            x[valid], group.loc[valid, column],
                                            yerr=group.loc[valid, "SEM"], fmt="none",
                                            ecolor=colors[treatment], capsize=3,
                                        )
                                    means = group.groupby("Time_hr")[column].mean()
                                    axis.scatter(means.index, means.values, marker=mean_marker,
                                                 color=colors[treatment], edgecolors="black", s=55)
                                    trend(axis, group, column, treatment, style)
                            ticks = sorted(set(morph["Time_hr"]) | set(cyto["Time_hr"]))
                            left.set_xticks(ticks, [f"{t:g} hr" for t in ticks])
                            left.set_xlim(min(ticks) - 3, max(ticks) + 3)
                            dose_label = "5 and 100 ng/mL" if dose is None else f"{dose} ng/mL"
                            left.set(xlabel="Observation time", ylabel=metric_label,
                                     title=f"{cytokine} and {metric_label}\n{view.replace('_', ' ')} — {dose_label}")
                            right.set_ylabel(f"{cytokine}: cytokine value (as stored)")
                            left.grid(axis="y", linestyle="--", alpha=0.25)
                            plotted = set(morph["Treatment"]) | set(cyto["Treatment"])
                            handles = [
                                Line2D([0], [0], color=colors[t], label=TREATMENT_LABELS[t])
                                for t in TREATMENT_LABELS if t in plotted
                            ]
                            handles.extend([
                                Line2D([0], [0], color="black", marker="^", linestyle="--",
                                       label="Morphology bio rep ± SEM (left)"),
                                Line2D([0], [0], color="black", marker="o", label="Cytokine replicate (right)"),
                                Line2D([0], [0], color="black", marker="s", linestyle="none", label="Morphology mean"),
                                Line2D([0], [0], color="black", marker="D", linestyle="none", label="Cytokine mean"),
                            ])
                            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.77, 0.96), fontsize=8)
                            fig.subplots_adjust(right=0.69, bottom=0.13)
                            if save_results:
                                safe = lambda s: re.sub(r"[^\w-]+", "_", s).strip("_")
                                dose_tag = "5ng_100ng" if dose is None else f"{dose}ng"
                                path = output / f"DualAxis_{safe(metric)}__{safe(cytokine)}__{view}__{dose_tag}__morphology_{scale}.png"
                                fig.savefig(path, dpi=300, bbox_inches="tight")
                                paths.append(str(path))
                                if (metric in ("Area (µm²)", "Perimeter (µm)") and scale == "ln") or (metric == "Circularity" and scale == "raw"):
                                    compilation = output / f"Compilation_{safe(metric)}__{view}__{dose_tag}__morphology_{scale}.png"
                                    compilation_panels.setdefault(compilation, []).append(path)
                            if display_results:
                                plt.show()
                        finally:
                            plt.close(fig)
    if save_results:
        from PIL import Image

        for path, panels in compilation_panels.items():
            sizes = []
            for panel in panels:
                with Image.open(panel) as image:
                    sizes.append(image.size)
            with Image.new("RGB", (sum(w for w, _ in sizes), max(h for _, h in sizes)), "white") as result:
                x = 0
                for panel in panels:
                    with Image.open(panel) as image:
                        result.paste(image, (x, 0))
                        x += image.width
                result.save(path, dpi=(300, 300))
            paths.append(str(path))
        print(f"Saved {len(paths)} PNGs to {output.resolve()}.")
    return cyto_points, morphology, paths


def Plot_LPS_w3_Cytokine_Plots(
    cytokine_file=CYTOKINE_FILE,
    *,
    cytokines=None,
    output_dir=None,
    save_results=True,
    display_results=False,
):
    """Compare selected LPS-only and w3 cytokines without reading morphology.

    Uses NP Analysis rows 121–129 and Timed LPS rows 305–317, preserving
    stored values and negative values. No additional transformation is applied.
    Each cytokine gets 5 ng/mL, 100 ng/mL and combined-dose time plots where
    both datasets have numeric dose-specific data. w3-only points are included
    as context, not counted as dose-matched observations. Missing conditions
    are reported and never imputed or extended to other observation times.

    Points show biological replicates, diamonds show means, and error bars
    show SEM across replicates (unavailable at n=1). LPS-only means are joined
    by straight lines; isolated w3 time points are not extrapolated.
    Small fixed treatment offsets and replicate jitter separate overlapping
    24-hour groups without changing their underlying Time_hr values.
    This is descriptive only: no tests, pairing, or cross-assay correlations.
    Returns replicate points, group summaries and saved PNG paths.
    """
    observations, points = _read_cytokines(cytokine_file)
    cytokines = _selection(
        cytokines, list(observations["Cytokine"].unique()), "cytokines"
    )
    observations = observations.loc[observations["Cytokine"].isin(cytokines)].copy()
    points = points.loc[points["Cytokine"].isin(cytokines)].copy()
    summary = points.groupby(
        ["Sheet", "Cytokine", "Treatment", "Time_hr"], sort=False, as_index=False
    ).agg(Mean=("Value", "mean"), SEM=("Value", "sem"), N=("Value", "size"))
    output = Path(output_dir) if output_dir is not None else Path(cytokine_file).parent / "LPS_vs_w3_Cytokine_Plots"
    if save_results:
        output.mkdir(parents=True, exist_ok=True)
        observations.to_csv(output / "Cytokine_Source_Observations.csv", index=False)
        points.to_csv(output / "Cytokine_Replicate_Points.csv", index=False)
        summary.to_csv(output / "Cytokine_Group_Summary.csv", index=False)
    colors = dict(zip(TREATMENT_LABELS, plt.get_cmap("tab10").colors))
    paths = []
    for cytokine in cytokines:
        data = points.loc[points["Cytokine"].eq(cytokine)]
        for dose in (5, 100, None):
            doses = (5, 100) if dose is None else (dose,)
            treatments = [
                t for t in TREATMENT_LABELS
                if t == "Tri_w3" or any(
                    t.endswith(f"_{d}ng") or t == f"{d}ng_LPS" for d in doses
                )
            ]
            frame = data.loc[data["Treatment"].isin(treatments)]
            missing = []
            for d in doses:
                if not frame["Treatment"].eq(f"{d}ng_LPS").any():
                    missing.append(f"{d} ng/mL LPS-only")
                if not frame["Treatment"].str.endswith(f"_{d}ng").any():
                    missing.append(f"{d} ng/mL w3+LPS")
            if missing:
                print(f"Skipping {cytokine}, dose={dose}: no numeric data for {missing}.")
                continue
            absent = sorted(set(treatments) - set(frame["Treatment"]))
            print(f"{cytokine}, dose={dose}: absent conditions {absent}.")
            means = summary.loc[
                summary["Cytokine"].eq(cytokine)
                & summary["Treatment"].isin(frame["Treatment"])
            ]
            present = [t for t in treatments if t in set(frame["Treatment"])]
            offsets = dict(zip(present, np.linspace(-1.5, 1.5, len(present))))
            fig, axis = plt.subplots(figsize=(13, 6))
            try:
                rng = np.random.default_rng(42)
                for treatment in present:
                    group = frame.loc[frame["Treatment"].eq(treatment)]
                    group_means = means.loc[means["Treatment"].eq(treatment)].sort_values("Time_hr")
                    is_lps = treatment.endswith("_LPS")
                    axis.scatter(
                        group["Time_hr"] + offsets[treatment] + rng.uniform(-0.15, 0.15, len(group)),
                        group["Value"], color=colors[treatment],
                        marker="o" if is_lps else "^", s=45,
                    )
                    axis.errorbar(
                        group_means["Time_hr"] + offsets[treatment], group_means["Mean"],
                        yerr=group_means["SEM"].fillna(0), color=colors[treatment],
                        marker="D", markeredgecolor="black", capsize=3,
                        linestyle="-" if is_lps else "--", label=TREATMENT_LABELS[treatment],
                    )
                    single = group_means.loc[group_means["N"].eq(1), "Time_hr"].tolist()
                    if single:
                        print(f"{cytokine}, {treatment}: SEM unavailable (n=1) at {single} hr.")
                ticks = sorted(frame["Time_hr"].unique())
                axis.set_xticks(ticks, [f"{t:g} hr" for t in ticks])
                axis.set_xlim(min(ticks) - 4, max(ticks) + 4)
                dose_label = "5 and 100 ng/mL" if dose is None else f"{dose} ng/mL"
                axis.set(
                    xlabel="Observation time (points offset for visibility)",
                    ylabel=f"{cytokine}: cytokine value (as stored)",
                    title=f"{cytokine}: LPS only vs w3 conditions — {dose_label}",
                )
                axis.grid(axis="y", linestyle="--", alpha=0.25)
                handles, _ = axis.get_legend_handles_labels()
                handles.extend([
                    Line2D([0], [0], color="black", marker="o", linestyle="none", label="LPS-only replicate"),
                    Line2D([0], [0], color="black", marker="^", linestyle="none", label="w3 replicate"),
                    Line2D([0], [0], color="black", marker="D", label="Mean ± SEM across replicates"),
                ])
                axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
                fig.tight_layout()
                if save_results:
                    safe = re.sub(r"[^\w-]+", "_", cytokine).strip("_")
                    tag = "5ng_100ng" if dose is None else f"{dose}ng"
                    path = output / f"Cytokine_{safe}__LPS_vs_w3__{tag}.png"
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    paths.append(str(path))
                if display_results:
                    plt.show()
            finally:
                plt.close(fig)
    if save_results:
        print(f"Saved {len(paths)} cytokine-only PNGs to {output.resolve()}.")
    return points, summary, paths


if __name__ == "__main__":
    Plot_LPS_w3_Cytokine_Plots()
