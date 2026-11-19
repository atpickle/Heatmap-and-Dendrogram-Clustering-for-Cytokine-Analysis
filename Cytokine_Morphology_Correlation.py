"""Cytokine-morphology correlation plots with optional time adjustment."""

_PAIRED_METRICS = {
    "Area (µm²)": ("ln(area + 0.5)", "ln_area_plus_0p5", 0.5),
    "Perimeter (µm)": ("ln(perimeter)", "ln_perimeter", 0.0),
    "Circularity": ("circularity", "raw_circularity", None),
    "AUC": ("ln(Sholl AUC)", "ln_sholl_auc", 0.0),
}


def Plot_Cytokine_Morphology_Correlation_Plots(
    cytokine_points,
    morphology_points,
    *,
    cytokines=None,
    metrics=None,
    morphology_scale="ln",
    control_for_time=False,
    replicate_map=None,
    descriptive_only=False,
    label_inset_points=True,
    separate_panels=False,
    concentration_unit=None,
    output_dir=None,
    save_results=True,
    display_results=False,
):
    """Plot dual-axis assay time courses with morphology-regression insets.

    Inputs are the first two DataFrames returned by Plot_Cytokine_Morphology_Plots.
    Morphology must already be averaged within wells, then within biological
    replicates. The selected raw/ln scale is used as supplied, not transformed
    again. Cytokine technical repeats are averaged within Sample/Rep.
    Pass a verified one-to-one replicate_map (cytokine Rep label to morphology
    Biological_Rep label) with control_for_time=True for paired residuals.
    In that mode, supply raw positive cytokine concentrations in Obs_Conc
    and the per-culture, equal-well summary in Morphology_Transformed
    (legacy area inputs may supply Area_Log_Adjusted instead).
    The paired inset uses log2 cytokine on x and the metric-specific summary
    on y: ln(area + 0.5), ln(perimeter), raw circularity, or ln(Sholl AUC),
    controlling for categorical time and replicate within dose. Results
    still supplies the morphology values for the main time course.
    With descriptive_only=True, show the unadjusted linear fit, Pearson r,
    and R² for the condition means without testing significance.
    Set label_inset_points=False to hide only the individual inset annotations.
    Set separate_panels=True to save standalone time-course and correlation
    PNGs in metric/dose subfolders instead of a combined plot, with one
    separate legend PNG per panel folder. Standalone titles name the cytokine
    and morphology readout without units or dose. Standalone correlation plots omit
    the methods footer; fit notes stay in the separate legend.
    concentration_unit optionally labels the standalone time-course cytokine axis.
    Each assay's condition mean gives each biological replicate equal weight.
    Without an explicit replicate map, assays match by Sample and Time_hr,
    not by replicate ID. Only plain
    5/100 ng/mL LPS conditions are used; each dose has its own regression.

    By default the inset regresses matched condition means without adjustment.
    With control_for_time=True, each assay's matched means are separately
    regressed on linear exposure time (hours) within dose. The inset plots
    cytokine residuals against morphology residuals and reports partial R²
    (the square of their Pearson correlation). This requires at least four
    matched conditions, variation in time, and residual variation in both
    assays. Three conditions leave zero residual degrees of freedom and
    would force the residual correlation to ±1 whenever defined.
    These are exploratory associations across time, not replicate-level
    correlations or evidence of causation; no significance tests are reported.
    Missing conditions are reported and omitted from regression, never imputed.
    Time courses retain all available conditions separately for each assay:
    morphology replicates on the left (triangles), cytokine replicates on
    the right (circles), with dashed/solid descriptive trends calculated from
    condition means. Time-course mean markers are not shown. Sample
    colors are fixed by dose and time across assays, plot modes, and legends.
    Optional morphology SEM values are plotted
    unchanged (within-replicate well SEM from the existing morphology reader).
    Legends sit at the far right, outside both the time course and regression
    inset, with dedicated space so they do not overlap either plot.

    Returns matched condition means, regression statistics, and saved PNG paths.
    Saving writes PNGs and a workbook to output_dir (default:
    ./Cytokine_Morphology_Correlations). Time-adjusted files have distinct names
    and the workbook includes the residuals. Inputs are not modified.
    None selects all available cytokines/metrics; lists select exact names.
    """
    from pathlib import Path
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from scipy.interpolate import PchipInterpolator
    from scipy.stats import linregress

    if morphology_scale not in ("raw", "ln"):
        raise ValueError("morphology_scale must be 'raw' or 'ln'.")
    if not isinstance(control_for_time, bool):
        raise ValueError("control_for_time must be True or False.")
    paired = replicate_map is not None
    if descriptive_only and control_for_time:
        raise ValueError("descriptive_only cannot be combined with control_for_time.")
    if paired and (not control_for_time or not isinstance(replicate_map, dict) or not replicate_map
                   or any(not isinstance(label, str) or not label.strip()
                          for pair in replicate_map.items() for label in pair)
                   or len(set(replicate_map.values())) != len(replicate_map)):
        raise ValueError("Paired analysis requires control_for_time=True and a one-to-one replicate_map of label strings.")

    def prepare(frame, required, numeric, label):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{label} must be a nonempty DataFrame.")
        if not required.issubset(frame.columns):
            raise ValueError(f"{label}: missing columns {sorted(required - set(frame.columns))}")
        frame = frame.copy()
        for column in required - set(numeric):
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{label}: missing {column}.")
            frame[column] = frame[column].astype(str).str.strip()
        for column in numeric:
            original = frame[column]
            frame[column] = pd.to_numeric(original, errors="coerce")
            if (original.notna() & frame[column].isna()).any() or np.isinf(frame[column]).any():
                raise ValueError(f"{label}: invalid/nonfinite {column}.")
        if frame["Time_hr"].isna().any() or frame["Time_hr"].lt(0).any():
            raise ValueError(f"{label}: Time_hr must be finite and nonnegative.")
        labels = frame["Sample"].str.extract(
            r"^\s*(\d+)\s*(?:hr|h)\s*[_ ]+([\d.]+)\s*ng(?:/mL)?(?:\s*LPS)?\s*$",
            flags=re.I,
        )
        frame["LPS_ng_mL"] = pd.to_numeric(labels[1], errors="coerce")
        eligible = frame["LPS_ng_mL"].isin([5, 100])
        print(f"{label}: excluded {int((~eligible).sum())} non-plain-LPS/unsupported-dose rows.")
        frame = frame.loc[eligible].copy()
        if not frame["Time_hr"].eq(pd.to_numeric(labels.loc[eligible, 0])).all():
            raise ValueError(f"{label}: Sample hour labels disagree with Time_hr.")
        return frame

    cyto = prepare(
        cytokine_points, {"Cytokine", "Sample", "Rep", "Time_hr", "Obs_Conc"},
        ["Time_hr", "Obs_Conc"], "cytokine_points",
    )
    if (paired and isinstance(morphology_points, pd.DataFrame)
            and "Area_Log_Adjusted" in morphology_points and "Morphology_Transformed" not in morphology_points):
        morphology_points = morphology_points.assign(
            Morphology_Transformed=morphology_points["Area_Log_Adjusted"],
        )
    morph = prepare(
        morphology_points,
        {"Metric", "Morphology_Scale", "Sample", "Biological_Rep", "Time_hr", "Results"}
        | ({"Morphology_Transformed"} if paired else set()),
        ["Time_hr", "Results", *(["Morphology_Transformed"] if paired else []),
         *(["SEM"] if isinstance(morphology_points, pd.DataFrame)
                                and "SEM" in morphology_points.columns else [])],
        "morphology_points",
    )
    morph = morph.loc[morph["Morphology_Scale"].eq(morphology_scale)].copy()
    if "SEM" in morph:
        if morph["SEM"].lt(0).any():
            raise ValueError("Morphology SEM must be nonnegative.")
        if morph["SEM"].isna().any():
            print("Morphology: unavailable SEM omitted; replicate means remain visible.")
    else:
        print("Morphology: no SEM column supplied; replicate means shown without error bars.")

    def select_names(frame, column, requested):
        available = list(dict.fromkeys(frame[column]))
        if not available:
            raise ValueError(f"No eligible {column} data for the requested morphology scale.")
        if requested is None:
            return available
        if (
            not isinstance(requested, list) or not requested
            or any(not isinstance(name, str) or not name.strip() for name in requested)
        ):
            raise ValueError(f"{column} selection must be a nonempty list of names, or None.")
        if len(set(requested)) != len(requested):
            raise ValueError(f"{column} selection contains duplicates.")
        unknown = set(requested) - set(available)
        if unknown:
            raise ValueError(f"Unknown {column}: {sorted(unknown)}. Available: {available}")
        return requested

    cytokine_names = select_names(cyto, "Cytokine", cytokines)
    metric_names = select_names(morph, "Metric", metrics)
    if paired:
        unsupported = set(metric_names) - _PAIRED_METRICS.keys() - {"Area"}
        if unsupported:
            raise ValueError(f"No paired transform defined for metrics: {sorted(unsupported)}.")
    cyto = cyto.loc[cyto["Cytokine"].isin(cytokine_names)]
    morph = morph.loc[morph["Metric"].isin(metric_names)]
    keys = ["Sample", "Time_hr", "LPS_ng_mL"]
    if morph.duplicated(["Metric", *keys, "Biological_Rep"]).any():
        raise ValueError("morphology_points must contain one row per metric/condition/biological replicate.")
    for frame, value, label in ((cyto, "Obs_Conc", "Cytokine"), (morph, "Results", "Morphology")):
        print(f"{label}: missing measurements omitted: {int(frame[value].isna().sum())}.")
    if paired:
        for observed, mapped, assay in ((set(cyto["Rep"]), set(replicate_map), "cytokine"),
                                         (set(morph["Biological_Rep"]), set(replicate_map.values()), "morphology")):
            if observed != mapped:
                raise ValueError(f"replicate_map must cover the selected {assay} labels exactly: {sorted(observed)}.")
        conditions = cyto[["Time_hr", "LPS_ng_mL"]].drop_duplicates()
        morph = morph.merge(conditions, on=["Time_hr", "LPS_ng_mL"], validate="many_to_one")
        observed_cyto = cyto.dropna(subset=["Obs_Conc"])
        if observed_cyto.duplicated(["Cytokine", "Time_hr", "LPS_ng_mL", "Rep"]).any():
            raise ValueError("Paired analysis requires one measured cytokine value per cytokine, dose, time, and replicate.")
        if observed_cyto["Obs_Conc"].le(0).any():
            raise ValueError("Paired analysis requires positive measured cytokine values for log2.")
    cyto = cyto.dropna(subset=["Obs_Conc"]).groupby(
        ["Cytokine", *keys, "Rep"], sort=False, as_index=False,
    ).agg(Obs_Conc=("Obs_Conc", "mean"))
    cyto_means = cyto.groupby(["Cytokine", *keys], sort=False, as_index=False).agg(
        Cytokine_Mean=("Obs_Conc", "mean"), Cytokine_n=("Rep", "size"),
    )
    morph_means = morph.dropna(subset=["Results"]).groupby(
        ["Metric", *keys], sort=False, as_index=False,
    ).agg(Morphology_Mean=("Results", "mean"), Morphology_n=("Biological_Rep", "size"))
    matched_frames = []
    missing_pairs = []
    for cytokine in cytokine_names:
        for metric in metric_names:
            if paired:
                left = cyto.loc[cyto["Cytokine"].eq(cytokine)].copy()
                left["Biological_Rep"] = left["Rep"].map(replicate_map)
                left["Cytokine_Log2"] = np.log2(left["Obs_Conc"])
                right = morph.loc[morph["Metric"].eq(metric)].dropna(subset=["Morphology_Transformed"])
                right = right.rename(columns={"Sample": "Morphology_Sample"})
                join_keys = ["Time_hr", "LPS_ng_mL", "Biological_Rep"]
            else:
                left = cyto_means.loc[cyto_means["Cytokine"].eq(cytokine)]
                right = morph_means.loc[morph_means["Metric"].eq(metric)]
                join_keys = keys
            joined = left.merge(right, on=join_keys, how="outer", validate="one_to_one", indicator=True)
            missing = joined.loc[joined["_merge"].ne("both"), [*join_keys, "_merge"]].copy()
            if not missing.empty:
                print(f"{cytokine}, {metric}: unmatched conditions omitted: {missing.to_dict('records')}")
                if paired:
                    missing["Cytokine"] = cytokine
                    missing["Metric"] = metric
                    missing["Missing_Assay"] = missing.pop("_merge").map(
                        {"left_only": "Morphology", "right_only": "Cytokine"}
                    )
                    missing_pairs.append(missing)
            joined = joined.loc[joined["_merge"].eq("both")].drop(columns="_merge")
            if not joined.empty:
                label, _, _ = _PAIRED_METRICS.get(metric, _PAIRED_METRICS["Area (µm²)"])
                joined["Morphology_Scale"] = f"{label}, equal well weights" if paired else morphology_scale
                matched_frames.append(joined)
    if not matched_frames:
        raise ValueError("No numeric conditions match between the selected assays.")
    matched = pd.concat(matched_frames, ignore_index=True)
    output = Path(output_dir) if output_dir is not None else Path.cwd() / "Cytokine_Morphology_Correlations"
    if save_results:
        output.mkdir(parents=True, exist_ok=True)
    colors = {5: "#E65400", 100: "#0072B2"}
    condition_colors = {
        (5, 6): "#1f77b4", (5, 24): "#d62728", (5, 48): "#9467bd",
        (5, 72): "#f7b6d2", (5, 96): "#17becf",
        (100, 6): "#ffbb78", (100, 24): "#8c564b", (100, 48): "#2ca02c",
        (100, 72): "#bcbd22", (100, 96): "#7f7f7f",
    }
    fallback_palette = plt.get_cmap("tab20").colors

    def condition_color(dose, time):
        return condition_colors.get((dose, time), fallback_palette[int(time) % len(fallback_palette)])
    rng = np.random.default_rng(42)
    statistics, saved_paths = [], []
    panel_legends = {}
    if control_for_time:
        matched["Morphology_Residual"] = np.nan
        matched["Cytokine_Residual"] = np.nan
    for (cytokine, metric, dose), group in matched.groupby(
        ["Cytokine", "Metric", "LPS_ng_mL"], sort=False,
    ):
        group = group.sort_values("Time_hr")
        times = group["Time_hr"].to_numpy()
        x_name, y_name = ("Cytokine_Log2", "Morphology_Transformed") if paired else ("Morphology_Mean", "Cytokine_Mean")
        transform_label, transform_slug, _ = _PAIRED_METRICS.get(metric, _PAIRED_METRICS["Area (µm²)"])
        x, y = group[x_name].to_numpy(), group[y_name].to_numpy()
        row = dict(
            Cytokine=cytokine, Metric=metric, Morphology_Scale=group["Morphology_Scale"].iloc[0],
            LPS_ng_mL=dose, n_conditions=len(group),
        )
        if control_for_time:
            r_key, r2_key = "Partial_r", "Partial_R_squared"
            slope_key, intercept_key = "Residual_Slope", "Residual_Intercept"
        else:
            r_key, r2_key, slope_key, intercept_key = "Pearson_r", "R_squared", "Slope", "Intercept"
        row.update({name: np.nan for name in (r_key, r2_key, slope_key, intercept_key)})
        row["Status"] = "ok"
        inset_has_points = not control_for_time
        if paired:
            row["n_conditions"] = group["Time_hr"].nunique()
            row["n_pairs"] = len(group)
            row["n_biological_reps"] = group["Biological_Rep"].nunique()
            design = pd.get_dummies(group[["Time_hr", "Biological_Rep"]].astype(str)).to_numpy(dtype=float)
            centered = np.column_stack((x - x.mean(), y - y.mean()))
            coefficients, _, rank, _ = np.linalg.lstsq(design, centered, rcond=None)
            residuals = centered - design @ coefficients
            row["Residual_df"] = len(group) - rank - 1
            if row["n_conditions"] < 2 or row["n_biological_reps"] < 2:
                row["Status"] = "At least two times and two biological replicates required"
            elif row["Residual_df"] < 1:
                row["Status"] = "Insufficient residual degrees of freedom"
            else:
                matched.loc[group.index, ["Cytokine_Residual", "Morphology_Residual"]] = residuals
                x, y = residuals.T
                inset_has_points = True
                tolerance = np.finfo(float).eps * len(group) * np.linalg.norm(centered, axis=0)
                if (np.linalg.norm(residuals, axis=0) <= tolerance).any():
                    row["Status"] = "No residual variation after adjustment"
        elif control_for_time:
            row["Residual_df"] = len(group) - 3 if len(group) >= 3 and np.ptp(times) else np.nan
            if len(group) < 4:
                row["Status"] = "At least 4 matched conditions required"
            elif np.ptp(times) == 0:
                row["Status"] = "Exposure time does not vary"
            else:
                centered = np.column_stack((x - x.mean(), y - y.mean()))
                time = times - times.mean()
                time /= np.linalg.norm(time)
                residuals = centered - np.outer(time, time @ centered)
                matched.loc[group.index, ["Morphology_Residual", "Cytokine_Residual"]] = residuals
                x, y = residuals.T
                inset_has_points = True
                tolerance = np.finfo(float).eps * len(group) * np.linalg.norm(centered, axis=0)
                if (np.linalg.norm(residuals, axis=0) <= tolerance).any():
                    row["Status"] = "No residual variation after time adjustment"
        elif len(group) < 3:
            row["Status"] = "At least 3 matched conditions required"
        elif np.ptp(x) == 0 or np.ptp(y) == 0:
            row["Status"] = "Constant condition means; correlation undefined"
        if row["Status"] == "ok":
            fit = linregress(x, y)
            row.update({r_key: fit.rvalue, r2_key: fit.rvalue ** 2,
                        slope_key: fit.slope, intercept_key: fit.intercept})
        statistics.append(row)
        unit = "pairs" if paired else "conditions"
        print(f"{cytokine}, {metric}, {dose:g} ng/mL: {row['Status']}; n={len(group)} {unit}.")
        fig, ax = plt.subplots(figsize=(9, 5) if separate_panels else (20, 7))
        correlation_fig = None
        try:
            right = ax.twinx()
            if separate_panels:
                correlation_fig, inset = plt.subplots(figsize=(7, 5))
            else:
                fig.subplots_adjust(left=0.07, right=0.40, bottom=0.27, top=0.88)
                inset = ax.inset_axes([1.40, 0.15, 0.76, 0.72])
            points = cyto.loc[
                cyto["Cytokine"].eq(cytokine) & cyto["LPS_ng_mL"].eq(dose)
            ]
            morphology = morph.loc[
                morph["Metric"].eq(metric) & morph["LPS_ng_mL"].eq(dose)
            ].dropna(subset=["Results"])
            color = colors[dose]
            for axis, frame, value, marker, linestyle in (
                (ax, morphology, "Results", "^", "--"),
                (right, points, "Obs_Conc", "o", "-"),
            ):
                for time, sample_points in frame.groupby("Time_hr", sort=False):
                    positions = sample_points["Time_hr"] + rng.uniform(-0.6, 0.6, len(sample_points))
                    point_color = condition_color(dose, time)
                    axis.scatter(positions, sample_points[value], marker=marker,
                                 color=point_color, s=55, edgecolors="white",
                                 linewidths=0.6, zorder=3)
                    if value == "Results" and "SEM" in sample_points:
                        has_sem = sample_points["SEM"].notna()
                        axis.errorbar(
                            positions.loc[has_sem], sample_points.loc[has_sem, value],
                            yerr=sample_points.loc[has_sem, "SEM"], fmt="none",
                            ecolor=point_color, capsize=3, zorder=2,
                        )
                means = frame.groupby(["Sample", "Time_hr"])[value].mean().groupby("Time_hr").mean().sort_index()
                if len(means) >= 3:
                    grid = np.linspace(means.index.min(), means.index.max(), 200)
                    axis.plot(grid, PchipInterpolator(means.index, means.values)(grid),
                              color=color, linestyle=linestyle, linewidth=1.8)
                elif len(means) == 2:
                    axis.plot(means.index, means.values, color=color, linestyle=linestyle, linewidth=1.8)
            ticks = sorted(set(points["Time_hr"]) | set(morphology["Time_hr"]))
            display_metric = "Sholl AUC" if metric == "AUC" else metric
            readout = {"Area (µm²)": "Area", "Perimeter (µm)": "Perimeter"}.get(metric, display_metric)
            metric_label = f"ln({display_metric})" if morphology_scale == "ln" else display_metric
            ax.set_xticks(ticks)
            ax.set_xlim(min(ticks) - 3, max(ticks) + 3)
            panel_title = f"{cytokine} and {readout}"
            ax.set(xlabel="Exposure time (hr)", ylabel=metric_label,
                   title=panel_title if separate_panels else f"{cytokine} and {metric_label}: {dose:g} ng/mL LPS")
            if separate_panels:
                unit_label = f", {concentration_unit}" if concentration_unit else ""
                right.set_ylabel(f"{cytokine} Concentration{unit_label}")
            else:
                right.set_ylabel(f"{cytokine} observed concentration")
            ax.grid(axis="y", alpha=0.25, linestyle="--")
            handles = [
                Line2D([], [], linestyle="none", marker="o", color=condition_color(dose, time),
                       label=f"{time:g}hr {dose:g}ng")
                for time in ticks
            ]
            handles.extend([
                Line2D([], [], marker="^", linestyle="--", color=color,
                       label="Morphology reps +/- SEM (left); dashed trend"
                       if "SEM" in morphology
                       else "Morphology reps (left); dashed trend"),
                Line2D([], [], marker="o", linestyle="-", color=color, label="Cytokine bio rep (right)"),
            ])
            main_handle_count = len(handles)
            if inset_has_points:
                inset.scatter(x, y, color=color, s=35)
                if label_inset_points:
                    labels = ([f"{time:g}h / R{rep}" for time, rep in zip(times, group["Biological_Rep"])]
                              if paired else [f"{time:g}h" for time in times])
                    for xv, yv, label in zip(x, y, labels):
                        inset.annotate(label, (xv, yv), xytext=(4, 4),
                                       textcoords="offset points", fontsize=7)
            else:
                inset.set(xticks=[], yticks=[])
            if row["Status"] == "ok":
                grid = np.linspace(x.min(), x.max(), 100)
                inset.plot(grid, row[intercept_key] + row[slope_key] * grid, color=color)
            if control_for_time:
                annotation = (f"Partial R² = {row[r2_key]:.3f}" if row["Status"] == "ok"
                              else "Partial R² unavailable")
                if paired:
                    legend_label = (f"Controls for categorical time and replicate\n"
                                    f"{row['n_pairs']} pairs across {row['n_biological_reps']} biological replicates")
                else:
                    legend_label = f"Controls for linear exposure time\nn = {len(group)} conditions"
                if pd.notna(row["Residual_df"]):
                    legend_label += f"; residual df = {row['Residual_df']}"
                if row["Status"] == "ok":
                    legend_label += f"\nPartial r = {row[r_key]:.3f}"
                    if paired:
                        legend_label += f"; slope = {row[slope_key]:.3g}"
                else:
                    legend_label += f"\n{row['Status']}"
                handles.append(Line2D([], [], linestyle="none", label=legend_label))
                if paired:
                    for missing in missing_pairs:
                        subset = missing.loc[missing["Cytokine"].eq(cytokine)
                                             & missing["Metric"].eq(metric) & missing["LPS_ng_mL"].eq(dose)]
                        for item in subset.itertuples(index=False):
                            label = f"Missing {item.Missing_Assay.lower()}: {item.Time_hr:g} h, replicate {item.Biological_Rep}"
                            handles.append(Line2D([], [], linestyle="none", label=label))
                if paired:
                    inset.set(xlabel=f"log₂({cytokine} concentration) residual",
                              ylabel=f"{transform_label} residual",
                              title="Adjusted biological-replicate pairs")
                else:
                    inset.set(xlabel=f"{metric_label} residual vs. time",
                              ylabel=f"{cytokine} residual vs. time",
                              title="Time-adjusted condition means")
                inset.text(0.04, 0.96, annotation, transform=inset.transAxes, fontsize=8,
                           va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
            elif descriptive_only:
                annotation = (f"Unadjusted R² = {row[r2_key]:.3f}" if row["Status"] == "ok"
                              else "Unadjusted R² unavailable")
                inset.text(0.04, 0.96, annotation, transform=inset.transAxes, fontsize=8,
                           va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
                summary = f"{len(group)} matched time-point means"
                if row["Status"] == "ok":
                    summary = "Unadjusted linear fit\n" + summary
                    summary += f"\nUnadjusted Pearson r = {row[r_key]:.3f}"
                else:
                    summary += f"\n{row['Status']}"
                handles.append(Line2D([], [], color=color,
                                      linestyle="-" if row["Status"] == "ok" else "none", label=summary))
            elif row["Status"] == "ok":
                annotation = (
                    f"r = {row['Pearson_r']:.3f}; R\u00b2 = {row['R_squared']:.3f}\n"
                    f"slope = {row['Slope']:.3g}; intercept = {row['Intercept']:.3g}\n"
                    f"n = {len(group)} conditions"
                )
            else:
                annotation = f"{row['Status']}\nn = {len(group)} conditions"
            if not control_for_time:
                inset.set(xlabel=f"{metric_label} condition mean",
                          ylabel=f"{cytokine} condition mean",
                          title="Descriptive condition means" if descriptive_only else "Exploratory condition-mean regression")
                if not descriptive_only:
                    inset.text(0, -0.32, annotation, transform=inset.transAxes, fontsize=8, va="top")
            caption = ("Biological-replicate pairs within dose; adjusted for time and replicate. No significance test."
                       if paired else "Matched condition means across time; assays are not paired by replicate. No significance test.")
            if correlation_fig is not None:
                inset.set_title(panel_title, fontsize=9)
            else:
                fig.legend(handles=handles, fontsize=8, loc="upper left",
                           bbox_to_anchor=(0.83, 0.88), ncol=1)
                inset.set_title(inset.get_title(), fontsize=9)
                fig.text(0.5, 0.04, caption, ha="center", fontsize=9)
            inset.tick_params(labelsize=8)
            if save_results:
                safe = re.sub(r"[^\w-]+", "_", f"{cytokine}__{metric}").strip("_")
                suffix = "partial_correlation" if control_for_time else "correlation"
                if paired:
                    suffix = "paired_partial_correlation"
                elif descriptive_only: 
                    suffix = "descriptive_cross_plot"
                scale_slug = transform_slug if paired else morphology_scale
                stem = f"{len(statistics):02d}_{safe}__{dose:g}ng__{scale_slug}_{suffix}"
                if correlation_fig is None:
                    path = output / f"{stem}.png"
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    saved_paths.append(str(path))
                else:
                    folder_name = re.sub(r"[^\w -]", "_", readout).strip() or "Metric"
                    folder = output / folder_name / f"{dose:g} ng"
                    folder.mkdir(parents=True, exist_ok=True)
                    for panel, panel_fig in (("Main", fig), ("Correlation", correlation_fig)):
                        panel_dir = folder / panel
                        panel_dir.mkdir(exist_ok=True)
                        path = panel_dir / f"{stem}__{panel.lower()}.png"
                        panel_fig.savefig(path, dpi=300, bbox_inches="tight")
                        saved_paths.append(str(path))
                    main_legend = panel_legends.setdefault(folder / "Main", {})
                    for handle in handles[:main_handle_count]:
                        main_legend.setdefault(handle.get_label(), handle)
                    correlation_handles = handles[main_handle_count:]
                    if correlation_handles:
                        notes = "\n".join(handle.get_label() for handle in correlation_handles)
                        panel_legends.setdefault(folder / "Correlation", {})[
                            f"{cytokine}\n{notes}"
                        ] = correlation_handles[0]
            if display_results:
                plt.show()
        finally:
            if correlation_fig is not None:
                plt.close(correlation_fig)
            plt.close(fig)
    statistics = pd.DataFrame(statistics)
    if save_results:
        for panel_dir, entries in panel_legends.items():
            legend_fig = plt.figure()
            try:
                legend_fig.legend(list(entries.values()), list(entries), loc="center",
                                  fontsize=9, labelspacing=0.9)
                legend_fig.savefig(panel_dir / "Legend.png", dpi=300, bbox_inches="tight")
            finally:
                plt.close(legend_fig)
        workbook = ("Cytokine_Morphology_Partial_Correlations.xlsx" if control_for_time
                    else "Cytokine_Morphology_Correlations.xlsx")
        if paired:
            workbook = "Cytokine_Morphology_Paired_Partial_Correlations.xlsx"
        elif descriptive_only:
            workbook = "Cytokine_Morphology_Descriptive_Cross_Plots.xlsx"
        with pd.ExcelWriter(output / workbook) as writer:
            matched.to_excel(writer, sheet_name="Matched Replicate Pairs" if paired else "Matched Condition Means", index=False)
            if missing_pairs:
                pd.concat(missing_pairs, ignore_index=True).to_excel(writer, sheet_name="Unmatched Replicates", index=False)
            sheet = "Partial Correlations" if control_for_time else "Exploratory Regressions"
            statistics.to_excel(writer, sheet_name=sheet, index=False)
        print(f"Saved {len(saved_paths)} correlation-inset plots and workbook to {output}.")
    return matched, statistics, saved_paths


def read_cell_area_points(area_csv):
    """Read paired area points with the legacy area-specific column names."""
    return read_paired_morphology_points(area_csv, "Area (µm²)").rename(columns={
        "Morphology_Transformed": "Area_Log_Adjusted",
        "Morphology_Measurements": "Area_Measurements",
        "Morphology_Wells": "Area_Wells",
    })


def read_paired_morphology_points(csv_path, metric):
    """Keep raw means for time courses and equal-well summaries for insets."""
    import numpy as np
    import pandas as pd

    _, _, offset = _PAIRED_METRICS[metric]
    cells = pd.read_csv(csv_path, dtype={"Biological_Rep": str})
    labels = cells["Treatment"].str.extract(r"^(\d+)hr_(5|100)ng$")
    eligible = labels[0].notna()
    cells = cells.loc[eligible, ["Biological_Rep", "Well", "Results"]].copy()
    if cells[["Biological_Rep", "Well"]].isna().any().any():
        raise ValueError(f"{metric} CSV has missing biological replicate or well IDs.")
    cells["Results"] = pd.to_numeric(cells["Results"], errors="raise")
    if np.isinf(cells["Results"]).any():
        raise ValueError(f"{metric} CSV contains infinite measurements.")
    if cells["Results"].lt(0).any():
        raise ValueError(f"{metric} CSV contains negative measurements.")
    cells["Time_hr"] = labels.loc[eligible, 0].astype(int)
    cells["Sample"] = labels.loc[eligible, 0] + "hr " + labels.loc[eligible, 1] + "ng"
    if offset is not None and cells["Results"].add(offset).le(0).any():
        raise ValueError(f"{metric} CSV contains nonpositive values after applying the log offset.")
    cells["Morphology_Transformed"] = cells["Results"] if offset is None else np.log(cells["Results"] + offset)
    keys = ["Sample", "Time_hr", "Biological_Rep"]
    points = cells.groupby(keys).agg(
        Results=("Results", "mean"), Morphology_Measurements=("Results", "count"),
    )
    wells = cells.groupby([*keys, "Well"])["Morphology_Transformed"].mean().dropna()
    summaries = wells.groupby(level=keys).agg(Morphology_Transformed="mean", Morphology_Wells="size")
    return points.join(summaries).reset_index().assign(Metric=metric, Morphology_Scale="raw")


def main():
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description="Generate paired time-adjusted plots for six cytokines and "
                                        "four morphology metrics (area, perimeter, circularity, Sholl AUC).")
    parser.add_argument("--output-dir", type=Path, help="Override the output folder.")
    parser.add_argument("--no-point-labels", action="store_true",
                        help="Hide inset point labels and save to a separate folder by default.")
    parser.add_argument("--separate-panels", action="store_true",
                        help="Save standalone main and correlation PNGs with separate legends, grouped by metric and dose.")
    parser.add_argument("--descriptive", action="store_true",
                        help="Generate unpaired descriptive plots instead of paired time-adjusted plots.")
    parser.add_argument("--replicate-map", nargs=2, action="append", metavar=("CYTOKINE_REP", "MORPHOLOGY_REP"),
                        help="Override the verified default map; repeat for each replicate.")
    args = parser.parse_args()
    if args.descriptive and args.replicate_map:
        parser.error("--descriptive cannot be combined with --replicate-map.")
    if args.replicate_map:
        replicate_map = dict(args.replicate_map)
        if len(replicate_map) != len(args.replicate_map):
            parser.error("Provide each cytokine replicate label only once.")
    elif args.descriptive:
        replicate_map = None
    else:
        replicate_map = {"Rep 1": "1", "Rep 2": "2", "Rep 3": "3"}

    from Cytokine_Morphology_Analysis import Plot_Cytokine_Morphology_Plots
    from Cytokine_Scatter_Analysis import Plot_Cytokine_Scatter_Plots
    import pandas as pd

    desktop = Path.home() / "Desktop"
    workbook = desktop / "261007_Cytokine Reanalysis" / "261118_Compiled_Cytokines_3BioReps.xlsx"
    morphology_folder = (
        desktop / "Super LPS" / "3 BioReps Analysis" / "All BioReps Microglia Morphology FINAL"
    )
    cytokines = ["IL-6", "TNFα", "IL-1β"]
    metrics = ["Area (µm²)"]
    if replicate_map:
        cytokines += ["IL-10", "IL-1α", "Fractalkine"]
        metrics = list(_PAIRED_METRICS)
    mode = "Paired_Time_Adjusted_Expanded" if replicate_map else "Descriptive"
    if args.no_point_labels:
        mode += "_No_Point_Labels"
    if args.separate_panels:
        mode += "_Separate_Panels"
    output_dir = args.output_dir or workbook.parent / "Cytokine_Morphology_Correlations" / mode
    # Rows 194–235 already contain log2 values; the paired model logs its input.
    first_row, last_row = (149, 190) if replicate_map else (194, 235)
    parsed, _, _ = Plot_Cytokine_Scatter_Plots(
        workbook,
        sheet_selections=[
            {"sheet_name": "261018 Updated Analysis", "header_row": 1,
             "data_start_row": first_row, "data_end_row": last_row, "start_col": "F", "obs_conc_offset": 1},
        ],
        include_all_samples=False,
        include_48hr=False,
        run_statistics=False,
        save_results=False,
        display_results=False,
    )
    if replicate_map:
        cytokine_points = parsed
        morphology_points = pd.concat(
            [read_paired_morphology_points(morphology_folder / f"{metric}.csv", metric) for metric in metrics],
            ignore_index=True,
        )
    else:
        cytokine_points, morphology_points, _ = Plot_Cytokine_Morphology_Plots(
            parsed, morphology_folder, cytokines=cytokines, save_results=False, display_results=False,
        )
    Plot_Cytokine_Morphology_Correlation_Plots(
        cytokine_points, morphology_points, cytokines=cytokines, metrics=metrics,
        morphology_scale="raw" if replicate_map else "ln", control_for_time=bool(replicate_map),
        replicate_map=replicate_map, descriptive_only=replicate_map is None, output_dir=output_dir,
        label_inset_points=not args.no_point_labels, separate_panels=args.separate_panels,
        concentration_unit="normalized" if replicate_map else None,
    )


if __name__ == "__main__":
    main()
