"""Descriptive dual-axis cytokine and morphology time plots."""


def cytokine_source_suffix(cytokine_data):
    """Use original parser selections, or retained provenance for older inputs."""
    import re

    selections = cytokine_data.attrs.get("source_selections")
    if not selections and {"Sheet", "Excel_Row"}.issubset(cytokine_data.columns):
        selections = [
            {"sheet_name": sheet, "data_start_row": int(rows.min()),
             "data_end_row": int(rows.max())}
            for sheet, rows in cytokine_data.groupby("Sheet", sort=False)["Excel_Row"]
        ]
        print("Output naming uses retained Excel row bounds; rerun the scatter parser for original selection bounds.")
    if not selections:
        print("No cytokine sheet/row provenance available; output folder name is unchanged. Rerun the scatter parser to include source names.")
        return ""
    return "".join(
        "__" + (re.sub(r"[^\w-]+", "_", selection["sheet_name"]).strip("_") or "sheet")
        + f"__rows{selection['data_start_row']}-{selection['data_end_row']}"
        for selection in selections
    )


def Plot_Cytokine_Morphology_Plots(
    cytokine_data,
    morphology_folder,
    *,
    cytokines=None,
    samples=None,
    treatment_map=None,
    morphology_timepoints=None,
    pd_concentration=None,
    output_dir=None,
    show_mean=True,
    save_results=True,
    display_results=False,
):
    """Create dose-specific and combined-dose PNGs per cytokine/morphology CSV.

    Set pd_concentration="2mM" for categorical PD 100 ng/mL plots using the
    R batch folder containing MM_Outputs_*/Combined_Analysis_Data.csv.
    This mode uses confirmed 2 mM cytokine labels, combines the equivalent
    24hrPD_100ng/24hrPD_100ng24hr labels within replicate, and uses 6hr Ctrl
    as the explicitly approved cytokine control (not a time-matched control).
    Groups without numeric data in both assays are excluded per cytokine.
    There are no 5 mM cytokine measurements: those plots are not supported.
    PD mode retains raw/ln morphology, well-based SEM, and unpaired assay
    replicates; it does not draw time trends or perform statistical tests.

    cytokine_data is the parsed-observation DataFrame returned by
    Plot_Cytokine_Scatter_Plots; use include_all_samples=True there to retain
    all eligible times. Its Time_hr values determine cytokine placement.
    morphology_timepoints=None preserves cytokine-matched morphology times.
    A nonempty list of hours chosen from [6, 24, 48, 72, 96] independently
    selects morphology times, including those absent from cytokine data.
    Plain morphology Treatment names provide those times and 5/100 ng/mL doses;
    only doses present in the selected cytokine data are included.
    The x-axis extends to cover both assays; cytokine data are never extended.
    Missing requested morphology times are reported per CSV.
    Required columns: Sample, Rep, Cytokine, Obs_Conc, Time_hr.

    CSVs require Treatment, Biological_Rep, Well, Results. Images/rows are
    averaged within wells, then wells are averaged within each biological
    replicate. Each morphology point shows that replicate's mean +/- SEM across
    wells (sample SD / sqrt(number of wells)). SEM is unavailable for one well;
    its mean remains visible without an error bar and is reported explicitly.
    Morphology trends give each biological replicate equal weight.
    Both raw and natural-log morphology sets are generated. For the ln set,
    each well mean is transformed before biological-replicate means and SEM
    are calculated. All matched well means must be positive; no offset is used.
    Cytokine values are never transformed. Raw filenames are preserved; log
    filenames add __morphology_ln. Returned morphology rows include
    Morphology_Scale ("raw" or "ln") to distinguish their Results and SEM units.
    Saving also creates horizontal cytokine compilations for ln Area, ln
    Perimeter, ln AUC, and raw Circularity only, using 5 ng/mL, 100 ng/mL,
    and combined-dose views in requested cytokine order. Metric names match
    the CSV stems exactly; convex-hull Area/Perimeter are not included.
    Compilations preserve the original panels and legends at their full pixel
    resolution. Compilation paths are appended to the returned PNG paths.
    Cytokine technical repeats are averaged within Sample/Rep.
    Replicate IDs are NOT paired between assays.

    Matching ignores case, spaces and underscores only. Optional treatment_map
    maps exact morphology Treatment names to exact cytokine Sample names.
    Ambiguous automatic matches raise ValueError. Unmatched groups and missing
    values are reported, never imputed. Only plain 5/100 ng/mL LPS samples
    are included, with a separate timecourse figure for each dose and a third
    overlay when both doses have numeric data in both assays. The overlay keeps
    dose-specific means, SEM and trend lines separate. Controls/NP
    groups and unsupported doses are excluded and reported. Cytokines remain on
    the right; matching morphology groups appear on the left. A CSV with no
    numeric matching morphology data raises ValueError.

    Circles/diamonds are cytokine replicates/means on the right; triangles/squares
    are morphology biological replicates/means on the left. Colors identify sample names,
    matching the scatter function's include_all_samples=True presentation.
    Solid cytokine and dashed morphology PCHIP curves (or straight segments for
    two times) use ONLY plain 5/100 ng/mL LPS groups, never controls/NP groups.
    Sparse groups still appear as points. Both axes retain their original units.
    This is descriptive: no cross-assay correlation, tests, or pairing.

    Only direct *.csv children are read, not nested analysis-output folders.
    With source provenance, outputs default to a Cytokine_Morphology_Plots
    subfolder of morphology_folder suffixed with sheet names and selected
    Excel row ranges. PD mode uses PD_Cytokine_Morphology_Plots as its prefix.
    Rerun the scatter parser to preserve original selection bounds (including
    excluded samples); older Sheet/Excel_Row inputs use retained row bounds.
    Without provenance, the legacy location is kept and reported.
    An explicit output_dir is used unchanged. Returns cytokine replicate
    points, morphology biological-replicate points (including Biological_Rep,
    SEM, Number_Wells, Number_Images, Metric and Source_CSV), and PNG
    paths. Neither input data nor source CSVs are modified.
    """
    if pd_concentration is not None:
        if treatment_map is not None or morphology_timepoints is not None:
            raise ValueError("PD mode uses fixed treatment matching, not treatment_map/timepoints.")
        return _plot_pd_cytokine_morphology(
            cytokine_data, morphology_folder, pd_concentration=pd_concentration,
            cytokines=cytokines, samples=samples, output_dir=output_dir,
            show_mean=show_mean, save_results=save_results,
            display_results=display_results,
        )

    from pathlib import Path
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from scipy.interpolate import PchipInterpolator

    def name_list(values, label):
        if values is None:
            return None
        if not isinstance(values, list) or not values:
            raise ValueError(f"{label} must be a nonempty list, or None.")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{label} must contain nonempty names.")
        values = [value.strip() for value in values]
        if len(set(values)) != len(values):
            raise ValueError(f"{label} contains duplicate names.")
        return values

    def numeric_column(frame, column, source):
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any() or np.isinf(numeric).any():
            raise ValueError(f"Invalid/nonfinite {column} in {source}.")
        return numeric

    cytokines = name_list(cytokines, "cytokines")
    samples = name_list(samples, "samples")
    if morphology_timepoints is not None:
        allowed_times = {6, 24, 48, 72, 96}
        if (
            not isinstance(morphology_timepoints, list)
            or not morphology_timepoints
            or any(
                isinstance(time, bool) or not isinstance(time, (int, float))
                or time not in allowed_times
                for time in morphology_timepoints
            )
            or len(set(morphology_timepoints)) != len(morphology_timepoints)
        ):
            raise ValueError(
                "morphology_timepoints must be a nonempty list of unique hours "
                "chosen from [6, 24, 48, 72, 96], or None."
            )
    if not isinstance(cytokine_data, pd.DataFrame):
        raise ValueError("cytokine_data must be the parsed cytokine DataFrame.")
    required = {"Sample", "Rep", "Cytokine", "Obs_Conc", "Time_hr"}
    if not required.issubset(cytokine_data.columns):
        raise ValueError(f"Missing cytokine columns: {sorted(required - set(cytokine_data.columns))}")
    data = cytokine_data.copy()
    if data.empty:
        raise ValueError("cytokine_data is empty.")
    for column in ("Sample", "Rep", "Cytokine"):
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing cytokine {column}.")
        data[column] = data[column].astype(str).str.strip()
    for column, requested in (("Sample", samples), ("Cytokine", cytokines)):
        available = sorted(data[column].unique())
        if requested is not None:
            unknown = [name for name in requested if name not in available]
            if unknown:
                raise ValueError(f"Unknown {column}: {unknown}. Available: {available}")
            data = data.loc[data[column].isin(requested)].copy()
    for column in ("Obs_Conc", "Time_hr"):
        data[column] = numeric_column(data, column, "cytokine_data")
    if data["Time_hr"].isna().any() or data["Time_hr"].lt(0).any():
        raise ValueError("Each cytokine observation needs a nonnegative Time_hr.")
    if data.groupby("Sample")["Time_hr"].nunique().gt(1).any():
        raise ValueError("Each sample must have exactly one Time_hr.")
    sample_names = samples if samples is not None else list(dict.fromkeys(data["Sample"]))
    cytokine_names = cytokines if cytokines is not None else list(dict.fromkeys(data["Cytokine"]))
    trend_colors = {5: "#E65400", 100: "#0072B2"}
    plain_pattern = r"^\s*\d+\s*(?:hr|h)\s*[_ ]+([\d.]+)\s*ng(?:/mL)?(?:\s*LPS)?\s*$"
    plain_doses = {}
    for name in sample_names:
        match = re.fullmatch(plain_pattern, name, re.I)
        if match and float(match.group(1)) in trend_colors:
            plain_doses[name] = float(match.group(1))
    excluded = [name for name in sample_names if name not in plain_doses]
    if excluded:
        print(f"Excluded samples from dose-separated plain-LPS plots: {excluded}")
    sample_names = [name for name in sample_names if name in plain_doses]
    if not sample_names:
        raise ValueError("No plain 5/100 ng/mL LPS samples remain for dose-separated plots.")
    data = data.loc[data["Sample"].isin(sample_names)].copy()
    sample_times = data.groupby("Sample")["Time_hr"].first().to_dict()
    cytokine_sample_names = sample_names.copy()
    cytokine_doses = set(plain_doses.values())
    missing_count = int(data["Obs_Conc"].isna().sum())
    print(f"Cytokine missing concentrations omitted: {missing_count}.")
    cyto_points = data.dropna(subset=["Obs_Conc"]).groupby(
        ["Cytokine", "Sample", "Rep", "Time_hr"], sort=False, as_index=False
    ).agg(Obs_Conc=("Obs_Conc", "mean"), TechnicalRows=("Obs_Conc", "size"))
    if cyto_points.empty:
        raise ValueError("No numeric cytokine concentrations remain.")
    for cytokine in cytokine_names:
        if not cyto_points["Cytokine"].eq(cytokine).any():
            raise ValueError(f"No numeric data for selected cytokine {cytokine!r}.")

    folder = Path(morphology_folder)
    if not folder.is_dir():
        raise ValueError(f"Morphology folder does not exist: {folder}")
    csv_paths = sorted(folder.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No top-level CSV files in {folder}.")
    mapping = {} if treatment_map is None else treatment_map
    if not isinstance(mapping, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in mapping.items()
    ):
        raise ValueError("treatment_map must map morphology treatment names to cytokine sample names.")
    if set(mapping.values()) - set(sample_names):
        raise ValueError("treatment_map targets must be included cytokine sample names.")

    def normalize(name):
        return re.sub(r"[\s_]+", "", name).casefold()

    normalized = {}
    for name in sample_names:
        normalized.setdefault(normalize(name), []).append(name)
    morphology_frames = []
    log_morphology_frames = []
    all_treatments = set()
    for csv_path in csv_paths:
        raw = pd.read_csv(csv_path)
        required = {"Treatment", "Biological_Rep", "Well", "Results"}
        if not required.issubset(raw.columns):
            raise ValueError(f"{csv_path.name}: missing columns {sorted(required - set(raw.columns))}")
        if raw.empty:
            raise ValueError(f"{csv_path.name} is empty.")
        for column in ("Treatment", "Biological_Rep", "Well"):
            if raw[column].isna().any() or raw[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{csv_path.name}: missing {column}.")
            raw[column] = raw[column].astype(str).str.strip()
        all_treatments.update(raw["Treatment"].unique())
        matches = {}
        for treatment in raw["Treatment"].unique():
            candidate = None
            if treatment in mapping:
                candidate = mapping[treatment]
            else:
                candidates = normalized.get(normalize(treatment), [])
                if len(candidates) > 1:
                    raise ValueError(f"Ambiguous treatment {treatment!r}; supply treatment_map.")
                if candidates:
                    candidate = candidates[0]
                elif morphology_timepoints is not None:
                    match = re.fullmatch(plain_pattern, treatment, re.I)
                    if match and float(match.group(1)) in cytokine_doses:
                        time = int(re.match(r"^\s*(\d+)", treatment).group(1))
                        if time in morphology_timepoints:
                            dose = float(match.group(1))
                            candidate = f"{time}hr {dose:g}ng"
                            if candidate not in sample_times:
                                sample_times[candidate] = time
                                plain_doses[candidate] = dose
                                sample_names.append(candidate)
                                normalized.setdefault(normalize(candidate), []).append(candidate)
            if candidate is not None and (
                morphology_timepoints is None
                or sample_times[candidate] in morphology_timepoints
            ):
                matches[treatment] = candidate
        raw["Sample"] = raw["Treatment"].map(matches)
        unmatched = sorted(set(raw["Treatment"]) - set(matches))
        absent = [name for name in cytokine_sample_names if name not in set(matches.values())]
        print(f"{csv_path.name}: unmatched morphology treatments excluded: {unmatched}")
        print(f"{csv_path.name}: cytokine samples without morphology: {absent}")
        raw = raw.loc[raw["Sample"].notna()].copy()
        raw["Results"] = numeric_column(raw, "Results", csv_path.name)
        print(f"{csv_path.name}: missing matching morphology values omitted: {int(raw['Results'].isna().sum())}.")
        raw = raw.dropna(subset=["Results"])
        if raw.empty:
            raise ValueError(f"{csv_path.name}: no numeric morphology matches the included samples.")
        wells = raw.groupby(
            ["Sample", "Biological_Rep", "Well"], sort=False, as_index=False
        ).agg(Results=("Results", "mean"), Number_Images=("Results", "size"))
        nonpositive = wells.loc[
            wells["Results"].le(0), ["Sample", "Biological_Rep", "Well", "Results"]
        ]
        if not nonpositive.empty:
            raise ValueError(
                f"{csv_path.name}: ln morphology requires positive well means; "
                f"nonpositive wells: {nonpositive.to_dict('records')}"
            )
        for scale, frames in (("raw", morphology_frames), ("ln", log_morphology_frames)):
            values = wells.copy()
            if scale == "ln":
                values["Results"] = np.log(values["Results"])
            bio = values.groupby(
                ["Sample", "Biological_Rep"], sort=False, as_index=False
            ).agg(
                Results=("Results", "mean"),
                SEM=("Results", "sem"),
                Number_Wells=("Well", "size"),
                Number_Images=("Number_Images", "sum"),
            )
            bio["Time_hr"] = bio["Sample"].map(sample_times)
            bio["Metric"] = csv_path.stem
            bio["Source_CSV"] = str(csv_path.resolve())
            bio["Morphology_Scale"] = scale
            frames.append(bio)
        if morphology_timepoints is not None:
            missing_times = sorted(set(morphology_timepoints) - set(bio["Time_hr"]))
            print(f"{csv_path.name}: requested morphology hours without numeric data: {missing_times}")
        single_well = bio.loc[bio["Number_Wells"].eq(1), ["Sample", "Biological_Rep"]]
        if not single_well.empty:
            print(
                f"{csv_path.name}: SEM unavailable for single-well biological replicates: "
                f"{single_well.to_dict('records')}"
            )
    unknown_treatments = set(mapping) - all_treatments
    if unknown_treatments:
        raise ValueError(f"treatment_map contains unknown treatments: {sorted(unknown_treatments)}")
    morphology_points = pd.concat(morphology_frames + log_morphology_frames, ignore_index=True)

    if output_dir is not None:
        output = Path(output_dir)
    else:
        source_suffix = cytokine_source_suffix(cytokine_data)
        output = folder / f"Cytokine_Morphology_Plots{source_suffix}" if source_suffix else folder
    if save_results:
        output.mkdir(parents=True, exist_ok=True)
    sample_colors = dict(zip(
        sample_names, plt.get_cmap("tab20")(np.linspace(0, 1, len(sample_names)))
    ))
    def draw_trends(axis, frame, value_column, linestyle):
        for dose, color in trend_colors.items():
            names = [name for name, value in plain_doses.items() if value == dose]
            sub = frame.loc[frame["Sample"].isin(names)]
            if value_column == "Results":
                means = sub.groupby("Time_hr")[value_column].mean().sort_index()
            else:
                means = sub.groupby(["Sample", "Time_hr"])[value_column].mean()
                means = means.groupby("Time_hr").mean().sort_index()
            if len(means) >= 3:
                grid = np.linspace(means.index.min(), means.index.max(), 200)
                curve = PchipInterpolator(means.index, means.values, extrapolate=False)
                axis.plot(grid, curve(grid), color=color, linestyle=linestyle, linewidth=1.8)
            elif len(means) == 2:
                axis.plot(means.index, means.values, color=color, linestyle=linestyle, linewidth=1.8)

    rng = np.random.default_rng(42)
    saved_paths = []
    compilation_panels = {}
    compilation_scales = {
        "Area (µm²)": "ln",
        "Perimeter (µm)": "ln",
        "AUC": "ln",
        "Circularity": "raw",
    }
    plot_frames = [
        *enumerate(morphology_frames, start=1),
        *enumerate(log_morphology_frames, start=1),
    ]
    for metric_index, bio in plot_frames:
        metric = bio["Metric"].iloc[0]
        scale = bio["Morphology_Scale"].iloc[0]
        metric_label = metric if scale == "raw" else f"ln({metric})"
        if scale == "ln" and metric_index == 1:
            rng = np.random.default_rng(42)
        for cytokine_index, cytokine, plot_dose in (
            (index, name, dose)
            for index, name in enumerate(cytokine_names, start=1)
            for dose in [*[d for d in trend_colors if d in plain_doses.values()], None]
        ):
            dose_samples = [
                name for name in sample_names
                if plot_dose is None or plain_doses[name] == plot_dose
            ]
            cyto = cyto_points.loc[
                cyto_points["Cytokine"].eq(cytokine) & cyto_points["Sample"].isin(dose_samples)
            ]
            morphology = bio.loc[bio["Sample"].isin(dose_samples)]
            if plot_dose is None:
                available_cytokine_doses = set(cyto["Sample"].map(plain_doses))
                available_morphology_doses = set(morphology["Sample"].map(plain_doses))
                if not set(trend_colors).issubset(
                    available_cytokine_doses & available_morphology_doses
                ):
                    print(
                        f"{metric}, {cytokine}: combined-dose plot skipped; "
                        "both doses require numeric data in both assays."
                    )
                    continue
            dose_label = "5 and 100 ng/mL" if plot_dose is None else f"{plot_dose} ng/mL"
            if cyto.empty or morphology.empty:
                print(
                    f"{metric}, {cytokine}, {dose_label}: no numeric "
                    f"{'cytokine' if cyto.empty else 'morphology'} data; no plot generated."
                )
                continue
            fig, left = plt.subplots(figsize=(12, 6))
            try:
                right = left.twinx()
                for axis, frame, value_column, marker, mean_marker in (
                    (right, cyto, "Obs_Conc", "o", "D"),
                    (left, morphology, "Results", "^", "s"),
                ):
                    for sample in dose_samples:
                        points = frame.loc[frame["Sample"].eq(sample)]
                        if points.empty:
                            continue
                        x_positions = points["Time_hr"] + rng.uniform(-0.6, 0.6, len(points))
                        axis.scatter(
                            x_positions,
                            points[value_column], marker=marker, color=sample_colors[sample],
                            s=55, edgecolors="white", linewidths=0.6, zorder=3,
                        )
                        if value_column == "Results":
                            has_sem = points["SEM"].notna()
                            if has_sem.any():
                                axis.errorbar(
                                    x_positions.loc[has_sem], points.loc[has_sem, "Results"],
                                    yerr=points.loc[has_sem, "SEM"], fmt="none",
                                    ecolor=sample_colors[sample], capsize=3,
                                    elinewidth=1.2, zorder=2,
                                )
                        if show_mean:
                            axis.scatter(
                                [sample_times[sample]], [points[value_column].mean()],
                                marker=mean_marker, color=sample_colors[sample],
                                s=65, edgecolors="black", zorder=4,
                            )
                draw_trends(right, cyto, "Obs_Conc", "-")
                draw_trends(left, morphology, "Results", "--")
                ticks = sorted({sample_times[sample] for sample in dose_samples})
                left.set_xticks(ticks)
                left.set_xticklabels([f"{time:g} hr" for time in ticks])
                left.set_xlim(min(ticks) - 3, max(ticks) + 3)
                left.set(xlabel="Exposure time", ylabel=metric_label,
                         title=f"{cytokine} and {metric_label} — {dose_label} LPS")
                right.set_ylabel(f"{cytokine} observed concentration")
                left.grid(axis="y", linestyle="--", alpha=0.25)
                handles = []
                for sample in dose_samples:
                    unavailable = []
                    if not cyto["Sample"].eq(sample).any():
                        unavailable.append("no cytokine")
                    if not morphology["Sample"].eq(sample).any():
                        unavailable.append("no morphology")
                    suffix = f" ({', '.join(unavailable)})" if unavailable else ""
                    handles.append(Line2D(
                        [0], [0], linestyle="none", marker="o", color=sample_colors[sample],
                        label=sample + suffix,
                    ))
                handles.extend([
                    Line2D([0], [0], linestyle="none", marker="o", color="black", label="Cytokine replicate (right)"),
                    Line2D([0], [0], linestyle="none", marker="^", color="black", label="Morphology bio rep ± SEM across wells (left)"),
                ])
                if show_mean:
                    handles.extend([
                        Line2D([0], [0], linestyle="none", marker="D", color="black", label="Cytokine mean"),
                        Line2D([0], [0], linestyle="none", marker="s", color="black", label="Morphology mean"),
                    ])
                for dose, color in trend_colors.items():
                    if plot_dose is None or dose == plot_dose:
                        handles.extend([
                            Line2D([0], [0], color=color, label=f"{dose} ng/mL cytokine trend"),
                            Line2D([0], [0], color=color, linestyle="--", label=f"{dose} ng/mL morphology trend"),
                        ])
                fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.79, 0.98), fontsize=8)
                fig.subplots_adjust(right=0.70, bottom=0.13)
                if save_results:
                    safe_metric = re.sub(r"[^\w-]+", "_", metric).strip("_")
                    safe_cytokine = re.sub(r"[^\w-]+", "_", cytokine).strip("_")
                    dose_suffix = "5ng_100ng" if plot_dose is None else f"{plot_dose}ng"
                    scale_suffix = "" if scale == "raw" else "__morphology_ln"
                    path = output / f"DualAxis_{metric_index:02d}_{safe_metric}__{cytokine_index:02d}_{safe_cytokine}__{dose_suffix}{scale_suffix}.png"
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    saved_paths.append(str(path))
                    if compilation_scales.get(metric) == scale:
                        compilation_path = output / (
                            f"Compilation_{metric_index:02d}_{safe_metric}"
                            f"__{dose_suffix}__morphology_{scale}.png"
                        )
                        compilation_panels.setdefault(compilation_path, []).append(path)
                if display_results:
                    plt.show()
            finally:
                plt.close(fig)
    if save_results:
        from PIL import Image

        for compilation_path, panels in compilation_panels.items():
            dimensions = []
            for panel in panels:
                with Image.open(panel) as image:
                    dimensions.append(image.size)
            width = sum(size[0] for size in dimensions)
            height = max(size[1] for size in dimensions)
            with Image.new("RGB", (width, height), "white") as compilation:
                x_offset = 0
                for panel, (panel_width, _) in zip(panels, dimensions):
                    with Image.open(panel) as image:
                        compilation.paste(image, (x_offset, 0))
                    x_offset += panel_width
                compilation.save(compilation_path, dpi=(300, 300))
            saved_paths.append(str(compilation_path))
        print(
            f"Saved {len(saved_paths)} PNGs including {len(compilation_panels)} "
            f"selected-morphology compilations to {output.resolve()}."
        )
    return cyto_points, morphology_points, saved_paths


def _plot_pd_cytokine_morphology(
    cytokine_data, morphology_folder, *, pd_concentration, cytokines, samples,
    output_dir, show_mean, save_results, display_results,
):
    """Plot confirmed 2 mM PD groups without inferring exposure times from labels."""
    from pathlib import Path
    import re

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd

    if pd_concentration != "2mM":
        raise ValueError("PD cytokine measurements are 2 mM only; cannot match 5 mM morphology.")
    mapping = {
        "6hr Ctrl": "Tri_Ctrl",
        "2mM PD": "PD_0LPS",
        "24hr 100ng": "100ng_LPS",
        "24hrPD_100ng": "24hrPD_100ng",
        "24hrPD_100ng24hr": "24hrPD_100ng",
        "1hrPD_100ng": "1hrPD_100ng",
    }
    order = list(dict.fromkeys(mapping.values()))
    required = {"Sample", "Rep", "Cytokine", "Obs_Conc"}
    if not isinstance(cytokine_data, pd.DataFrame) or not required.issubset(cytokine_data.columns):
        raise ValueError(f"PD cytokine data require columns {sorted(required)}.")
    data = cytokine_data.copy()
    for column in ("Sample", "Rep", "Cytokine"):
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing cytokine {column}.")
        data[column] = data[column].astype(str).str.strip()
    for column, requested in (("Sample", samples), ("Cytokine", cytokines)):
        if requested is not None:
            if (
                not isinstance(requested, list) or not requested
                or any(not isinstance(name, str) or not name.strip() for name in requested)
                or len(set(requested)) != len(requested)
            ):
                raise ValueError(f"{column} selection must be a nonempty list of unique names.")
            unknown = set(requested) - set(data[column])
            if unknown:
                raise ValueError(f"Unknown {column}: {sorted(unknown)}")
            data = data.loc[data[column].isin(requested)].copy()
    cytokine_names = cytokines if cytokines is not None else list(dict.fromkeys(data["Cytokine"]))
    data["Treatment"] = data["Sample"].map(mapping)
    print("PD cytokine samples excluded:", sorted(set(data.loc[data["Treatment"].isna(), "Sample"])))
    print("PD control mapping: 6hr Ctrl cytokines vs Tri_Ctrl morphology; endpoints differ.")
    print("No CoPD or 5 mM cytokine data; these dual-axis groups are excluded.")
    data = data.loc[data["Treatment"].notna()].copy()

    def numeric(frame, column, source):
        values = pd.to_numeric(frame[column], errors="coerce")
        if (frame[column].notna() & values.isna()).any() or np.isinf(values).any():
            raise ValueError(f"Invalid/nonfinite {column} in {source}.")
        return values

    data["Obs_Conc"] = numeric(data, "Obs_Conc", "cytokines")
    print(f"Missing PD cytokine concentrations omitted: {data['Obs_Conc'].isna().sum()}.")
    cyto = data.dropna(subset=["Obs_Conc"]).groupby(
        ["Cytokine", "Treatment", "Rep"], sort=False, as_index=False
    ).agg(Obs_Conc=("Obs_Conc", "mean"), TechnicalRows=("Obs_Conc", "size"))
    if cyto.empty:
        raise ValueError("No numeric matching PD cytokine observations.")
    cyto["Sample"] = cyto["Treatment"]
    folder = Path(morphology_folder)
    paths = sorted(folder.glob("MM_Outputs_PD_Morphology_2mM_*/Combined_Analysis_Data.csv"))
    if not paths:
        raise ValueError(f"No 2 mM R Combined_Analysis_Data.csv metric files in {folder}.")
    output = (
        Path(output_dir) if output_dir is not None
        else folder / f"PD_Cytokine_Morphology_Plots{cytokine_source_suffix(cytokine_data)}"
    )
    if save_results:
        output.mkdir(parents=True, exist_ok=True)
    saved = []
    frames = []
    rng = np.random.default_rng(42)
    colors = dict(zip(order, plt.get_cmap("tab10")(np.arange(len(order)))))
    for path in paths:
        raw = pd.read_csv(path)
        needed = {"Treatment", "Biological_Rep", "Well", "Results"}
        if not needed.issubset(raw.columns):
            raise ValueError(f"{path}: missing columns {sorted(needed - set(raw.columns))}")
        for column in ("Treatment", "Biological_Rep", "Well"):
            if raw[column].isna().any() or raw[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{path}: missing {column}.")
            raw[column] = raw[column].astype(str).str.strip()
        if "Source" in raw and not raw.loc[raw["Treatment"].eq("100ng_LPS"), "Source"].eq("LPS_only").all():
            raise ValueError(f"{path}: 100ng_LPS must come from the external LPS-only dataset.")
        print(f"{path.parent.name}: excluded morphology groups {sorted(set(raw['Treatment']) - set(order))}")
        raw = raw.loc[raw["Treatment"].isin(order)].copy()
        raw["Results"] = numeric(raw, "Results", str(path))
        print(f"{path.parent.name}: missing morphology values omitted: {raw['Results'].isna().sum()}.")
        raw = raw.dropna(subset=["Results"])
        wells = raw.groupby(["Treatment", "Biological_Rep", "Well"], as_index=False).agg(
            Results=("Results", "mean"), Number_Images=("Results", "size")
        )
        if wells.empty or wells["Results"].le(0).any():
            raise ValueError(f"{path}: raw/ln plots require positive matching well means.")
        metric = path.parent.name.removeprefix("MM_Outputs_PD_Morphology_2mM_")
        for scale in ("raw", "ln"):
            values = wells.copy()
            if scale == "ln":
                values["Results"] = np.log(values["Results"])
            bio = values.groupby(["Treatment", "Biological_Rep"], as_index=False).agg(
                Results=("Results", "mean"), SEM=("Results", "sem"),
                Number_Wells=("Well", "size"), Number_Images=("Number_Images", "sum")
            )
            single = bio.loc[bio["Number_Wells"].eq(1), ["Treatment", "Biological_Rep"]]
            if not single.empty:
                print(f"{metric}: SEM unavailable for single-well replicates: {single.to_dict('records')}")
            for cytokine in cytokine_names:
                c = cyto.loc[cyto["Cytokine"].eq(cytokine)]
                groups = [name for name in order if name in set(c["Treatment"]) & set(bio["Treatment"])]
                print(f"{metric}, {cytokine}: groups without both assays excluded: {sorted(set(order) - set(groups))}")
                if not groups:
                    raise ValueError(f"{metric}, {cytokine}: no groups with numeric data in both assays.")
                b = bio.loc[bio["Treatment"].isin(groups)].copy()
                b = b.assign(Sample=b["Treatment"], Cytokine=cytokine, Metric=metric,
                             Source_CSV=str(path.resolve()), Morphology_Scale=scale)
                frames.append(b)
                fig, left = plt.subplots(figsize=(12, 6))
                try:
                    right = left.twinx()
                    for axis, points, column, marker, mean_marker, offset in (
                        (left, b, "Results", "^", "s", -0.12),
                        (right, c, "Obs_Conc", "o", "D", 0.12),
                    ):
                        for index, name in enumerate(groups):
                            sub = points.loc[points["Treatment"].eq(name)]
                            x = index + offset + rng.uniform(-0.04, 0.04, len(sub))
                            axis.scatter(x, sub[column], marker=marker, color=colors[name], s=50)
                            if column == "Results":
                                valid = sub["SEM"].notna().to_numpy()
                                axis.errorbar(x[valid], sub.loc[valid, column],
                                              yerr=sub.loc[valid, "SEM"], fmt="none",
                                              ecolor=colors[name], capsize=3)
                            if show_mean:
                                axis.scatter(index + offset, sub[column].mean(),
                                             marker=mean_marker, color=colors[name],
                                             edgecolors="black", s=70)
                    label = metric if scale == "raw" else f"ln({metric})"
                    left.set_xticks(range(len(groups)))
                    left.set_xticklabels(groups, rotation=45, ha="right")
                    left.set(xlabel="Treatment", ylabel=label,
                             title=f"2 mM PD: {cytokine} and {label} (100 ng/mL LPS)")
                    right.set_ylabel(f"{cytokine} observed concentration (input scale)")
                    left.grid(axis="y", alpha=0.25)
                    handles = [
                        Line2D([], [], marker="^", linestyle="none", color="black",
                               label="Morphology bio rep +/- SEM across wells (left)"),
                        Line2D([], [], marker="o", linestyle="none", color="black",
                               label="Cytokine replicate (right)"),
                    ]
                    if show_mean:
                        handles.extend([
                            Line2D([], [], marker="s", linestyle="none", color="black", label="Morphology mean"),
                            Line2D([], [], marker="D", linestyle="none", color="black", label="Cytokine mean"),
                        ])
                    left.legend(handles=handles, fontsize=8, loc="best")
                    fig.text(0.5, 0.01, "Control: 6hr cytokines / PD morphology; descriptive, not paired across assays.",
                             ha="center", fontsize=8)
                    fig.tight_layout(rect=(0, 0.04, 1, 1))
                    if save_results:
                        safe = re.sub(r"[^\w-]+", "_", f"{metric}__{cytokine}").strip("_")
                        target = output / f"PD_2mM_100ng_{safe}__morphology_{scale}.png"
                        fig.savefig(target, dpi=300, bbox_inches="tight")
                        saved.append(str(target))
                    if display_results:
                        plt.show()
                finally:
                    plt.close(fig)
    return cyto, pd.concat(frames, ignore_index=True), saved
