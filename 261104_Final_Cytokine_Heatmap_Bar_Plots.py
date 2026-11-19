"""Standalone cytokine heatmap, raw bar plots, and exploratory statistics.

Requires numpy, pandas, matplotlib, seaborn, scipy, and openpyxl.
Import plot_cytokine_heatmap for use in another script or notebook. Running this
file directly executes the editable example at the bottom. Importing does not
read any workbook or generate plots. No notebook or local helper file is needed.

Returns (heatmap_data, long_df), preserving the notebook interface. PNGs and
statistics workbooks are saved beside the input workbook unless output_dir is set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import stats


def analyze_heatmap_groups(
    observations: pd.DataFrame,
    samples: list[str],
    cytokines: list[str],
    control: str,
    alpha: float,
    censored: set[tuple[str, str | None, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Average technical rows, then run Kruskal-Wallis and Dunn vs control.

    Every included group needs at least two numeric biological replicates.
    Rep IDs identify repeats within groups, not pairs between groups. All
    requested comparisons are exported, including explicit not-tested rows.
    """
    selected = observations.loc[
        observations.Sample.isin(samples) & observations.Cytokine.isin(cytokines)
    ].copy()
    numeric = selected.Obs_Conc.dropna()
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite concentrations cannot be used for cytokine statistics.")
    points = selected.groupby(
        ["Cytokine", "Sample", "Rep"], sort=False, as_index=False, dropna=False
    ).agg(
        Value=("Obs_Conc", "mean"),
        Number_Values=("Obs_Conc", "count"),
        Number_Rows=("Obs_Conc", "size"),
    )
    others = [sample for sample in samples if sample != control]
    overall_rows, pair_rows = [], []
    for cytokine in cytokines:
        metric = points.loc[points.Cytokine.eq(cytokine)]
        groups = {
            sample: metric.loc[
                metric.Sample.eq(sample) & metric.Rep.notna(), "Value"
            ].dropna().to_numpy(dtype=float)
            for sample in samples
        }
        counts = {sample: len(values) for sample, values in groups.items()}
        inadequate = [sample for sample, count in counts.items() if count < 2]
        if len(samples) < 2:
            status = "not tested: at least two groups are required"
        elif metric.Rep.isna().any() or metric.Rep.fillna("").astype(str).str.strip().eq("").any():
            status = "not tested: missing biological replicate IDs"
        elif any(sample in samples and name == cytokine for sample, _, name in censored):
            status = "not tested: censored OOR < observations require an assay-specific handling policy"
        elif inadequate:
            status = f"not tested: fewer than two numeric biological replicates in {inadequate}"
        else:
            status = "tested"

        h_value = p_value = np.nan
        comparisons = []
        if status == "tested":
            pooled = np.concatenate(list(groups.values()))
            if np.ptp(pooled) == 0:
                h_value, p_value = 0.0, 1.0
            else:
                result = stats.kruskal(*groups.values())
                h_value, p_value = float(result.statistic), float(result.pvalue)
            ranks = stats.rankdata(pooled)
            boundaries = np.cumsum([0, *counts.values()])
            mean_ranks = {
                sample: ranks[boundaries[index]:boundaries[index + 1]].mean()
                for index, sample in enumerate(samples)
            }
            _, ties = np.unique(pooled, return_counts=True)
            total = len(pooled)
            variance = total * (total + 1) / 12 - np.sum(ties ** 3 - ties) / (12 * (total - 1))
            for sample in others:
                difference = mean_ranks[sample] - mean_ranks[control]
                se = np.sqrt(max(0.0, variance) * (1 / counts[sample] + 1 / counts[control]))
                z_value = difference / se if se > 0 else 0.0
                comparisons.append((sample, z_value, float(2 * stats.norm.sf(abs(z_value)))))
            adjusted = _holm(np.array([p for _, _, p in comparisons]))
        else:
            adjusted = np.full(len(others), np.nan)

        overall_rows.append({
            "Cytokine": cytokine, "Number_Groups": len(samples),
            "N_Total": sum(counts.values()), "DF": len(samples) - 1,
            "Replicates_Per_Group": "; ".join(f"{sample}: n={count}" for sample, count in counts.items()),
            "H_Statistic": h_value, "P_Value": p_value, "Q_BH": np.nan,
            "Status": status,
        })
        tested_pairs = {sample: (z_value, p) for sample, z_value, p in comparisons}
        for index, sample in enumerate(others):
            z_value, raw_p = tested_pairs.get(sample, (np.nan, np.nan))
            pair_rows.append({
                "Cytokine": cytokine, "Group": sample, "Control": control,
                "N_Group": counts[sample], "N_Control": counts[control],
                "Z_Statistic": z_value, "P_Unadjusted": raw_p,
                "P_Holm": adjusted[index],
                "Significant": bool(np.isfinite(adjusted[index]) and adjusted[index] < alpha),
                "Status": status,
            })
        print(f"{cytokine}: {status}; " + "; ".join(f"{s}: n={n}" for s, n in counts.items()))

    overall = pd.DataFrame(overall_rows)
    valid = overall.P_Value.dropna().sort_values()
    if len(valid):
        adjusted_bh = valid.to_numpy() * len(valid) / np.arange(1, len(valid) + 1)
        overall.loc[valid.index, "Q_BH"] = np.minimum.accumulate(adjusted_bh[::-1])[::-1].clip(0, 1)
    overall["FDR_Family_Size"] = len(valid)
    pairs = pd.DataFrame(pair_rows, columns=[
        "Cytokine", "Group", "Control", "N_Group", "N_Control",
        "Z_Statistic", "P_Unadjusted", "P_Holm", "Significant", "Status",
    ])
    return points, overall, pairs


def _holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    adjusted[order] = np.maximum.accumulate(
        values[order] * np.arange(len(values), 0, -1)
    ).clip(0, 1)
    return adjusted


def export_heatmap_statistics(
    path: Path, observations: pd.DataFrame, points: pd.DataFrame,
    overall: pd.DataFrame, pairs: pd.DataFrame, control: str, alpha: float,
) -> None:
    notes = {
        "Control": control,
        "Alpha": str(alpha),
        "Minimum numeric biological replicates per group": "2; unequal group sizes allowed",
        "Unit of inference": "Technical rows averaged within cytokine/sample/Rep; groups independent, not paired",
        "Missing data": "Omit missing numeric values; do not impute or drop entire treatment groups",
        "Censored data": "Skip cytokine if OOR < values are present; descriptive zero is not used for inference",
        "Omnibus": "Kruskal-Wallis, tie-corrected chi-square approximation; BH FDR across valid selected cytokines",
        "Post-hoc": "Dunn, pooled tie-corrected ranks, two-sided; Holm within cytokine vs control; no omnibus gate",
        "Small sample warning": "Exploratory at n=2-3: asymptotic p-values may be inaccurate and power is limited",
        "Plots": "Raw row-weighted mean +/- SD and observations; tests use biological means, not heatmap z-scores",
    }
    with pd.ExcelWriter(path) as writer:
        overall.to_excel(writer, sheet_name="Kruskal_Wallis", index=False)
        pairs.to_excel(writer, sheet_name="Dunn_vs_Control", index=False)
        points.to_excel(writer, sheet_name="Biological_Replicates", index=False)
        observations.to_excel(writer, sheet_name="Input_Observations", index=False)
        pd.DataFrame(notes.items(), columns=["Setting", "Value"]).to_excel(
            writer, sheet_name="Analysis_Notes", index=False,
        )


def annotate_heatmap_bar(
    ax: Axes, cytokine: str, samples: list[str], overall: pd.DataFrame,
    pairs: pd.DataFrame, control: str,
) -> None:
    """Label skipped tests and Holm-adjusted Dunn significance without clipping."""
    row = overall.loc[overall.Cytokine.eq(cytokine)].iloc[0]
    if row.Status != "tested":
        ax.set_title(f"{cytokine}\n{row.Status}", fontsize=9, wrap=True)
        return
    ax.set_title(
        f"{cytokine}\nKruskal-Wallis p={row.P_Value:.3g}; BH q={row.Q_BH:.3g}\n"
        f"Dunn vs {control} (Holm); exploratory, small n", fontsize=9,
    )
    lower, upper = ax.get_ylim()
    span = upper - lower
    significant = pairs.loc[pairs.Cytokine.eq(cytokine) & pairs.Significant]
    for row in significant.itertuples():
        p_value = row.P_Holm
        stars = "****" if p_value < 0.0001 else "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        ax.text(samples.index(row.Group), upper + 0.03 * span, stars, ha="center", va="bottom")
    if not significant.empty:
        ax.set_ylim(lower, upper + 0.15 * span)


def plot_cytokine_heatmap(
    file_path,
    sheet_name=0,
    sample_col="B",
    rep_col="A",
    header_row=16,
    start_col="F",
    col_step=3,
    obs_conc_offset=1,
    data_start_row=None,
    data_end_row=None,
    sample_order=None,
    preserve_sample_order=True,
    use_log2fc=False,
    scale_mode=None,          # <-- "none", "log2fc", or "zscore_row". Overrides use_log2fc if set.
    ref_sample=None,
    pseudocount=1e-9,
    cluster_rows=True,
    output_dir=None,
    fontsize_row=7,
    fontsize_col=9,
    sheet_selections=None,
    cytokines=None,
    samples=None,
    run_statistics=True,
    control_sample="6hr Ctrl",
    alpha=0.05,
):
    """
    Builds a cytokine x sample heatmap (mean Obs. Conc. across bio reps) directly from
    the SAME wide-format Excel layout used by Plot_Cytokine_Bar_Charts.
    Exports the heatmap and one raw-concentration bar plot per selected cytokine
    as PNGs. Bar plots use the same selected samples and numeric observations,
    with mean +/- SD, individual observation dots, and distinct Rep counts.
    Technical rows retain equal weight, matching the heatmap's raw means.
    Bars are never scaled. By default, raw biological-replicate means enter
    Kruskal-Wallis and Dunn vs control_sample, with at least 2 numeric reps
    per included group (unequal n is allowed). Technical rows are averaged
    within Sample/Rep for tests only; descriptive bar means remain unchanged.
    Holm adjustment covers all treatment-vs-control contrasts per cytokine;
    BH adjusts valid omnibus p-values across the selected cytokines. Stars
    show Holm-adjusted Dunn p-values, without gating on omnibus significance.
    These asymptotic tests are exploratory at n=2-3 and assume independent
    groups, not matched Rep IDs. Censored values/missing IDs skip testing;
    missing numeric observations are omitted, never imputed. An XLSX records
    counts, p-values, and skip reasons. Set run_statistics=False to disable.
    All statistics helpers are included in this standalone module. Raw bar plots
    are saved in the output directory's Raw Bar Plots subfolder. Returns unchanged.

    scale_mode options (overrides use_log2fc if provided):
      - "none"        : raw mean Obs. Conc. values (no transform)
      - "log2fc"      : log2 fold-change vs `ref_sample` (per cytokine row)
      - "zscore_row"  : z-score standardization per cytokine row:
                        z = (value - row_mean) / row_std
                        (equivalent to R's scale() applied row-wise / pheatmap's scale="row")

    If scale_mode is None, falls back to use_log2fc (True -> "log2fc", False -> "none").

    Optional sheet_selections is a nonempty list of dictionaries containing
    sheet_name and optional data_start_row/data_end_row (1-based, inclusive).
    Omitted bounds inherit the function's data_start_row/data_end_row settings.
    Each selection may also override header_row (1-based cytokine-name row),
    start_col (Excel column letter), and
    obs_conc_offset (integer column distance from cytokine name to concentration).
    Omitted column settings inherit the function-level defaults. All other
    column settings are shared. Cytokines align by name, not column position;
    pooling and scaling use the combined data from every selected sheet.
    Numeric or duplicate cytokine headers are rejected for combined input.
    Repeat a sheet to select
    disjoint ranges; overlapping ranges are rejected to prevent double-counting.
    Matching sample/cytokine names are pooled across all selected rows before
    scaling. Each valid row has equal weight, not each sheet. Multi-sheet long_df
    includes Sheet and Excel_Row provenance. Without sheet_selections, the
    existing single-sheet behavior and returned columns are unchanged.

    cytokines=None includes all cytokines. Otherwise supply a nonempty list
    of exact, case-sensitive header names (e.g. ["IL-6", "IL-10", "TNFα"]).
    Only those cytokines enter averaging/scaling/clustering and returned long_df.
    Unknown names or selected cytokines with no numeric data raise ValueError.
    With cluster_rows=False, rows follow the list order; otherwise they cluster.

    samples=None includes all samples. Otherwise supply a nonempty list of
    exact, case-sensitive sample names. Both filters apply before averaging,
    scaling, and clustering, and both returned tables contain only this subset.
    With preserve_sample_order=True, columns follow samples list order unless
    sample_order supplies a permutation of that list. X-axis labels show the
    number of distinct Rep IDs with numeric data for each sample. Unknown names or
    samples with no numeric data for the selected cytokines raise ValueError.
    For log2fc, ref_sample must be included. Row z-scores require at least two
    included samples and nonzero variance for a cytokine to remain in the plot.
    """
    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from openpyxl.utils import column_index_from_string

    def _validate_name_list(names, label):
        if names is None:
            return None
        if not isinstance(names, list) or not names:
            raise ValueError(f"{label} must be a nonempty list of names, or None.")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError(f"Each name in {label} must be a nonempty string.")
        names = [name.strip() for name in names]
        if len(set(names)) != len(names):
            raise ValueError(f"{label} must not contain duplicate names.")
        return names

    cytokines = _validate_name_list(cytokines, "cytokines")
    samples = _validate_name_list(samples, "samples")
    if run_statistics:
        if not isinstance(control_sample, str) or not control_sample.strip():
            raise ValueError("control_sample must be a nonempty sample name.")
        control_sample = control_sample.strip()
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
    if samples is not None and sample_order is not None:
        sample_order = _validate_name_list(sample_order, "sample_order")
        if set(sample_order) != set(samples):
            raise ValueError("When samples is set, sample_order must contain exactly the same sample names.")

    # ---- Resolve scale_mode (backward-compatible with use_log2fc) ----
    if scale_mode is None:
        scale_mode = "log2fc" if use_log2fc else "none"
    valid_modes = {"none", "log2fc", "zscore_row"}
    if scale_mode not in valid_modes:
        raise ValueError(f"scale_mode must be one of {valid_modes}, got '{scale_mode}'.")

    # ---- Resolve output directory ----
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(file_path)), "Cytokine Heatmaps")
    os.makedirs(output_dir, exist_ok=True)

    def _parse_value(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.number)):
            return float(v)
        s = str(v).strip()
        if re.match(r"^OOR\s*<", s, flags=re.IGNORECASE):
            return 0.0
        try:
            return float(s)
        except ValueError:
            return np.nan

    records = []
    multi_sheet = sheet_selections is not None
    censored_observations = set()
    selections = sheet_selections if multi_sheet else [{"sheet_name": sheet_name}]
    if not isinstance(selections, list) or not selections:
        raise ValueError("sheet_selections must be a nonempty list of dictionaries.")
    allowed_keys = {"sheet_name", "data_start_row", "data_end_row", "start_col", "obs_conc_offset", "header_row"}
    for selection in selections:
        if not isinstance(selection, dict) or "sheet_name" not in selection:
            raise ValueError("Each sheet selection must be a dictionary containing sheet_name.")
        if set(selection) - allowed_keys:
            raise ValueError(f"Unknown sheet selection keys: {set(selection) - allowed_keys}")

    sample_col_idx = column_index_from_string(sample_col) - 1
    rep_col_idx = column_index_from_string(rep_col) - 1
    if col_step <= 0:
        raise ValueError("col_step must be positive.")
    selected_rows = set()
    source_tags = []
    with pd.ExcelFile(file_path) as workbook:
        sheet_cache = {}
        for selection in selections:
            selected_sheet = selection["sheet_name"]
            if isinstance(selected_sheet, int) and not isinstance(selected_sheet, bool):
                if not 0 <= selected_sheet < len(workbook.sheet_names):
                    raise ValueError(f"Invalid sheet index: {selected_sheet}")
                selected_sheet = workbook.sheet_names[selected_sheet]
            if not isinstance(selected_sheet, str) or selected_sheet not in workbook.sheet_names:
                raise ValueError(f"Unknown sheet {selected_sheet!r}; available: {workbook.sheet_names}")
            selected_start_col = selection.get("start_col", start_col)
            selected_offset = selection.get("obs_conc_offset", obs_conc_offset)
            selected_header_row = selection.get("header_row", header_row)
            if not isinstance(selected_header_row, int) or isinstance(selected_header_row, bool) or selected_header_row < 1:
                raise ValueError(f"header_row must be a positive integer in sheet {selected_sheet!r}.")
            header_row_idx = selected_header_row - 1
            if not isinstance(selected_start_col, str) or not re.fullmatch(r"[A-Za-z]+", selected_start_col):
                raise ValueError(f"start_col must be an Excel column letter in sheet {selected_sheet!r}.")
            start_col_idx = column_index_from_string(selected_start_col) - 1
            if not isinstance(selected_offset, int) or isinstance(selected_offset, bool):
                raise ValueError(f"obs_conc_offset must be an integer in sheet {selected_sheet!r}.")
            if selected_sheet not in sheet_cache:
                sheet_cache[selected_sheet] = pd.read_excel(workbook, sheet_name=selected_sheet, header=None)
            raw = sheet_cache[selected_sheet]
            n_rows, n_cols = raw.shape
            if not 0 <= header_row_idx < n_rows or sample_col_idx >= n_cols:
                raise ValueError(f"Invalid header_row or sample_col for sheet {selected_sheet!r}.")
            cytokine_names = {}
            for col_idx in range(start_col_idx, n_cols, col_step):
                name = raw.iat[header_row_idx, col_idx]
                if pd.notna(name) and str(name).strip():
                    if multi_sheet and (isinstance(name, (int, float, np.number))
                                        or re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(name).strip())):
                        raise ValueError(f"Numeric cytokine header {name!r} in sheet {selected_sheet!r}, row {selected_header_row}. "
                                         "Set this selection's header_row to the row containing cytokine names.")
                    cytokine_names[col_idx] = str(name).strip()
            if not cytokine_names:
                raise ValueError(f"No cytokine names found in sheet {selected_sheet!r} at row {selected_header_row}.")
            if multi_sheet and len(set(cytokine_names.values())) != len(cytokine_names):
                raise ValueError(f"Duplicate cytokine names in sheet {selected_sheet!r}, row {selected_header_row}; check header/column settings.")
            print(f"Sheet {selected_sheet!r}: found {len(cytokine_names)} cytokine(s): {list(cytokine_names.values())}")

            start_row = selection.get("data_start_row", data_start_row)
            end_row = selection.get("data_end_row", data_end_row)
            start_row = selected_header_row + 1 if start_row is None else start_row
            end_row = n_rows if end_row is None else end_row
            for bound in (start_row, end_row):
                if not isinstance(bound, int) or isinstance(bound, bool):
                    raise ValueError(f"Row bounds must be integers in sheet {selected_sheet!r}.")
            if not 1 <= start_row <= n_rows or end_row < start_row:
                raise ValueError(f"Invalid rows {start_row}-{end_row} in sheet {selected_sheet!r} ({n_rows} rows).")
            if multi_sheet and end_row > n_rows:
                raise ValueError(f"End row {end_row} exceeds sheet {selected_sheet!r} ({n_rows} rows).")
            end_row = min(end_row, n_rows)
            row_keys = {(selected_sheet, row) for row in range(start_row, end_row + 1)}
            if selected_rows & row_keys:
                raise ValueError(f"Overlapping row selections in sheet {selected_sheet!r}.")
            selected_rows.update(row_keys)
            source_tags.append(re.sub(r"[^A-Za-z0-9]+", "_", selected_sheet)
                               + f"_rows{start_row}-{end_row}_header{selected_header_row}_start{selected_start_col.upper()}_offset{selected_offset}")
            print(f"Reading sheet {selected_sheet!r}, rows {start_row}-{end_row} (1-indexed, inclusive)")
            records_before = len(records)
            for row_idx in range(start_row - 1, end_row):
                samp = raw.iat[row_idx, sample_col_idx]
                rep = raw.iat[row_idx, rep_col_idx] if rep_col_idx < n_cols else None
                if pd.isna(samp) or not str(samp).strip():
                    continue
                samp = str(samp).strip()
                rep = str(rep).strip() if pd.notna(rep) else None
                for col_idx, cytokine in cytokine_names.items():
                    data_col_idx = col_idx + selected_offset
                    if not 0 <= data_col_idx < n_cols:
                        if multi_sheet:
                            raise ValueError(f"Missing concentration column for {cytokine!r} in sheet {selected_sheet!r}.")
                        continue
                    val = _parse_value(raw.iat[row_idx, data_col_idx])
                    if re.match(r"^OOR\s*<", str(raw.iat[row_idx, data_col_idx]).strip(), re.I):
                        censored_observations.add((samp, rep, cytokine))
                    record = {"Sample": samp, "Rep": rep, "Cytokine": cytokine, "Obs_Conc": val}
                    if multi_sheet:
                        record.update(Sheet=selected_sheet, Excel_Row=row_idx + 1)
                    records.append(record)
            if multi_sheet and len(records) == records_before:
                raise ValueError(f"No data parsed from sheet {selected_sheet!r}, rows {start_row}-{end_row}.")
            if multi_sheet and not any(pd.notna(record["Obs_Conc"]) for record in records[records_before:]):
                raise ValueError(f"No numeric concentrations in sheet {selected_sheet!r}, rows {start_row}-{end_row}.")

    long_df = pd.DataFrame(records)
    if long_df.empty:
        raise ValueError("No data rows were parsed. Check row/column settings.")

    if samples is not None:
        available_samples = sorted(long_df["Sample"].unique())
        unknown_samples = [name for name in samples if name not in available_samples]
        if unknown_samples:
            raise ValueError(f"Unknown samples: {unknown_samples}. Available samples: {available_samples}")
        long_df = long_df.loc[long_df["Sample"].isin(samples)].reset_index(drop=True)
        print(f"Analyzing selected samples: {samples}")

    if cytokines is not None:
        available = sorted(long_df["Cytokine"].unique())
        unknown = [name for name in cytokines if name not in available]
        if unknown:
            raise ValueError(f"Unknown cytokines: {unknown}. Available cytokines: {available}")
        long_df = long_df.loc[long_df["Cytokine"].isin(cytokines)].reset_index(drop=True)
        numeric_names = set(long_df.loc[long_df["Obs_Conc"].notna(), "Cytokine"])
        missing = [name for name in cytokines if name not in numeric_names]
        if missing:
            raise ValueError(f"Selected cytokines have no numeric concentrations in the selected rows: {missing}")
        print(f"Analyzing selected cytokines: {cytokines}")

    if samples is not None:
        numeric_samples = set(long_df.loc[long_df["Obs_Conc"].notna(), "Sample"])
        missing_samples = [name for name in samples if name not in numeric_samples]
        if missing_samples:
            raise ValueError(f"Selected samples have no numeric concentrations for the included cytokines: {missing_samples}")
        if scale_mode == "zscore_row" and len(samples) < 2:
            raise ValueError("Row z-scores require at least two included samples; use scale_mode='none' for one sample.")

    print(f"Parsed {len(long_df)} data points across {long_df['Sample'].nunique()} sample(s) "
          f"and {long_df['Cytokine'].nunique()} cytokine(s).")

    long_df_valid = long_df.dropna(subset=["Obs_Conc"])
    replicate_counts = long_df_valid.groupby("Sample")["Rep"].nunique()

    # ---- Collapse bio reps -> mean per Sample x Cytokine ----
    mean_df = (
        long_df_valid
        .groupby(["Cytokine", "Sample"], as_index=False)["Obs_Conc"]
        .mean()
    )

    # Pivot to Cytokine (rows) x Sample (cols)
    heatmap_data = mean_df.pivot(index="Cytokine", columns="Sample", values="Obs_Conc")
    if cytokines is not None:
        heatmap_data = heatmap_data.reindex(cytokines)

    # ---- Determine sample order ----
    if sample_order is None:
        original_sample_order = samples if samples is not None else list(dict.fromkeys(long_df["Sample"]))
    else:
        original_sample_order = sample_order

    if run_statistics and control_sample not in heatmap_data.columns:
        raise ValueError(f"Control {control_sample!r} must be included in the selected samples; otherwise set run_statistics=False.")

    # =========================================================================
    # Apply scaling mode: none / log2fc / zscore_row
    # =========================================================================
    if scale_mode == "log2fc":
        if ref_sample is None or ref_sample not in heatmap_data.columns:
            raise ValueError(
                f"ref_sample '{ref_sample}' not found. Available samples: {list(heatmap_data.columns)}"
            )
        ref_vals = heatmap_data[ref_sample].astype(float)
        heatmap_data = np.log2((heatmap_data.astype(float) + pseudocount).div(ref_vals + pseudocount, axis=0))
        heatmap_data[ref_sample] = 0.0
        heatmap_title = f"log2 fold-change vs {ref_sample}"
        value_tag = "log2FC"

    elif scale_mode == "zscore_row":
        # z = (value - row_mean) / row_std, computed independently per cytokine (row)
        row_means = heatmap_data.mean(axis=1, skipna=True)
        row_stds = heatmap_data.std(axis=1, skipna=True, ddof=1)

        # Avoid divide-by-zero for cytokines with no variance across samples
        row_stds_safe = row_stds.replace(0, np.nan)

        heatmap_data = heatmap_data.sub(row_means, axis=0).div(row_stds_safe, axis=0)

        n_flat = row_stds.eq(0).sum()
        if n_flat > 0:
            print(f"  ⚠ {n_flat} cytokine(s) had zero variance across samples "
                  f"(all values equal) and could not be z-scored; they will appear blank (NaN).")

        heatmap_title = "Row Z-score (per cytokine, across samples)"
        value_tag = "zscoreRow"

    else:  # "none"
        heatmap_title = "Raw values (mean of bio reps)"
        value_tag = "raw"

    # ---- NEW: Drop any rows that are entirely NaN, and fill remaining scattered NaNs ----
    # (scipy's hierarchical clustering cannot handle NaN/inf values)
    n_rows_before = heatmap_data.shape[0]
    heatmap_data = heatmap_data.dropna(axis=0, how="all")  # drop fully-NaN cytokine rows
    n_dropped = n_rows_before - heatmap_data.shape[0]
    if n_dropped > 0:
        print(f"  ⚠ Dropped {n_dropped} cytokine row(s) that were entirely NaN "
              f"(zero variance or all missing data) before clustering.")

    if heatmap_data.isna().any().any():
        n_remaining_na = int(heatmap_data.isna().sum().sum())
        print(f"  ⚠ {n_remaining_na} remaining NaN cell(s) found; filling with 0 "
              f"to allow clustering (these will appear as the midpoint color).")
        heatmap_data = heatmap_data.fillna(0)

    if heatmap_data.shape[0] == 0:
        raise ValueError("No cytokine rows remain after dropping all-NaN rows. "
                          "Check your data range and scale_mode settings.")

    # ---- Sample (column) order ----
    if preserve_sample_order:
        ordered_cols = [s for s in original_sample_order if s in heatmap_data.columns]
        heatmap_data = heatmap_data[ordered_cols]
        cluster_cols = False
        order_tag = "_ORIGINAL_ORDER"
        print("Using ORIGINAL sample order")
    else:
        cluster_cols = True
        order_tag = "_CLUSTERED"
        print("Using HIERARCHICAL CLUSTERING for sample order")

    # ---- Output file paths ----
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    sheet_tag = re.sub(r"[^A-Za-z0-9]+", "_", str(sheet_name))
    row_tag = f"_rows{data_start_row}-{data_end_row}" if (data_start_row or data_end_row) else ""
    base_tag = f"{base_name}__{sheet_tag}{row_tag}"
    if multi_sheet:
        import hashlib
        selection_id = hashlib.sha256(repr(source_tags).encode("utf-8")).hexdigest()[:10]
        base_tag = f"{base_name}__combined_{len(selections)}selections_{selection_id}"
        print("Combined sources: " + "; ".join(source_tags))
    if cytokines is not None:
        import hashlib
        cytokine_id = hashlib.sha256(repr(cytokines).encode("utf-8")).hexdigest()[:10]
        base_tag += f"__selected{len(cytokines)}_{cytokine_id}"
    if samples is not None:
        import hashlib
        sample_id = hashlib.sha256(repr(samples).encode("utf-8")).hexdigest()[:10]
        base_tag += f"__samples{len(samples)}_{sample_id}"

    heatmap_png = os.path.join(output_dir, f"{base_tag}__heatmap_{value_tag}{order_tag}.png")

    # ---- Color scale ----
    # For z-score, cap the symmetric range at ±3 SD (conventional) unless data range is smaller.
    if scale_mode == "zscore_row":
        finite_vals = heatmap_data.values[np.isfinite(heatmap_data.values)]
        lim = np.nanmax(np.abs(finite_vals)) if finite_vals.size > 0 else 3.0
        lim = min(lim, 3.0)
    else:
        lim = np.nanmax(np.abs(heatmap_data.values))

    if lim == 0 or np.isnan(lim):
        lim = 1.0

    g = sns.clustermap(
        heatmap_data,
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        vmin=-lim,
        vmax=lim,
        center=0,
        row_cluster=cluster_rows and heatmap_data.shape[0] > 1,
        col_cluster=cluster_cols and heatmap_data.shape[1] > 1,
        figsize=(max(8, heatmap_data.shape[1] * 0.6), max(6, heatmap_data.shape[0] * 0.3)),
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(1.02, 0.3, 0.03, 0.4),  # moved to the right of the plot
    )

    sample_labels = [
        f"{label.get_text()}\n(n={replicate_counts.get(label.get_text(), 0)})"
        for label in g.ax_heatmap.get_xticklabels()
    ]
    g.ax_heatmap.set_xticklabels(
        sample_labels, rotation=45, ha="right", fontsize=fontsize_col
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=fontsize_row
    )
    colorbar_title = {
        "none": "Mean conc.",
        "log2fc": "log2FC",
        "zscore_row": "Z-score",
    }[scale_mode]
    g.cax.set_ylabel("")
    g.cax.set_title(colorbar_title, fontsize=fontsize_col, pad=8)
    g.fig.suptitle(heatmap_title, y=1.02, fontsize=14)

    g.savefig(heatmap_png, dpi=200, bbox_inches="tight")
    plt.close(g.fig)

    print(f"Heatmap exported to:\n  {heatmap_png}")

    bar_output_dir = os.path.join(output_dir, "Raw Bar Plots")
    os.makedirs(bar_output_dir, exist_ok=True)
    bar_samples = list(g.data2d.columns)
    bar_cytokines = cytokines if cytokines is not None else list(mean_df["Cytokine"].unique())
    if run_statistics:
        print("Exploratory Kruskal-Wallis / Dunn tests: independent groups, >=2 numeric bio reps; asymptotic p-values may be inaccurate at n=2-3.")
        bio_points, kw_results, dunn_results = analyze_heatmap_groups(
            long_df, bar_samples, bar_cytokines, control_sample, alpha, censored_observations,
        )
        statistics_path = Path(bar_output_dir) / f"{base_tag}__Kruskal_Wallis_Results.xlsx"
        export_heatmap_statistics(
            statistics_path, long_df, bio_points, kw_results, dunn_results, control_sample, alpha,
        )
        print(f"Statistics exported to:\n  {statistics_path}")
    x_positions = np.arange(len(bar_samples))
    for index, cytokine in enumerate(bar_cytokines, start=1):
        observations = long_df_valid.loc[long_df_valid["Cytokine"].eq(cytokine)]
        summary = observations.groupby("Sample")["Obs_Conc"].agg(["mean", "std"])
        summary = summary.reindex(bar_samples)
        counts = observations.groupby("Sample")["Rep"].nunique()
        fig, ax = plt.subplots(figsize=(max(8, len(bar_samples)), 5))
        try:
            ax.bar(x_positions, summary["mean"], color="skyblue", edgecolor="black", zorder=1)
            has_sd = summary["std"].notna().to_numpy()
            if has_sd.any():
                ax.errorbar(x_positions[has_sd], summary.loc[has_sd, "mean"],
                            yerr=summary.loc[has_sd, "std"], fmt="none",
                            color="black", capsize=4, zorder=2)
            for x, sample in enumerate(bar_samples):
                values = observations.loc[observations["Sample"].eq(sample), "Obs_Conc"].to_numpy()
                offsets = np.linspace(-0.15, 0.15, len(values)) if len(values) > 1 else np.zeros(len(values))
                ax.scatter(x + offsets, values, color="black", s=30, zorder=3)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"{sample}\n(n={counts.get(sample, 0)})" for sample in bar_samples],
                               rotation=45, ha="right", fontsize=fontsize_col)
            ax.set_ylabel("Obs. Conc. (mean +/- SD)")
            ax.set_title(cytokine)
            ax.axhline(0, color="0.7", linewidth=0.7, zorder=0)
            if run_statistics:
                annotate_heatmap_bar(
                    ax, cytokine, bar_samples, kw_results, dunn_results, control_sample,
                )
            fig.tight_layout()
            safe_name = re.sub(r"[^\w-]+", "_", cytokine).strip("_")
            bar_path = os.path.join(bar_output_dir, f"{base_tag}__bar_{index:02d}_{safe_name}_raw.png")
            fig.savefig(bar_path, dpi=300, bbox_inches="tight")
        finally:
            plt.close(fig)
    print(f"Exported {len(bar_cytokines)} raw-concentration bar plots to:\n  {bar_output_dir}")

    return heatmap_data, long_df


if __name__ == "__main__":
    # ================================================================================
    # EXAMPLE USAGE
    # ================================================================================
    file_path = "/Users/allisonpickle/Desktop/261007_Cytokine Reanalysis/261118_Compiled_Cytokines_3BioReps.xlsx"  # <-- CHANGE ME

    # Set to None to include every cytokine, or provide a list to select a subset.
    cytokines_to_plot = ["Fractalkine", "IL-1α", "IL-1β", "IL-6", "IL-10", "TNFα"]
    # cytokines_to_plot = None
    heatmap_data, long_df = plot_cytokine_heatmap(
        file_path,
        sheet_selections=[
            {
                "sheet_name": "261018 Updated Analysis",
                "header_row": 1,
                "data_start_row": 194,
                "data_end_row": 235,
                "start_col": "F",
                "obs_conc_offset": 1,
            },
        ],
        preserve_sample_order=True,
        scale_mode="zscore_row",
        run_statistics=True, control_sample="6hr Ctrl", alpha=0.05,
        cytokines=cytokines_to_plot,
        samples=[
            "6hr Ctrl",
            # "6hr 5ng",
            # "6hr 100ng",
            # "24hr 5ng",
            "24hr 100ng",
            # "48hr 5ug",
            # "72hr 5 ng",
            # "72hr 100ng",

            # "10w3",
            # "10 uM w3, 100ng LPS 24hr"
            "2mM PD",
            "2mM PD, 100ng LPS 24hr"
        ],
        cluster_rows=True,
        sample_col="B",
        rep_col="A",
        header_row=16,
        start_col="F",
        col_step=3,
        obs_conc_offset=1

    )
